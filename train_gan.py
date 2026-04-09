import yaml
import torch
import utils
import argparse
import models
import losses
import datasets
import train_funcs 
import os
import csv
import json
import torch.distributed as dist
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import shutil
from train_funcs.train_utils import HyperAttentionLoss
import gc

DEBUG = os.getenv("DEBUG", "True").lower() == "true"
if DEBUG:
    if "CUDA_VISIBLE_DEVICES" not in os.environ: os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RANK"] = "0"; os.environ["WORLD_SIZE"] = "1"; os.environ["LOCAL_RANK"] = "0"
    device_id = 0
else:
    if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group("nccl")
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
torch.backends.cudnn.benchmark = True 

def is_main_process():
    if not dist.is_available() or not dist.is_initialized(): return True
    return dist.get_rank() == 0

def log_to_csv(save_path, epoch, train_loss, val_loss, accuracy, lr):
    csv_file = save_path / 'loss_log.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists: writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'LR'])
        writer.writerow([epoch, train_loss, val_loss, accuracy, lr])

def make_dataloader(spec, tag='', save_path=None):
    dataset = datasets.make(spec['dataset'])
    wrapper_args = {'dataset': dataset, 'corners_only': False} 
    dataset = datasets.make(spec['wrapper'], args=wrapper_args)
    
    sampler = None
    shuffle = True
    
    # if tag == 'train':
    if False:
        # 1. Generate the Hard-Mining Weights using the REAL path
        stats_path = save_path / 'confusion_stats.json' if save_path else Path('outputs') / 'confusion_stats.json'
        weights = get_confusion_weights(dataset, stats_path)
        
        if not DEBUG:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False
        else:
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
            shuffle = False

    loader = DataLoader(
        dataset, batch_size=spec['batch'], shuffle=(tag == 'train'), 
        num_workers=16, pin_memory=True, 
        collate_fn=dataset.collate_fn, drop_last=(tag == 'train'), prefetch_factor=4
    )
    return loader, sampler

def get_confusion_weights(dataset, stats_path, top_k=4):
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f: 
                confusions = json.load(f)
            # Python dictionaries preserve order. The JSON is already sorted!
            worst_keys = list(confusions.keys())[:top_k]
            trouble_chars = set(worst_keys)
            
            if is_main_process():
                print(f"🔪 SURGICAL MINING: Filtering down to the Top {top_k} worst characters: {trouble_chars}")
        except: 
            trouble_chars = set()
    else: 
        trouble_chars = set()

    weights = []
    found_targets = 0
    
    # Fast iteration through the dataset
    for item in dataset.dataset:
        if any(char in trouble_chars for char in item['gt']):
            weights.append(5.0)
            found_targets += 1
        else: 
            weights.append(1.0)
    
    if is_main_process() and found_targets > 0:
        print(f"🎯 HARD MINING ACTIVE: Boosted {found_targets} difficult plates out of {len(weights)}!")
        
    # Safety Valve: If even the top 4 characters still infect > 50% of the dataset,
    # gently reduce the multiplier so the optimizer doesn't collapse.
    if len(weights) > 0 and (found_targets / len(weights)) > 0.5:
        if is_main_process(): 
            print(f"⚠️ High Error Density ({found_targets/len(weights):.1%}). Dilating sampler down to 2.5x.")
        weights = [w * 0.5 if w > 1.0 else w for w in weights]

    return torch.DoubleTensor(weights)

def create_scheduler(optimizer, epoch_max):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-6)

def create_teacher_scheduler(optimizer):
    # Patience is only 2! It will quickly drop the LR when the Teacher hits 99%
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)

def main(config, save_path):
    local_rank = int(os.environ["LOCAL_RANK"])
    epoch_max = config['epoch_max']
    
    val_loader, _ = make_dataloader(config.get('val_dataset', config['train_dataset']), tag='val', save_path=save_path)

    if is_main_process(): print("Creating Student VSR Model...")
    model_g = models.make(config['model_g']).to(local_rank)
    model_g = model_g.to(memory_format=torch.channels_last)
    
    # --- UPDATED: Optimizer Setup (WITH LAYOUT MULTIPLIER) ---
    base_lr = float(config['optimizer_sr']['args']['lr'])
    
    # We will separate the parameters into two lists
    deform_offset_params = []
    base_params = []
    
    for name, param in model_g.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if this parameter belongs to an offset predictor
        if 'offset_conv' in name:
            deform_offset_params.append(param)
        else:
            base_params.append(param)
            
    # Create the parameter groups
    optim_groups = [
        {'params': base_params, 'lr': base_lr},
        {'params': deform_offset_params, 'lr': base_lr},
    ]
    
    if is_main_process(): 
        print(f"Base params: {len(base_params)} | Deform: {len(deform_offset_params)}")
        print(f"Base LR: {base_lr} | Deform LR: {base_lr} ")

    optimizer_g = torch.optim.Adam(optim_groups)
    loss_fn_spread = HyperAttentionLoss(grid_h=12, grid_w=36).to(local_rank)
    hyper_groups = [
        {'params': [loss_fn_spread.v_stretch, loss_fn_spread.iris_scale], 'lr': 1e-2},
        {'params': [loss_fn_spread.monotonic_scale, 
                    loss_fn_spread.boundary_scale, 
                    loss_fn_spread.ortho_scale], 'lr': 1e-2}
    ]
    optimizer_hyper = torch.optim.Adam(hyper_groups)
    # -----------------------------------------------------------
    
    # ==========================================
    # NEW: CONDITIONAL TEACHER SETUP
    # ==========================================
    use_distillation = config.get('use_distillation', False)
    model_t, optimizer_t, scheduler_t = None, None, None

    if use_distillation:
        if is_main_process(): print("🎓 Distillation Mode ON: Creating Teacher VSR Model...")
        model_t = models.make(config['model_g']).to(local_rank)
        model_t = model_t.to(memory_format=torch.channels_last)

        teacher_lr = base_lr * config.get('teacher_lr_multiplier', 2.0)
        
        # Reuse your group splitting logic here
        optimizer_t = torch.optim.Adam([
            {'params': [p for n, p in model_t.named_parameters() if 'offset_conv' not in n and p.requires_grad], 'lr': teacher_lr},
            {'params': [p for n, p in model_t.named_parameters() if 'offset_conv' in n and p.requires_grad], 'lr': teacher_lr * 10.0}
        ])
        scheduler_t = create_teacher_scheduler(optimizer_t)
        
        if is_main_process(): print(f"Teacher Base LR: {teacher_lr:.2e}")
    # ==========================================

    # --- 3. Initialize Student Scheduler ---
    scheduler_g = create_scheduler(optimizer_g, epoch_max)
    # --- 4. Resume Logic ---
    start_epoch = 1; best_accuracy = 0.0
    
    # Initialize state variables here so they can be overwritten by the checkpoint
    best_models = [] 
    early_stop_patience = config.get('early_stop_patience', 50)
    epochs_without_improvement = 0
    
    resume_path = config.get('resume')
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=f'cuda:{local_rank}', weights_only=False)
        
        # Smart Filter for Teacherless transition
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_g_sd'].items()}
        model_dict = model_g.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        model_g.load_state_dict(pretrained_dict, strict=False)
        
        if len(model_dict) != len(pretrained_dict):
            if is_main_process(): print(f"♻️ Architecture Mismatch (Likely removed Teacher). Resetting Optimizer.")
        else:
            if is_main_process(): print("✅ Perfect Match. Full Resume.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_accuracy = checkpoint.get('best_acc', 0.0)
            
            # --- NEW: Safely resume the tracker and patience ---
            best_models = checkpoint.get('best_models', [])
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            
            if is_main_process() and epochs_without_improvement > 0:
                print(f"♻️ Resumed Early Stopping counter at {epochs_without_improvement}/{early_stop_patience} epochs.")
            if is_main_process() and len(best_models) > 0:
                print(f"♻️ Resumed Top-{len(best_models)} models tracker.")
            
            if 'optimizer_hyper' in checkpoint and 'loss_hyper_sd' in checkpoint:
                if is_main_process(): print("✅ Resuming Hyper-Weights.")
                loss_fn_spread.load_state_dict(checkpoint['loss_hyper_sd'], strict=False)
                
                try:
                    optimizer_hyper.load_state_dict(checkpoint['optimizer_hyper'])
                    if is_main_process(): print("✅ Resuming Hyper-Optimizer State.")
                except ValueError:
                    if is_main_process(): print("⚠️ Hyper-Optimizer state mismatch. Starting fresh optimizer for hyper-weights.")
            if 'optimizer_g' in checkpoint: 
                try: 
                    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
                    
                    if config.get('force_lr', False):
                        if is_main_process(): 
                            print(f"🔥 FORCE_LR is True: Overriding loaded LR to {base_lr} and resetting Scheduler.")
                        
                        optimizer_g.param_groups[0]['lr'] = base_lr          
                        if len(optimizer_g.param_groups) > 1:
                            optimizer_g.param_groups[1]['lr'] = base_lr * 10.0
                            
                        scheduler_g = create_scheduler(optimizer_g, epoch_max)
                    else:
                        if 'scheduler_g' in checkpoint:
                            if is_main_process(): print("✅ Loading previous Scheduler state.")
                            scheduler_g.load_state_dict(checkpoint['scheduler_g'])
                    
                except Exception as e:
                    if is_main_process(): print(f"⚠️ Optimizer/Scheduler resume failed: {e}")

            # Safely resume Teacher if distillation is active
            if use_distillation:
                if 'model_t_sd' in checkpoint:
                    model_t.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_t_sd'].items()}, strict=False)
                
                if 'optimizer_t' in checkpoint: 
                    optimizer_t.load_state_dict(checkpoint['optimizer_t'])
                    
                    # ---> THE FIX: Force LR for the Teacher too!
                    if config.get('force_lr', False):
                        teacher_lr = base_lr * config.get('teacher_lr_multiplier', 2.0)
                        if is_main_process(): 
                            print(f"🔥 FORCE_LR is True: Overriding Teacher LR to {teacher_lr:.2e}")
                        
                        optimizer_t.param_groups[0]['lr'] = teacher_lr
                        if len(optimizer_t.param_groups) > 1:
                            optimizer_t.param_groups[1]['lr'] = teacher_lr * 10.0
                            
                        scheduler_t = create_teacher_scheduler(optimizer_t)
                    else:
                        if 'scheduler_t' in checkpoint: 
                            scheduler_t.load_state_dict(checkpoint['scheduler_t'])
                    
                # ---> NEW: Restore the freeze state!
                teacher_frozen = checkpoint.get('teacher_frozen', False)
                teacher_best_acc = checkpoint.get('teacher_best_acc', 0.0)
                teacher_patience_counter = checkpoint.get('teacher_patience_counter', 0)
                
                if is_main_process() and teacher_frozen:
                    print("🧊 Resumed with a FROZEN Teacher Oracle.")

    if not DEBUG: 
        model_g = DDP(model_g, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if use_distillation:
            model_t = DDP(model_t, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
    loss_fn = losses.make(config['loss']).to(local_rank)
    train_step = train_funcs.make(config['func_train']) 
    val_step = train_funcs.make(config['func_val'])

    stats_path = save_path / 'confusion_stats.json'

    # --- DASHBOARD & LOG INITIALIZATION ---
    if is_main_process():
        # 1. Initialize loss_log.csv with an "Epoch 0" row so the LR shows up immediately
        csv_file = save_path / 'loss_log.csv'
        if not os.path.isfile(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'LR'])
                # Pre-seed Epoch 0 so the dashboard has immediate data
                writer.writerow([0, 0.0, 0.0, 0.0, base_lr]) 

        # 2. Deploy Dashboard Template
        viz_dir = save_path / 'train_features'
        os.makedirs(viz_dir, exist_ok=True)
        
        template_src = Path(__file__).parent / 'monitor_template.html'
        template_dst = viz_dir / 'live_monitor.html'
        
        if template_src.exists():
            shutil.copy(template_src, template_dst)
            print(f"🚀 Dashboard deployed to: {template_dst}")
        else:
            print("⚠️ Warning: monitor_template.html not found. Dashboard not deployed.")

        # 3. NEW: Pre-seed hyper.txt with initial values to prevent UI placeholders
        hyper_path = viz_dir / 'hyper.txt'
        with open(hyper_path, 'w') as f:
            f.write("V: 1.20 | M: 15.0 | B: 20.0 | O: 5.0")

    teacher_frozen = False 
    teacher_best_acc = 0.0           # <--- NEW
    teacher_patience_counter = 0     # <--- NEW
    
    checkpoint = {}
    try:
        for epoch in range(start_epoch, epoch_max + 1):
            
            train_loader, train_sampler = make_dataloader(config['train_dataset'], tag='train', save_path=save_path)
            if hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(epoch)
            if is_main_process(): print(f"🔄 Starting Epoch {epoch}...")
            
            model_g.train()
            active_optimizer_t = optimizer_t 
            
            # APPLY ORACLE FREEZE (If triggered)
            if use_distillation:
                if teacher_frozen:
                    # ---> THE FIX: Keep it in train()!
                    # This prevents the KL Divergence from exploding by maintaining 
                    # the exact same dropout distribution the student is used to.
                    model_t.train() 
                    for param in model_t.parameters():
                        param.requires_grad = False 
                    active_optimizer_t = None 
                else:
                    model_t.train()
            
            train_out = train_step(
                train_loader, val_loader,
                model_g, model_t,         
                optimizer_g, active_optimizer_t, # <--- Pass the dynamic optimizer
                optimizer_hyper, None, config,          
                epoch=epoch, save_path=save_path
            )
            teacher_train_acc = 0.0
            if isinstance(train_out, tuple):
                train_loss, teacher_train_acc = train_out
            else:
                train_loss = train_out
            val_loss, accuracy, teacher_val_acc = val_step(val_loader, model_g, model_t, loss_fn, config, save_path=save_path)

            # ==========================================
            # THE FIX: TRACK PATIENCE USING HR VAL ACCURACY
            # ==========================================
            if use_distillation:
                if not teacher_frozen:
                    if teacher_val_acc > teacher_best_acc:
                        teacher_best_acc = teacher_val_acc
                        teacher_patience_counter = 0
                    else:
                        teacher_patience_counter += 1
                        
                    freeze_patience = config.get('teacher_freeze_patience', 20)
                    
                    if teacher_patience_counter > freeze_patience:
                        if is_main_process():
                            print(f"🥶 TEACHER FREEZE TRIGGERED: {freeze_patience} VAL epochs without improvement (Peaked at {teacher_best_acc:.4f}).")
                        teacher_frozen = True

            if scheduler_g: 
                if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau): 
                    # Student uses Student Validation Accuracy
                    scheduler_g.step(accuracy)
                    
                    # Teacher uses Teacher Training Accuracy
                    if use_distillation and scheduler_t is not None: 
                        scheduler_t.step(teacher_train_acc)
                else: 
                    scheduler_g.step()
                    if use_distillation and scheduler_t is not None: 
                        scheduler_t.step()
    
            if is_main_process():
                current_lr = optimizer_g.param_groups[0]['lr']
                print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.2e}")
                log_to_csv(save_path, epoch, train_loss, val_loss, accuracy, current_lr)
                
                checkpoint = {
                    'epoch': epoch, 
                    'model_g_sd': model_g.module.state_dict() if hasattr(model_g, 'module') else model_g.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(), 
                    'optimizer_hyper': optimizer_hyper.state_dict(), 
                    'loss_hyper_sd': loss_fn_spread.state_dict(),    
                    'scheduler_g': scheduler_g.state_dict(),
                    'best_acc': best_accuracy,
                    'best_models': best_models,
                    'epochs_without_improvement': epochs_without_improvement
                }
                
                # Safely inject Teacher states into the dict
                if use_distillation:
                    checkpoint['model_t_sd'] = model_t.module.state_dict() if hasattr(model_t, 'module') else model_t.state_dict()
                    checkpoint['optimizer_t'] = optimizer_t.state_dict()
                    checkpoint['scheduler_t'] = scheduler_t.state_dict()
                    
                    # ---> NEW: Save the freeze state!
                    checkpoint['teacher_frozen'] = teacher_frozen
                    checkpoint['teacher_best_acc'] = teacher_best_acc
                    checkpoint['teacher_patience_counter'] = teacher_patience_counter

                torch.save(checkpoint, save_path / 'last.pth')
                
                if len(best_models) < 5 or accuracy > best_models[-1]['acc']:
                    ckpt_name = f"model_acc_{accuracy:.4f}_ep_{epoch}.pth"
                    torch.save(checkpoint, save_path / ckpt_name)
                    print(f"⭐ Saving Top-5 Model: {ckpt_name}")
                    best_models.append({'acc': accuracy, 'path': save_path / ckpt_name})
                    best_models.sort(key=lambda x: x['acc'], reverse=True)
                    if len(best_models) > 5:
                        to_remove = best_models.pop()
                        if os.path.exists(to_remove['path']): os.remove(to_remove['path'])
                
                # --- NEW: Early Stopping Check ---
                if accuracy > best_accuracy: 
                    best_accuracy = accuracy
                    epochs_without_improvement = 0  # Reset the counter
                else:
                    epochs_without_improvement += 1
                    
                if epochs_without_improvement >= early_stop_patience:
                    print(f"🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs.")
                    break # Kills the epoch loop
                
            # --- NEW: EXPLICIT MEMORY CLEANUP ---
            # Kill the 8 multiprocessing workers and free the shared RAM
            del train_loader
            del train_sampler
            gc.collect()
            torch.cuda.empty_cache() # Optional, but keeps VRAM perfectly fragmented

    except KeyboardInterrupt:
        if is_main_process():
            print("\n🛑 KeyboardInterrupt! Saving emergency checkpoint...")
            torch.save(checkpoint, save_path / 'emergency_exit.pth')
        if not DEBUG: dist.destroy_process_group()
        exit(0)
    if not DEBUG: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--save", required=True); parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    utils.setup_seed(42)
    with open(args.config, "r") as f: config = yaml.load(f, Loader=yaml.FullLoader)
    save_path = Path(args.save) / Path(args.config).stem
    if args.tag: save_path = Path(str(save_path) + "_" + args.tag)
    if not os.path.exists(save_path): os.makedirs(save_path)
    with open(save_path / 'config_snapshot.yaml', 'w') as f: yaml.dump(config, f, default_flow_style=False)
    main(config, save_path)
