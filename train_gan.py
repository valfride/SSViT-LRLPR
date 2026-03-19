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

def make_dataloader(spec, tag=''):
    dataset = datasets.make(spec['dataset'])
    wrapper_args = {'dataset': dataset, 'corners_only': False} 
    dataset = datasets.make(spec['wrapper'], args=wrapper_args)
    
    sampler = None
    shuffle = True
    
    if tag == 'train':
        # 1. Generate the Hard-Mining Weights
        stats_path = Path('outputs') / spec.get('name', 'config_snapshot') / 'confusion_stats.json'
        weights = get_confusion_weights(dataset, stats_path)
        
        if not DEBUG:
            # 2. Distributed Hard Mining
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False
            # We will handle the actual weighting in the loss function for DDP safety,
            # but setting up the foundation here is key.
        else:
            # Single GPU Hard Mining
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
            shuffle = False

    loader = DataLoader(
        dataset, batch_size=spec['batch'], shuffle=shuffle, 
        sampler=sampler, num_workers=8, pin_memory=True, 
        collate_fn=dataset.collate_fn, drop_last=(tag == 'train'), prefetch_factor=2
    )
    return loader, sampler

def get_confusion_weights(dataset, stats_path):
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f: confusions = json.load(f)
            trouble_chars = set(confusions.keys())
        except: trouble_chars = set()
    else: trouble_chars = set()

    weights = []
    found_targets = 0
    for item in dataset.dataset:
        if any(char in trouble_chars for char in item['gt']):
            weights.append(5.0); found_targets += 1
        else: weights.append(1.0)
    
    if len(weights) > 0 and (found_targets / len(weights)) > 0.5:
        if is_main_process(): print(f"⚠️ High Error Density ({found_targets/len(weights):.1%}). Dilating sampler.")
        weights = [w * 0.5 if w > 1.0 else w for w in weights]
    return torch.DoubleTensor(weights)

def create_scheduler(optimizer, epoch_max):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

def main(config, save_path):
    local_rank = int(os.environ["LOCAL_RANK"])
    epoch_max = config['epoch_max']
    
    train_loader, train_sampler = make_dataloader(config['train_dataset'], tag='train')
    val_loader, _ = make_dataloader(config.get('val_dataset', config['train_dataset']), tag='val')

    if is_main_process(): print("Creating Teacherless VSR Model...")
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
        {'params': deform_offset_params, 'lr': base_lr * 10.0},
    ]
    
    if is_main_process(): 
        print(f"Base params: {len(base_params)} | Deform: {len(deform_offset_params)}")
        print(f"Base LR: {base_lr} | Deform LR: {base_lr * 10.0} ")

    optimizer_g = torch.optim.Adam(optim_groups)
    loss_fn_spread = HyperAttentionLoss(grid_h=16, grid_w=48).to(local_rank)
    hyper_groups = [
        # Added iris_scale here!
        {'params': [loss_fn_spread.v_stretch, loss_fn_spread.iris_scale], 'lr': 1e-2},
        {'params': [loss_fn_spread.monotonic_scale, 
                    loss_fn_spread.boundary_scale, 
                    loss_fn_spread.ortho_scale], 'lr': 1e-2}
    ]
    optimizer_hyper = torch.optim.Adam(hyper_groups)
    # -----------------------------------------------------------
    
    # --- 3. Initialize Scheduler BEFORE Resume ---
    scheduler_g = create_scheduler(optimizer_g, epoch_max)
    
    # --- 4. Resume Logic ---
    start_epoch = 1; best_accuracy = 0.0
    resume_path = config.get('resume')
    if resume_path and os.path.isfile(resume_path):
        if is_main_process(): print(f"⚠️ Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=f'cuda:{local_rank}')
        
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
            
            if 'optimizer_hyper' in checkpoint and 'loss_hyper_sd' in checkpoint:
                if is_main_process(): print("✅ Resuming Hyper-Weights.")
                
                # --- THE FIX 1: strict=False ---
                # This allows new parameters (like iris_scale) to be ignored by the old checkpoint
                # and initialize safely at their default values (0.5).
                loss_fn_spread.load_state_dict(checkpoint['loss_hyper_sd'], strict=False)
                
                # --- THE FIX 2: Safe Optimizer Loading ---
                try:
                    optimizer_hyper.load_state_dict(checkpoint['optimizer_hyper'])
                    if is_main_process(): print("✅ Resuming Hyper-Optimizer State.")
                except ValueError:
                    if is_main_process(): print("⚠️ Hyper-Optimizer state mismatch (new parameters added). Starting fresh optimizer for hyper-weights.")
            if 'optimizer_g' in checkpoint: 
                try: 
                    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
                    
                    # NEW: Conditional LR and Scheduler Force
                    if config.get('force_lr', False):
                        if is_main_process(): 
                            print(f"🔥 FORCE_LR is True: Overriding loaded LR to {base_lr} and resetting Scheduler.")
                        
                        # Group 0: Base Params
                        optimizer_g.param_groups[0]['lr'] = base_lr          
                        
                        # Group 1: Deform Offset Params
                        if len(optimizer_g.param_groups) > 1:
                            optimizer_g.param_groups[1]['lr'] = base_lr * 10.0
                            
                        # Re-create scheduler to wipe its patience/history
                        scheduler_g = create_scheduler(optimizer_g, epoch_max)
                        
                    else:
                        # If NOT forcing LR, load the old scheduler state to continue smoothly
                        if 'scheduler_g' in checkpoint:
                            if is_main_process(): print("✅ Loading previous Scheduler state.")
                            scheduler_g.load_state_dict(checkpoint['scheduler_g'])
                    
                except Exception as e:
                    if is_main_process(): print(f"⚠️ Optimizer/Scheduler resume failed: {e}")

    if not DEBUG: model_g = DDP(model_g, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        
    loss_fn = losses.make(config['loss']).to(local_rank)
    train_step = train_funcs.make(config['func_train']) 
    val_step = train_funcs.make(config['func_val'])

    stats_path = save_path / 'confusion_stats.json'
    best_models = [] 

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
        

    try:
        for epoch in range(start_epoch, epoch_max + 1):
            # 1. Properly set the epoch for Distributed Training shuffles
            if hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(epoch)
            
            if is_main_process(): print(f"🔄 Starting Epoch {epoch}...")
            
            model_g.train()
            
            # --- THE FIX: Perfectly aligned arguments (8 total + kwargs) ---
            train_loss = train_step(
                train_loader,
                val_loader,      # <--- YOU MUST ADD THIS LINE HERE
                model_g, 
                None,            # model_d
                optimizer_g, 
                None,            # optimizer_d (Fixed the missing argument)
                optimizer_hyper, # optimizer_hyper
                loss_fn_spread,  # loss_fn_spread
                config,          # config
                epoch=epoch, 
                save_path=save_path
            )
            
            val_loss, accuracy, _ = val_step(val_loader, model_g, None, loss_fn, config)
    
            if scheduler_g: 
                if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler_g.step(accuracy)
                else: scheduler_g.step()
    
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
                    'best_acc': best_accuracy 
                }
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
                if accuracy > best_accuracy: best_accuracy = accuracy

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