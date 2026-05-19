import yaml
import torch
import utils
import argparse
import models
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
import shutil
import math
import copy

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

def log_to_csv(save_path, epoch, train_loss, val_loss, accuracy, ghost_acc, lr):
    csv_file = save_path / 'loss_log.csv'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists: 
            writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Student_Acc', 'Ghost_Acc', 'LR'])
        writer.writerow([epoch, train_loss, val_loss, accuracy, ghost_acc, lr])

def make_dataloader(spec, tag='', save_path=None):
    dataset = datasets.make(spec['dataset'])
    wrapper_args = {'dataset': dataset, 'corners_only': False} 
    dataset = datasets.make(spec['wrapper'], args=wrapper_args)
    
    sampler = None
    shuffle = True
    
    if not DEBUG and tag == 'train':
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False

    loader = DataLoader(
        dataset, batch_size=spec['batch'], shuffle=shuffle, sampler=sampler,
        num_workers=16, pin_memory=True, 
        collate_fn=dataset.collate_fn, drop_last=(tag == 'train'), prefetch_factor=4
    )
    return loader, sampler

def create_scheduler(config, optimizer, epoch_max):
    sched_config = config.get('LRScheduler', {})
    name = sched_config.get('name', 'ReduceLROnPlateau')

    if name == 'OneCycleLR':
        max_lrs = [group['lr'] for group in optimizer.param_groups]
        warmup_epoch = sched_config.get('warmup_epoch', 1.5)
        pct_start = warmup_epoch / epoch_max
        cycle_momentum = sched_config.get('cycle_momentum', False)
        
        if is_main_process():
            print(f"📈 Initializing OneCycleLR | Max LRs: {max_lrs} | Warmup: {warmup_epoch} epochs")
            
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=epoch_max, 
            pct_start=pct_start, cycle_momentum=cycle_momentum, anneal_strategy='cos'
        )
    elif name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_max, eta_min=1e-6)
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_config.get('step_size', 10), gamma=sched_config.get('gamma', 0.5))
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-6)

def main(config, save_path):
    local_rank = int(os.environ["LOCAL_RANK"])
    epoch_max = config['epoch_max']
    
    val_loader, _ = make_dataloader(config.get('val_dataset', config['train_dataset']), tag='val', save_path=save_path)

    if is_main_process(): print("Creating Student VSR Model...")
    model_g = models.make(config['model_g']).to(local_rank)
    model_g = model_g.to(memory_format=torch.channels_last)
    
    # ---> ADD THIS: Actually wrap the model in DDP if not debugging!
    if not DEBUG:
        model_g = DDP(model_g, device_ids=[local_rank], output_device=local_rank)
    
    base_lr = float(config['optimizer_sr']['args'].get('lr', 1e-4))
    
    deform_offset_params, base_params = [], []
    for name, param in model_g.named_parameters():
        if not param.requires_grad: continue
        if 'offset_predictor' in name or 'refine_shifts' in name:
            deform_offset_params.append(param)
        else:
            base_params.append(param)
            
    optim_groups = [
        {'params': base_params, 'lr': base_lr},
        # {'params': deform_offset_params, 'lr': base_lr * 10.0 if len(deform_offset_params) > 0 else base_lr},
    ]
    
    opt_name = config['optimizer_sr'].get('name', 'adamw').lower()
    opt_kwargs = config['optimizer_sr'].get('args', {}).copy()
    if 'lr' in opt_kwargs: del opt_kwargs['lr']
        
    if opt_name == 'adamw': optimizer_g = torch.optim.AdamW(optim_groups, **opt_kwargs)
    elif opt_name == 'adam': optimizer_g = torch.optim.Adam(optim_groups, **opt_kwargs)
    else: raise ValueError(f"Unknown optimizer: {opt_name}")
        
    # ==========================================
    # GHOST EMA SETUP (No Optimizers, No Schedulers)
    # ==========================================
    use_ema_ghost = config.get('use_ema_ghost', False)
    model_ghost = None

    if use_ema_ghost:
        print("👻 Ghost EMA Tracker Mode ON...")
        # Deepcopy creates the Ghost WITHOUT consuming new random numbers!
        model_ghost = copy.deepcopy(model_g) 
        
        # Freeze the Ghost
        for param in model_ghost.parameters():
            param.requires_grad = False

    scheduler_g = create_scheduler(config, optimizer_g, epoch_max)
    
    start_epoch = 1; best_accuracy = 0.0
    best_models = [] 
    early_stop_patience = config.get('early_stop_patience', 25)
    epochs_without_improvement = 0
    
    resume_path = config.get('resume')
    is_finetune = config.get('finetune', False) # <--- ADDED FLAG
    
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=f'cuda:{local_rank}', weights_only=False)
        
        # 1. ALWAYS load the Student Model Weights
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_g_sd'].items()}
        model_g.load_state_dict(state_dict, strict=False)
        
        # 2. THE FIX: Smart Ghost Loading
        if use_ema_ghost:
            if 'model_ghost_sd' in checkpoint:
                # Fallback if they were ever saved together
                model_ghost.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_ghost_sd'].items()}, strict=False)
            else:
                # Automatically hunt down the sister directory
                resume_file = Path(resume_path)
                ghost_path = resume_file.parent.parent / 'ghost_weights' / resume_file.name
                
                if ghost_path.exists():
                    if is_main_process(): print(f"👻 Successfully located and loaded sister Ghost checkpoint: {ghost_path.name}")
                    ghost_ckpt = torch.load(ghost_path, map_location=f'cuda:{local_rank}', weights_only=False)
                    model_ghost.load_state_dict({k.replace('module.', ''): v for k, v in ghost_ckpt['model_ghost_sd'].items()}, strict=False)
                else:
                    if is_main_process(): print("⚠️ Warning: Could not find corresponding Ghost checkpoint. Ghost is starting from scratch!")
        
        # 2. THE SPLIT: Fine-tune vs. True Resume
        if is_finetune:
            if is_main_process(): 
                print("🔥 FINE-TUNE MODE: Loaded pre-trained weights.")
                print("🧠 Wiping optimizer, scheduler, and starting fresh at Epoch 1.")
            start_epoch = 1
            best_accuracy = 0.0
            best_models = []
            epochs_without_improvement = 0
            
        else:
            if is_main_process(): print("✅ FULL RESUME: Restoring model, optimizer, and training state.")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_accuracy = checkpoint.get('best_acc', 0.0)
            best_models = checkpoint.get('best_models', [])
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            
            # Only load optimizer/scheduler if it's a true resume
            if 'optimizer_g' in checkpoint: 
                optimizer_g.load_state_dict(checkpoint['optimizer_g'])
                if 'scheduler_g' in checkpoint:
                    scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        
    train_step = train_funcs.make(config['func_train']) 
    val_step = train_funcs.make(config['func_val'])

    if is_main_process():
        csv_file = save_path / 'loss_log.csv'
        if not os.path.isfile(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'LR'])
                writer.writerow([0, 0.0, 0.0, 0.0, base_lr]) 
    
    train_loader, train_sampler = make_dataloader(config['train_dataset'], tag='train', save_path=save_path)
    
    try:
        for epoch in range(start_epoch, epoch_max + 1):
            if hasattr(train_sampler, 'set_epoch'): train_sampler.set_epoch(epoch)
            
            if is_main_process(): 
                print(f"🔄 Starting Epoch {epoch}...")
                # Assuming your config specifies ema_warmup_epochs or similar for the schedule
                warmup = config.get('ema_warmup_epochs', 5)
                max_eps = 2.0 # Or whatever your config/loss defines
                if epoch <= warmup:
                    progress = (epoch - 1) / max(1, warmup - 1)
                    current_eps = max_eps * 0.5 * (1.0 - math.cos(math.pi * progress))
                else:
                    current_eps = max_eps
                
                print(f"📉 PolyLoss Schedule | Epoch {epoch} | Epsilon: {current_eps:.3f}")

            model_g.train()
            if use_ema_ghost: model_ghost.train() # Keep dropout identical
            
            train_loss = train_step(
                train_loader, val_loader,
                model_g, model_ghost,         
                optimizer_g, config,          
                epoch=epoch, save_path=save_path,
                epochs_without_improvement=epochs_without_improvement # <--- ADD THIS
            )
            
            val_loss, accuracy, ghost_val_acc = val_step(val_loader, model_g, model_ghost, config, save_path=save_path)

            if scheduler_g: 
                if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau): 
                    scheduler_g.step(accuracy)
                else: 
                    scheduler_g.step()
    
            if is_main_process():
                current_lr = optimizer_g.param_groups[0]['lr']
                
                # ---> THE UPGRADE: Explicitly print both accuracies if Ghost is ON
                if use_ema_ghost:
                    print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | S-Acc: {accuracy:.4f} | G-Acc: {ghost_val_acc:.4f} | LR: {current_lr:.2e}")
                else:
                    print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {accuracy:.4f} | LR: {current_lr:.2e}")
                
                # Pass the ghost_val_acc to the upgraded CSV logger
                log_to_csv(save_path, epoch, train_loss, val_loss, accuracy, ghost_val_acc, current_lr)
                
                # ==========================================================
                # DUAL TOP-5 CHECKPOINTING LOGIC (SEPARATE FOLDERS)
                # ==========================================================
                
                # 1. Create subdirectories if they don't exist
                student_dir = save_path / 'student_weights'
                if not student_dir.exists(): student_dir.mkdir(parents=True)
                
                if use_ema_ghost:
                    ghost_dir = save_path / 'ghost_weights'
                    if not ghost_dir.exists(): ghost_dir.mkdir(parents=True)

                # Initialize Top-5 trackers
                if not hasattr(model_g, 'top_students'): model_g.top_students = []
                if use_ema_ghost and not hasattr(model_ghost, 'top_ghosts'): model_ghost.top_ghosts = []

                student_ckpt = {
                    'epoch': epoch, 
                    'model_g_sd': model_g.module.state_dict() if hasattr(model_g, 'module') else model_g.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(), 
                    'scheduler_g': scheduler_g.state_dict(),
                    # ---> ADD THESE 3 LINES:
                    'best_acc': model_g.top_students[0]['acc'] if len(model_g.top_students) > 0 else accuracy,
                    'best_models': model_g.top_students,
                    'epochs_without_improvement': epochs_without_improvement
                }
                
                # --- 2. SAVE LAST ---
                torch.save(student_ckpt, student_dir / 'last.pth')
                
                if use_ema_ghost:
                    ghost_ckpt = {
                        'epoch': epoch,
                        'model_ghost_sd': model_ghost.module.state_dict() if hasattr(model_ghost, 'module') else model_ghost.state_dict(),
                        # ---> ADD THESE 2 LINES:
                        'best_acc': model_ghost.top_ghosts[0]['acc'] if len(model_ghost.top_ghosts) > 0 else ghost_val_acc,
                        'best_models': getattr(model_ghost, 'top_ghosts', [])
                    }
                    torch.save(ghost_ckpt, ghost_dir / 'last.pth')

                # --- 3. TRACK & SAVE TOP 5 STUDENT ---
                student_filepath = student_dir / f"student_acc_{accuracy:.4f}_ep_{epoch}.pth"
                model_g.top_students.append({'acc': accuracy, 'epoch': epoch, 'path': student_filepath})
                model_g.top_students = sorted(model_g.top_students, key=lambda x: x['acc'], reverse=True)
                
                # Prune if > 5
                if len(model_g.top_students) > 5:
                    to_remove = model_g.top_students.pop()
                    if os.path.exists(to_remove['path']): os.remove(to_remove['path'])
                
                # Save if in Top 5
                if any(x['epoch'] == epoch for x in model_g.top_students):
                    torch.save(student_ckpt, student_filepath)
                    print(f"⭐ Saved to Student Top 5 -> {accuracy:.4f}")

                # --- 4. TRACK & SAVE TOP 5 GHOST ---
                if use_ema_ghost:
                    ghost_filepath = ghost_dir / f"ghost_acc_{ghost_val_acc:.4f}_ep_{epoch}.pth"
                    model_ghost.top_ghosts.append({'acc': ghost_val_acc, 'epoch': epoch, 'path': ghost_filepath})
                    model_ghost.top_ghosts = sorted(model_ghost.top_ghosts, key=lambda x: x['acc'], reverse=True)
                    
                    # Prune if > 5
                    if len(model_ghost.top_ghosts) > 5:
                        to_remove_g = model_ghost.top_ghosts.pop()
                        if os.path.exists(to_remove_g['path']): os.remove(to_remove_g['path'])
                    
                    # Save if in Top 5
                    if any(x['epoch'] == epoch for x in model_ghost.top_ghosts):
                        torch.save(ghost_ckpt, ghost_filepath)
                        print(f"👻 Saved to Ghost Top 5 -> {ghost_val_acc:.4f}")
                    

                # --- 5. EARLY STOPPING TRIGGER (COMBINED) ---
                student_improved = False
                ghost_improved = False
                
                # Check if Student hit a new all-time high
                if len(model_g.top_students) > 0 and accuracy >= model_g.top_students[0]['acc']:
                    student_improved = True
                    
                # Check if Ghost hit a new all-time high
                if use_ema_ghost and hasattr(model_ghost, 'top_ghosts') and len(model_ghost.top_ghosts) > 0:
                    if ghost_val_acc >= model_ghost.top_ghosts[0]['acc']:
                        ghost_improved = True

                # If EITHER model improved, reset the patience counter!
                if student_improved or ghost_improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stop_patience:
                    print(f"🛑 Early stopping triggered! No improvement in Student or Ghost for {early_stop_patience} epochs.")
                    break 

                
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