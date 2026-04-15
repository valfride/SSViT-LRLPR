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
        {'params': deform_offset_params, 'lr': base_lr * 10.0 if len(deform_offset_params) > 0 else base_lr},
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
        if is_main_process(): print("👻 Ghost EMA Tracker Mode ON: Creating secondary model...")
        model_ghost = models.make(config['model_g']).to(local_rank)
        model_ghost = model_ghost.to(memory_format=torch.channels_last)

        # Brain Transplant: Initialize Ghost with exact Student weights
        model_ghost.load_state_dict(model_g.state_dict())
        
        # Freeze the Ghost completely (Learns via EMA, not Backprop)
        for param in model_ghost.parameters():
            param.requires_grad = False
	
    scheduler_g = create_scheduler(config, optimizer_g, epoch_max)
    
    start_epoch = 1; best_accuracy = 0.0
    best_models = [] 
    early_stop_patience = config.get('early_stop_patience', 100)
    epochs_without_improvement = 0
    
    resume_path = config.get('resume')
    is_finetune = config.get('finetune', False) # <--- ADDED FLAG
    
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=f'cuda:{local_rank}', weights_only=False)
        
        # 1. ALWAYS load the Model Weights
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_g_sd'].items()}
        model_g.load_state_dict(state_dict, strict=False)
        
        # Safely load Ghost weights if they exist
        if use_ema_ghost and 'model_ghost_sd' in checkpoint:
            model_ghost.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_ghost_sd'].items()}, strict=False)
        
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
            if is_main_process(): print(f"🔄 Starting Epoch {epoch}...")
            
            # ---> THE CURRICULUM RAMP
            # Scales from 0.0 to 0.5 over the first 40 epochs
            new_p = min(0.5, (epoch / 40.0) * 0.5)
            train_loader.dataset.eraser_prob = new_p
            print(f"📈 Epoch {epoch}: RandomErasing Probability set to {new_p:.2f}")

            model_g.train()
            if use_ema_ghost: model_ghost.train() # Keep dropout identical
            
            train_loss = train_step(
                train_loader, val_loader,
                model_g, model_ghost,         
                optimizer_g, config,          
                epoch=epoch, save_path=save_path
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
                
                checkpoint = {
                    'epoch': epoch, 
                    'model_g_sd': model_g.module.state_dict() if hasattr(model_g, 'module') else model_g.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(), 
                    'scheduler_g': scheduler_g.state_dict(),
                    'best_acc': best_accuracy,
                    'best_models': best_models,
                    'epochs_without_improvement': epochs_without_improvement
                }
                
                if use_ema_ghost:
                    checkpoint['model_ghost_sd'] = model_ghost.module.state_dict() if hasattr(model_ghost, 'module') else model_ghost.state_dict()
				
                torch.save(checkpoint, save_path / 'last.pth')
                
                # Check early stopping against either the Student OR the Ghost
                tracking_acc = max(accuracy, ghost_val_acc) if use_ema_ghost else accuracy
                
                if len(best_models) < 5 or tracking_acc > best_models[-1]['acc']:
                    ckpt_name = f"model_acc_{tracking_acc:.4f}_ep_{epoch}.pth"
                    torch.save(checkpoint, save_path / ckpt_name)
                    print(f"⭐ Saving Top-5 Model: {ckpt_name}")
                    best_models.append({'acc': tracking_acc, 'path': save_path / ckpt_name})
                    best_models.sort(key=lambda x: x['acc'], reverse=True)
                    if len(best_models) > 5:
                        to_remove = best_models.pop()
                        if os.path.exists(to_remove['path']): os.remove(to_remove['path'])
                
                if tracking_acc > best_accuracy: 
                    best_accuracy = tracking_acc
                    epochs_without_improvement = 0  
                else:
                    epochs_without_improvement += 1
                    
                if epochs_without_improvement >= early_stop_patience:
                    print(f"🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs.")
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