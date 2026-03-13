import numpy as np
import cv2
import torch
import torch.nn as nn
import random
import json
import os
import albumentations as A
import torch.nn.functional as F
from pathlib import Path
from datasets import register
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize 
from albumentations.pytorch import ToTensorV2

# ==============================================================================
# DATASET WRAPPER (Cleaned: Realistic Physics Only)
# ==============================================================================
@register('VSR_collate_fn')
class Sequantial_lr_sr(Dataset):
    def __init__(self, imgW, imgH, aug, image_aspect_ratio, background,
                 test=False, in_images=5, synthetic_prob=0.5, anagram_prob=0.0, skip_low_res=False, dataset=None, **kwargs):
        
        self.in_images = in_images
        self.imgW = imgW      
        self.imgH = imgH      
        self.aug = aug 
        self.synthetic_prob = synthetic_prob 
        
        # We ignore these legacy params now
        self.anagram_prob = 0.0 
        self.skip_low_res = False 
        
        self.dataset = dataset
        self.test = test
        
        # CRITICAL: Restore [-1, 1] Normalization for VSR Stability
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # ----------------------------------------------------------------------
        # A. SYNTHETIC AUGMENTATION (Camera Physics)
        # ----------------------------------------------------------------------
        # 1. Blur (Optical Defect)
        self.blur_aug = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MotionBlur(blur_limit=5, p=0.5),
            ], p=0.5), # 50% chance of blur
        ])
        
        # 2. Resolution (Sensor Defect) - CRITICAL FOR VSR
        self.downsample_ops = A.OneOf([
            # Simulates 720p/1080p surveillance cropping
            A.Resize(height=imgH//2, width=imgW//2, interpolation=cv2.INTER_LINEAR),
            A.Resize(height=imgH//3, width=imgW//3, interpolation=cv2.INTER_LINEAR),
        ], p=0.5)
        
        # 3. Noise/Compression (Digital Defect)
        self.compression_aug = A.Compose([
            A.OneOf([
                # JPEG Artifacts
                A.ImageCompression(quality_range=[40, 75], p=1.0),
                # ISO Grain
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=1.0),
            ], p=0.5),
        ])
        
        # ----------------------------------------------------------------------
        # B. COMMON AUGMENTATION (Environment)
        # ----------------------------------------------------------------------
        self.common_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RGBShift(r_shift_limit=0.05, g_shift_limit=0.05, b_shift_limit=0.05, p=0.3),
        ])

        # ----------------------------------------------------------------------
        # C. REAL LR AUGMENTATION (Subtle)
        # ----------------------------------------------------------------------
        self.real_pixel_aug = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.2), p=0.5),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])

        assert self.dataset is not None, "Dataset is None"
        self.indices = list(range(len(self.dataset)))

    def _get_files_by_pattern(self, folder, prefix):
        valid_exts = ['.png', '.jpg', '.jpeg']
        all_files = sorted([p for p in folder.iterdir() if p.name.startswith(prefix) and p.suffix.lower() in valid_exts])
        return all_files

    # ==========================================================================
    # GEOMETRY HELPERS (Standard Affine Only)
    # ==========================================================================
    def generate_geo_params(self):
        # Reduced angles to match real-world rectified plates
        angle = random.uniform(-2, 2) 
        scale = random.uniform(0.95, 1.05)
        shift_x = random.uniform(-0.02, 0.02)
        shift_y = random.uniform(-0.02, 0.02)
        return angle, scale, shift_x, shift_y

    def apply_geo_transform(self, img, params, size=None):
        angle, scale, shift_x, shift_y = params
        h, w = size if size else img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += shift_x * w
        M[1, 2] += shift_y * h
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    def realistic_downsample(self, hr_img):
        """
        Physics-Based Degradation Pipeline
        """
        if self.aug: hr_img = self.blur_aug(image=hr_img)['image']
        
        # Resize (The VSR Task)
        lr_img = self.downsample_ops(image=hr_img)['image']
        
        if self.aug: 
            lr_img = self.compression_aug(image=lr_img)['image']
            lr_img = self.common_aug(image=lr_img)['image']
            
        return lr_img

    def __getitem__(self, idx):
        real_idx = self.indices[idx % len(self.indices)]
        item = self.dataset[real_idx]
        
        folder_path = Path(item['imgs'])
        plate = item['gt']
        
        paths_hr = self._get_files_by_pattern(folder_path, "hr-")
        paths_lr = self._get_files_by_pattern(folder_path, "lr-")
        
        if not paths_hr: return self.__getitem__(random.randint(0, len(self)-1))
        
        if self.synthetic_prob == 0.0 and len(paths_lr) == 0:
             return self.__getitem__(random.randint(0, len(self)-1))

        if len(paths_hr) > self.in_images:
            start = random.randint(0, len(paths_hr) - self.in_images)
            paths_hr = paths_hr[start:start+self.in_images]
            if len(paths_lr) >= len(paths_hr):
                paths_lr = paths_lr[start:start+self.in_images]
            elif self.synthetic_prob == 0.0:
                return self.__getitem__(random.randint(0, len(self)-1))
        else:
            paths_hr = (paths_hr * 5)[:self.in_images]
            if paths_lr: 
                paths_lr = (paths_lr * 5)[:self.in_images]
            elif self.synthetic_prob == 0.0:
                 return self.__getitem__(random.randint(0, len(self)-1))

        batch_lrs, batch_hrs = [], []
        anno_file = folder_path / 'annotations.json'
        anno = json.load(open(anno_file)) if anno_file.exists() else None

        valid_seq_hr, valid_seq_lr_real = [], []

        # 1. Load HR Images & Rectify
        for i, p in enumerate(paths_hr):
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            
            # --- REMOVED ASPECT RATIO FILTER HERE ---
            # We now accept all shapes and let warpPerspective handle it.

            # Rectification
            if anno and 'corners' in anno and p.name in anno['corners']:
                try:
                    corns = anno['corners'][p.name]
                    pts = np.array([corns['top-left'], corns['top-right'], corns['bottom-right'], corns['bottom-left']], dtype='float32')
                    dst = np.array([[0, 0], [2*self.imgW-1, 0], [2*self.imgW-1, 2*self.imgH-1], [0, 2*self.imgH-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(pts, dst)
                    img = cv2.warpPerspective(img, M, (2*self.imgW, 2*self.imgH))
                except:
                     img = cv2.resize(img, (2*self.imgW, 2*self.imgH))
            else:
                img = cv2.resize(img, (2*self.imgW, 2*self.imgH))
                
            valid_seq_hr.append(img)
            
            if i < len(paths_lr):
                img_l = cv2.cvtColor(cv2.imread(str(paths_lr[i])), cv2.COLOR_BGR2RGB)
                img_l = cv2.resize(img_l, (self.imgW, self.imgH))
                valid_seq_lr_real.append(img_l)
            else:
                valid_seq_lr_real.append(None)

        # --- REMOVED ANAGRAM ENGINE BLOCK ---
        # No more shuffling. The plate is always consistent.

        # 2. Generate LR Sequence
        final_seq_lr = []
        final_seq_hr = []

        # Determine Augmentation Geometry Once per Sequence
        geo_params = self.generate_geo_params() if self.aug else None

        for i, hr in enumerate(valid_seq_hr):
            use_real = False
            
            if valid_seq_lr_real[i] is not None:
                if self.synthetic_prob == 0.0:
                    use_real = True
                elif random.random() > self.synthetic_prob:
                    use_real = True

            # A. Real LR Path
            if use_real:
                lr = valid_seq_lr_real[i]
                if self.aug: 
                    lr = self.real_pixel_aug(image=lr)['image']
            
            # B. Synthetic LR Path
            else:
                if self.synthetic_prob == 0.0:
                     return self.__getitem__(random.randint(0, len(self)-1))
                # Apply Physics-Based Augmentations
                lr = self.realistic_downsample(hr)
            
            # Late Resize to ensure exact dims
            if lr.shape[0] != self.imgH or lr.shape[1] != self.imgW:
                lr = cv2.resize(lr, (self.imgW, self.imgH))
            
            # Apply Geometry to both (Rotation/Shift)
            if self.aug and geo_params is not None:
                lr = self.apply_geo_transform(lr, geo_params, size=(self.imgH, self.imgW))
                hr = self.apply_geo_transform(hr, geo_params, size=(2*self.imgH, 2*self.imgW))

            final_seq_lr.append(lr)
            final_seq_hr.append(hr)

        # --- REMOVED BOX PERMUTATION ---
        
        # 3. Finalize (Normalize & Stack)
        for lr, hr in zip(final_seq_lr, final_seq_hr):
            batch_lrs.append(self.normalize(ToTensor()(lr)))
            batch_hrs.append(self.normalize(ToTensor()(hr)))

        return {
            'lr': torch.stack(batch_lrs),
            'hr': torch.stack(batch_hrs),
            'gt': plate,
            'name': str(folder_path)
        }
    
    def __len__(self): return len(self.indices)
    def collate_fn(self, batch):
        lrs = torch.stack([b['lr'] for b in batch]).permute(0, 2, 1, 3, 4)
        hrs = torch.stack([b['hr'] for b in batch]).permute(0, 2, 1, 3, 4)
        gts = [b['gt'] for b in batch]
        names = [b['name'] for b in batch]
        return {'lr': lrs, 'hr': hrs, 'gt': gts, 'name': names}