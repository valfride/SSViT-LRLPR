import numpy as np
import cv2
import torch
import random
import albumentations as A
from pathlib import Path
from datasets import register
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize 
from tqdm import tqdm

from albumentations.core.transforms_interface import ImageOnlyTransform

class AdvancedPhysicsMotionBlur(ImageOnlyTransform):
    """
    Simulates physically accurate vehicle motion blur.
    """
    def __init__(self, velocity_range=(7, 25), angle_range=(-20, 20), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.velocity_range = velocity_range 
        self.angle_range = angle_range    

    def apply(self, img, **params):
        velocity = random.randint(self.velocity_range[0], self.velocity_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        
        ksize = velocity + 6
        if ksize % 2 == 0: ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        start_x = center - velocity // 2
        end_x = center + velocity // 2
        
        blur_type = random.random()
        if blur_type < 0.33:
            kernel[center, start_x:end_x] = 1.0
        elif blur_type < 0.66:
            fade = np.linspace(1.0, 0.1, end_x - start_x)
            if random.random() < 0.5: fade = fade[::-1] 
            kernel[center, start_x:end_x] = fade
        else:
            x_coords = np.linspace(-3, 3, end_x - start_x)
            gauss_fade = np.exp(-0.5 * (x_coords ** 2))
            kernel[center, start_x:end_x] = gauss_fade
            
        rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (ksize, ksize))
        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(img, -1, kernel)
        
    def get_transform_init_args_names(self):
        return ("velocity_range", "angle_range")

@register('VSR_collate_fn')
class Sequential_lr_sr(Dataset):
    def __init__(self, imgW, imgH, aug, image_aspect_ratio, background,
                 test=False, in_images=1, synthetic_prob=0.5, anagram_prob=0.0, 
                 use_cache=False, skip_low_res=False, dataset=None, **kwargs):
        
        self.imgW = imgW      
        self.imgH = imgH      
        self.aug = aug 
        self.synthetic_prob = synthetic_prob 
        self.dataset = dataset
        self.test = test
        self.use_cache = use_cache
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.in_images = in_images
        
        # 1. Soft Geometric Pipeline (Updated for Albumentations 1.4+)
        self.paired_geo_aug = A.Compose([
            # 🆕 UPDATE: 'alpha_affine' removed. 
            A.ElasticTransform(alpha=1, sigma=50, p=0.4), 
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        ], additional_targets={'hr_image': 'image'}) 

        # 2. Safe Photometric Pipeline (LR only)
        # 🆕 UPDATE: CoarseDropout uses ranges instead of max_holes/height/width
        self.safe_pixel_aug = A.Compose([
            A.RandomGamma(gamma_limit=(80, 120), p=0.8), 
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 8),      # Replaces max_holes=8
                hole_height_range=(1, 4),    # Replaces max_height=4
                hole_width_range=(1, 4),     # Replaces max_width=4
                p=0.5
            ),
        ])

        # 3. Teacher Degradation
        # 🆕 UPDATE: Fixed GaussNoise and CoarseDropout args
        self.teacher_degrade = A.Compose([
            A.OneOf([
                # GaussNoise now uses std_range (std deviation range) instead of var_limit
                A.GaussNoise(std_range=(0.1, 0.3), p=0.5), 
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ImageCompression(quality_range=[50, 90], p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 3),      # Replaces max_holes=3
                hole_height_range=(1, 8),    # Replaces max_height=8
                hole_width_range=(1, 8),     # Replaces max_width=8
                p=0.3
            ),
        ])

        # 4. Synthetic Degradation
        self.synthetic_degrade = A.Compose([
            A.Downscale(scale_range=[0.3, 0.3], interpolation_pair={"upscale":2,"downscale":2}, p=1.0),
            AdvancedPhysicsMotionBlur(velocity_range=(3, 3), angle_range=(-45, 45)),
            A.ImageCompression(quality_range=[45, 45], p=1.0),
        ])

        assert self.dataset is not None, "Dataset is None"
        
        # --- RAM CACHING ---
        self.image_cache = {} 
        if self.use_cache and not self.test: 
            print(f"\n🚀 RAM MODE: Pre-loading {len(self.dataset)} tracks into memory...")
            for item in tqdm(self.dataset, desc="Caching Dataset"):
                folder_path = Path(item['imgs'])
                for p in self._get_files_by_pattern(folder_path, "hr-"):
                    self.image_cache[str(p)] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                for p in self._get_files_by_pattern(folder_path, "lr-"):
                    self.image_cache[str(p)] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            print(f"✅ Cached {len(self.image_cache)} images in RAM.\n")

    def _get_files_by_pattern(self, folder, prefix):
        valid_exts = ['.png', '.jpg', '.jpeg']
        return sorted([p for p in folder.iterdir() if p.name.startswith(prefix) and p.suffix.lower() in valid_exts])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        track_idx = idx
        force_synthetic = False if self.test else (random.random() < self.synthetic_prob)

        item = self.dataset[track_idx]
        folder_path = Path(item['imgs'])
        plate = item['gt']

        paths_hr = self._get_files_by_pattern(folder_path, "hr-")
        paths_lr = self._get_files_by_pattern(folder_path, "lr-")

        t_lr_list, t_hr_list, t_teacher_list = [], [], []

        total_frames = len(paths_hr)
        if total_frames >= self.in_images:
            start_idx = 0 if self.test else random.randint(0, total_frames - self.in_images)
            frame_indices = range(start_idx, start_idx + self.in_images)
        else:
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (self.in_images - total_frames)

        for i in frame_indices:
            path_hr = paths_hr[i]
            path_lr = paths_lr[i] if i < len(paths_lr) else None

            str_hr = str(path_hr)
            if str_hr in self.image_cache: 
                hr_img_raw = self.image_cache[str_hr].copy()
            else: 
                hr_img_raw = cv2.cvtColor(cv2.imread(str_hr), cv2.COLOR_BGR2RGB)

            hr_img = cv2.resize(hr_img_raw, (self.imgW, self.imgH))
            
            if force_synthetic or path_lr is None:
                lr_img = self.synthetic_degrade(image=hr_img)['image']
            else:
                str_lr = str(path_lr)
                if str_lr in self.image_cache: lr_img = self.image_cache[str_lr].copy()
                else: lr_img = cv2.cvtColor(cv2.imread(str_lr), cv2.COLOR_BGR2RGB)

            if lr_img.shape[:2] != (self.imgH, self.imgW):
                lr_img = cv2.resize(lr_img, (self.imgW, self.imgH), interpolation=cv2.INTER_AREA)

            if self.aug:
                lr_img_up = cv2.resize(lr_img, (self.imgW * 2, self.imgH * 2), interpolation=cv2.INTER_LINEAR)
                hr_img_up = cv2.resize(hr_img, (self.imgW * 2, self.imgH * 2), interpolation=cv2.INTER_LINEAR)

                augmented = self.paired_geo_aug(image=lr_img_up, hr_image=hr_img_up)
                
                lr_img = cv2.resize(augmented['image'], (self.imgW, self.imgH), interpolation=cv2.INTER_AREA)
                hr_img = cv2.resize(augmented['hr_image'], (self.imgW, self.imgH), interpolation=cv2.INTER_AREA)
                
                lr_img = self.safe_pixel_aug(image=lr_img)['image']

            if self.aug and not self.test:
                teacher_img = self.teacher_degrade(image=hr_img.copy())['image']
            else:
                teacher_img = hr_img.copy()

            t_lr_list.append(self.normalize(ToTensor()(lr_img.copy())))
            t_hr_list.append(self.normalize(ToTensor()(hr_img.copy())))
            t_teacher_list.append(self.normalize(ToTensor()(teacher_img.copy())))

        return {
            'lr': torch.stack(t_lr_list, dim=0),       
            'hr': torch.stack(t_hr_list, dim=0),       
            'teacher_input': torch.stack(t_teacher_list, dim=0),
            'gt': plate,
            'name': str(folder_path.name),
            'img_path': str(paths_lr[frame_indices[0]].resolve() if len(paths_lr) > 0 else paths_hr[frame_indices[0]].resolve()) 
        }

    def collate_fn(self, batch):
        return {
            'lr': torch.stack([b['lr'] for b in batch]),
            'hr': torch.stack([b['hr'] for b in batch]),
            'teacher_input': torch.stack([b['teacher_input'] for b in batch]),
            'gt': [b['gt'] for b in batch],
            'name': [b['name'] for b in batch],
            'img_path': [b['img_path'] for b in batch]
        }