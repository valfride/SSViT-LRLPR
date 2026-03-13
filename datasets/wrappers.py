import numpy as np
import cv2
import torch
import random
import albumentations as A
from pathlib import Path
from datasets import register
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize 

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
                test=False, in_images=1, synthetic_prob=0.0, fully_synthetic_prob=0.0, 
                use_cache=False, skip_low_res=False, dataset=None, **kwargs):
        
        self.imgW = imgW      
        self.imgH = imgH      
        self.aug = aug 
        self.dataset = dataset
        self.test = test
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        assert self.dataset is not None, "Dataset is None"

        # ====================================================================
        # PIPELINE 1: SINGLE-IMAGE GEOMETRIC TWEAKS 
        # ====================================================================
        # Removed 'additional_targets' since we no longer track HR
        self.geo_aug = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=(-0.02, 0.02), 
                rotate_limit=5,    
                p=0.9, 
                border_mode=cv2.BORDER_REPLICATE
            ),
        ])

        # ====================================================================
        # PIPELINE 2: THE DEGRADATION ENGINE
        # ====================================================================
        self.hr_to_lr_degrade = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(7, 7), p=1.0),
            ], p=1.0), 

            A.Downscale(
                scale_range=[0.16, 0.18],
                interpolation_pair={"upscale": cv2.INTER_LANCZOS4 ,"downscale": cv2.INTER_LANCZOS4 },
                p=1.0
            ),
            
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=(-0.1, 0.1), 
                p=0.6
            ),
            
            A.ImageCompression(
                compression_type="webp",
                quality_range=[95, 100],
                p=0.8
            ),

            A.Resize(height=self.imgH, width=self.imgW, interpolation=cv2.INTER_CUBIC)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Fetch the decoded dictionary from the LMDBDataset
        item = self.dataset[idx]
        
        # 2. Extract the pre-loaded data
        img_raw = item['img_raw']      # ALREADY decoded and converted to RGB!
        plate_gt = item['gt']          
        filename = item['name']
        image_path_str = item['img_path'] 
        
        is_hr_file = filename.startswith("hr-")

        # ---> REMOVED: cv2.cvtColor(cv2.imread(image_path_str), cv2.COLOR_BGR2RGB) <---
        
        # 3. Generate the LR Image (The rest of your code remains exactly the same!)
        if is_hr_file:
            # If the dataset provides HR, we aggressively degrade it...
            hr_img = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
            
            if self.aug and not self.test:
                lr_img = self.hr_to_lr_degrade(image=hr_img)['image']
            else:
                lr_img = hr_img
        else:
            # If the dataset is already LR, just resize it to the expected dimensions
            lr_img = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)

        # 4. Apply Single-Image Geometric Augmentations
        if self.aug and not self.test:
            # Applied directly to the base resolution, no double-resize!
            augmented = self.geo_aug(image=lr_img)
            lr_img = augmented['image']

        # 5. Convert to Tensors 
        t_lr = self.normalize(ToTensor()(lr_img.copy())).unsqueeze(0)
        
        # Removed HR tensor return entirely
        return {
            'lr': t_lr,       
            'gt': plate_gt,
            'name': filename,
            'img_path': image_path_str
        }

    def collate_fn(self, batch):
        # Removed 'hr' from the stacked batch dictionary
        return {
            'lr': torch.stack([b['lr'] for b in batch]),
            'gt': [b['gt'] for b in batch],
            'name': [b['name'] for b in batch],
            'img_path': [b['img_path'] for b in batch]
        }

@register('VSR_Sequence_collate_fn')
class Sequential_Sequence_sr(Dataset):
    """
    Groups images by sequence (track_id) for validation.
    Assumes the dataset provides a 'track_id' or 'name' that can be parsed.
    """
    def __init__(self, imgW, imgH, aug, image_aspect_ratio, background,
                test=True, in_images=5, synthetic_prob=0.0, fully_synthetic_prob=0.0, 
                use_cache=False, skip_low_res=False, dataset=None, **kwargs):
        
        self.imgW = imgW      
        self.imgH = imgH      
        self.test = test
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.in_images = in_images # Expected images per sequence
        
        assert dataset is not None, "Dataset is None"

        # 1. Group the underlying dataset by track
        from collections import defaultdict
        self.sequences = defaultdict(list)
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            # Extract track ID from path (e.g., track_10019)
            # You might need to adjust this parsing based on your exact file structure
            path_str = str(item['img_path'])
            try:
                # Assuming structure like: .../track_12345/lr-001.jpg
                track_id = path_str.split('track_')[1].split('/')[0] 
            except IndexError:
                # Fallback if structure is different
                track_id = Path(path_str).parent.name 
                
            self.sequences[track_id].append(item)
            
        # Convert dict to a list of sequences
        self.grouped_dataset = list(self.sequences.values())

    def __len__(self):
        return len(self.grouped_dataset)

    def __getitem__(self, idx):
        sequence_items = self.grouped_dataset[idx]
        
        # We need exactly self.in_images (e.g., 5). 
        # Pad or truncate if necessary, though they should ideally be exactly 5.
        if len(sequence_items) < self.in_images:
            # Pad by repeating the last image
            sequence_items.extend([sequence_items[-1]] * (self.in_images - len(sequence_items)))
        elif len(sequence_items) > self.in_images:
            sequence_items = sequence_items[:self.in_images]

        lr_tensors = []
        gts = []
        names = []
        
        for item in sequence_items:
            img_raw = item['img_raw']
            # Standard validation resize (no degradation)
            lr_img = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
            t_lr = self.normalize(ToTensor()(lr_img.copy()))
            
            lr_tensors.append(t_lr)
            gts.append(item['gt'])
            names.append(item['name'])

        # Stack the 5 images into shape: (5, 3, 32, 96)
        sequence_tensor = torch.stack(lr_tensors)
        
        return {
            'lr_seq': sequence_tensor,       
            'gt': gts[0], # The ground truth is the same for all images in the sequence
            'names': names
        }

    def collate_fn(self, batch):
        return {
            # lr_seq shape: (Batch_Size, 5, 3, 32, 96)
            'lr_seq': torch.stack([b['lr_seq'] for b in batch]),
            'gt': [b['gt'] for b in batch],
            'names': [b['names'] for b in batch]
        }