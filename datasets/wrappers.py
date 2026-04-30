import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
import random
import albumentations as A
from pathlib import Path
from datasets import register
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize 
import io
import torch.fft
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from albumentations.core.transforms_interface import ImageOnlyTransform
import sys
import re

class TokenAwareMasking(object):
    """
    Drops image data strictly along token/patch boundaries (MAE-style).
    This forces the ViT to use global context without creating fractional tokens.
    """
    def __init__(self, patch_size=8, mask_ratio_range=(0.1, 0.25), p=1.0):
        self.patch_size = patch_size
        self.mask_ratio_range = mask_ratio_range
        self.p = p

    def __call__(self, img_tensor):
        if random.random() > self.p:
            return img_tensor
            
        c, h, w = img_tensor.shape
        # Calculate the 8x8 grid dimensions
        gh, gw = h // self.patch_size, w // self.patch_size
        num_patches = gh * gw
        
        # Decide how many patches to drop
        mask_ratio = random.uniform(*self.mask_ratio_range)
        num_mask = int(num_patches * mask_ratio)
        
        # Create a flat mask of 1s and 0s
        mask_idx = torch.randperm(num_patches)[:num_mask]
        mask_flat = torch.ones(num_patches, device=img_tensor.device)
        mask_flat[mask_idx] = 0.0
        
        # Reshape and upscale the mask back to image dimensions
        mask_grid = mask_flat.view(1, 1, gh, gw)
        mask_spatial = F.interpolate(mask_grid, size=(h, w), mode='nearest').squeeze(0)
        
        # Note: Since the tensor will already be normalized to [-1, 1], 
        # multiplying by 0 gives a neutral 50% grey, which is the perfect "blank" token!
        return img_tensor * mask_spatial

class AdvancedPhysicsMotionBlur(ImageOnlyTransform):
    """Simulates physically accurate vehicle motion blur."""
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

# --- Helper Math Functions for FDA ---
def _rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image.unbind(dim=-3)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    return torch.stack((y, cb, cr), dim=-3)

def _ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    y, cb, cr = image.unbind(dim=-3)
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=-3)

def _match_color_and_contrast(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    src_mean, src_std = src.mean(dim=(-2, -1), keepdim=True), src.std(dim=(-2, -1), keepdim=True)
    tgt_mean, tgt_std = target.mean(dim=(-2, -1), keepdim=True), target.std(dim=(-2, -1), keepdim=True)
    matched = (src - src_mean) / (src_std + 1e-8)
    matched = (matched * tgt_std) + tgt_mean
    return matched.clamp(0, 1)

class FourierCCTVDegradation(ImageOnlyTransform):
    """Transfers style/degradation of a real LR crop with ViT-safe Edge Preservation."""
    # ---> THE FIX 1: Lower the beta_range to a conservative (0.01, 0.04)
    def __init__(self, lr_image_pool, beta_range=(0.01, 0.04), jpeg_range=(65, 85), apply_jpeg=True, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.lr_image_pool = lr_image_pool 
        self.beta_range = beta_range
        self.jpeg_range = jpeg_range
        self.apply_jpeg = apply_jpeg

    def apply(self, img, **params):
        target_lr_np = random.choice(self.lr_image_pool)
        beta = random.uniform(self.beta_range[0], self.beta_range[1])
        blur_sigma = random.uniform(0.1, 0.8) 

        img_hr = TF.to_tensor(img).unsqueeze(0)
        img_lr = TF.to_tensor(target_lr_np).unsqueeze(0)

        img_hr_blurred = T.GaussianBlur(kernel_size=(3, 3), sigma=(blur_sigma, blur_sigma))(img_hr)
        lr_h, lr_w = img_lr.shape[-2:]
        img_hr_physical = F.interpolate(img_hr_blurred, size=(lr_h, lr_w), mode='bicubic', align_corners=False).clamp(0, 1)

        img_hr_color_matched = _match_color_and_contrast(img_hr_physical, img_lr)

        hr_ycbcr = _rgb_to_ycbcr(img_hr_color_matched)
        lr_ycbcr = _rgb_to_ycbcr(img_lr)
        
        hr_y = hr_ycbcr[:, 0:1, :, :]
        lr_y = lr_ycbcr[:, 0:1, :, :]
        
        fft_hr = torch.fft.fftn(hr_y, dim=(-2, -1))
        fft_lr = torch.fft.fftn(lr_y, dim=(-2, -1))
        
        fft_hr_shifted = torch.fft.fftshift(fft_hr, dim=(-2, -1))
        fft_lr_shifted = torch.fft.fftshift(fft_lr, dim=(-2, -1))
        
        amp_hr, phase_hr = torch.abs(fft_hr_shifted), torch.angle(fft_hr_shifted)
        amp_lr = torch.abs(fft_lr_shifted)
        
        _, _, H, W = img_hr_color_matched.shape
        c_h, c_w = H // 2, W // 2
        sigma = min(H, W) * beta
        
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist_sq = (X - c_w)**2 + (Y - c_h)**2
        
        # ---> THE FIX 2: Butterworth Filter for Sharp High-Frequency Preservation
        # The power of 4 creates a steep drop-off, protecting edges!
        mask = 1.0 / (1.0 + (dist_sq / (sigma**2))**4)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        amp_hr_mixed = (amp_lr * mask) + (amp_hr * (1 - mask))
        fft_mixed = amp_hr_mixed * torch.exp(1j * phase_hr)
        fft_mixed_unshifted = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
        degraded_y = torch.real(torch.fft.ifftn(fft_mixed_unshifted, dim=(-2, -1)))
        
        deg_mean, deg_std = degraded_y.mean(dim=(-2, -1), keepdim=True), degraded_y.std(dim=(-2, -1), keepdim=True)
        hr_mean, hr_std = hr_y.mean(dim=(-2, -1), keepdim=True), hr_y.std(dim=(-2, -1), keepdim=True)
        degraded_y = ((degraded_y - deg_mean) / (deg_std + 1e-8) * hr_std) + hr_mean
        degraded_y = degraded_y.clamp(0, 1)
        
        content_ycbcr_mixed = hr_ycbcr.clone()
        content_ycbcr_mixed[:, 0:1, :, :] = degraded_y
        final_rgb = _ycbcr_to_rgb(content_ycbcr_mixed).clamp(0, 1)

        if self.apply_jpeg:
            jpeg_quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
            pil_img = TF.to_pil_image(final_rgb.squeeze(0))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=jpeg_quality)
            return np.array(Image.open(buffer))
        else:
            return (final_rgb.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

class FourierLRtoLRMixup(ImageOnlyTransform):
    """Intra-Domain Style Transfer: Swaps styles between two real LR images without structural blur."""
    def __init__(self, lr_image_pool, beta_range=(0.02, 0.08), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.lr_image_pool = lr_image_pool
        self.beta_range = beta_range

    def apply(self, img, **params):
        target_lr_np = random.choice(self.lr_image_pool)
        beta = random.uniform(self.beta_range[0], self.beta_range[1])

        img_content = TF.to_tensor(img).unsqueeze(0)
        img_style = TF.to_tensor(target_lr_np).unsqueeze(0)
        img_style = F.interpolate(img_style, size=img_content.shape[-2:], mode='bilinear', align_corners=False)

        img_matched = _match_color_and_contrast(img_content, img_style)

        content_ycbcr = _rgb_to_ycbcr(img_matched)
        style_ycbcr = _rgb_to_ycbcr(img_style)
        
        content_y = content_ycbcr[:, 0:1, :, :]
        style_y = style_ycbcr[:, 0:1, :, :]
        
        fft_content = torch.fft.fftn(content_y, dim=(-2, -1))
        fft_style = torch.fft.fftn(style_y, dim=(-2, -1))
        
        fft_c_shifted = torch.fft.fftshift(fft_content, dim=(-2, -1))
        fft_s_shifted = torch.fft.fftshift(fft_style, dim=(-2, -1))
        
        amp_c, phase_c = torch.abs(fft_c_shifted), torch.angle(fft_c_shifted)
        amp_s = torch.abs(fft_s_shifted)
        
        _, _, H, W = img_content.shape
        c_h, c_w = H // 2, W // 2
        sigma = min(H, W) * beta
        
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist_sq = (X - c_w)**2 + (Y - c_h)**2
        mask = torch.exp(-dist_sq / (2 * (sigma**2))).unsqueeze(0).unsqueeze(0)
        
        amp_mixed = (amp_s * mask) + (amp_c * (1 - mask))
        fft_mixed = amp_mixed * torch.exp(1j * phase_c)
        fft_mixed_unshifted = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
        degraded_y = torch.real(torch.fft.ifftn(fft_mixed_unshifted, dim=(-2, -1)))
        
        deg_mean, deg_std = degraded_y.mean(dim=(-2, -1), keepdim=True), degraded_y.std(dim=(-2, -1), keepdim=True)
        c_mean, c_std = content_y.mean(dim=(-2, -1), keepdim=True), content_y.std(dim=(-2, -1), keepdim=True)
        degraded_y = ((degraded_y - deg_mean) / (deg_std + 1e-8) * c_std) + c_mean
        degraded_y = degraded_y.clamp(0, 1)
        
        content_ycbcr_mixed = content_ycbcr.clone()
        content_ycbcr_mixed[:, 0:1, :, :] = degraded_y
        final_rgb = _ycbcr_to_rgb(content_ycbcr_mixed).clamp(0, 1)

        out_np = (final_rgb.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return out_np

@register('VSR_collate_fn')
class Sequential_lr_sr(Dataset):
    def __init__(self, imgW, imgH, aug, image_aspect_ratio, background,
                test=False, in_images=1, synthetic_prob=0.0, fully_synthetic_prob=0.0, 
                use_cache=False, skip_low_res=False, dataset=None, 
                return_hr=False, use_fda_hr=True, use_fda_lr=True, **kwargs): # <--- NEW FLAGS
        
        self.imgW = imgW      
        self.imgH = imgH      
        self.aug = aug 
        self.dataset = dataset
        self.test = test
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.return_hr = return_hr
        self.eraser_prob = 0
        # ---> NEW: Save the flags to the class
        self.use_fda_hr = use_fda_hr
        self.use_fda_lr = use_fda_lr
        
        assert self.dataset is not None, "Dataset is None"

        assert self.dataset is not None, "Dataset is None"

        self.geo_aug = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.02,   # Drop from 0.05
                scale_limit=(-0.02, 0.02), 
                rotate_limit=5,     # Drop from 7
                p=0.5, 
                border_mode=cv2.BORDER_REPLICATE
            ),
            A.Perspective(scale=(0.01, 0.03), p=0.3, fit_output=True), # Drop scale limits
            # Turn GridDistortion OFF for now. It destroys tiny text.
            # A.GridDistortion(num_steps=4, distort_limit=0.1, p=0.3, border_mode=cv2.BORDER_REPLICATE), 
        ], additional_targets={'image_hr': 'image', 'image_sr': 'image'})

        # =========================================================
        # ---> NEW: LR-Safe PyTorch Augmentations for the ViT
        # =========================================================
        self.color_aug = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0)
        # Drops a tiny black box (2% to 8% of the image) 40% of the time to force ViT global attention
        # self.eraser = T.RandomErasing(p=0.4, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0)
        # =========================================================
        # 1. GROUP BY TRACK (Supports LR Tracks AND HR-Only RODOSOL)
        # =========================================================
        from collections import defaultdict
        self.track_dict = defaultdict(lambda: {'lr': [], 'hr': []})
        self.lr_pool = []
        
        for item in self.dataset:
            filename = item.get('name', '')
            path_str = str(item.get('img_path', ''))
            
            try:
                if 'track_' in path_str:
                    track_id = path_str.split('track_')[1].split('/')[0] 
                else:
                    track_id = Path(path_str).stem 
            except Exception:
                track_id = filename
                
            if filename.startswith('hr-') or 'hr' in path_str.lower() or 'rodosol' in path_str.lower():
                self.track_dict[track_id]['hr'].append(item)
            else:
                self.track_dict[track_id]['lr'].append(item)
                
                # ---> OPTIMIZATION: Only build the FDA style pool if FDA is actually enabled!
                if (self.use_fda_hr or self.use_fda_lr) and not self.test:
                    if len(self.lr_pool) < 50000:
                        img_raw = item['img_raw']
                        if img_raw is not None and len(img_raw.shape) == 3:
                            self.lr_pool.append(img_raw)
                
        self.valid_tracks = [
            tid for tid, data in self.track_dict.items() 
            if len(data['lr']) > 0 or len(data['hr']) > 0
        ]
        
        print(f"📦 Grouped dataset into {len(self.valid_tracks)} unique tracks for 5x faster epochs.")
        if self.use_fda_hr or self.use_fda_lr:
            print(f"📊 FDA Style Pool Size: {len(self.lr_pool)}")
        else:
            print("⚡ FDA Augmentations Disabled: Skipped building style pool.")

    def __len__(self):
        return len(self.valid_tracks)

    def __getitem__(self, idx):
        track_id = self.valid_tracks[idx]
        track_data = self.track_dict[track_id]
        
        has_lr = len(track_data['lr']) > 0
        has_hr = len(track_data['hr']) > 0
        
        # =========================================================
        # 2. FRAME SELECTION (Dynamic Routing)
        # =========================================================
        if has_lr:
            item_lr = random.choice(track_data['lr'])
            img_raw = item_lr['img_raw'].copy()   
            plate_gt = item_lr['gt']       
            filename = item_lr['name']
            is_hr_file = False
            
            if self.return_hr:
                if has_hr:
                    item_hr = random.choice(track_data['hr'])
                    img_hr_clean = item_hr['img_raw'].copy()
                else:
                    img_hr_clean = img_raw.copy() 
        else:
            item_hr = random.choice(track_data['hr'])
            img_raw = item_hr['img_raw'].copy() 
            plate_gt = item_hr['gt']
            filename = item_hr['name']
            is_hr_file = True 
            
            if self.return_hr:
                img_hr_clean = item_hr['img_raw'].copy()

        # =========================================================
        # 3. THE DEGRADATION BRIDGE (Now Optional!)
        # =========================================================
        if self.aug and not self.test and len(self.lr_pool) > 0:
            
            # ---> THE FIX: Gated behind the self.use_fda_hr flag
            if is_hr_file and self.use_fda_hr:
                try:
                    degrader = FourierCCTVDegradation(
                        lr_image_pool=self.lr_pool, 
                        beta_range=(0.01, 0.08),
                        jpeg_range=(65, 85)
                    )
                    img_raw = degrader.apply(img_raw)
                    
                    # if random.random() < 0.3:
                    #     bc = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(-0.1, 0.1), p=1.0)
                    #     img_raw = bc(image=img_raw)['image']
                except Exception as e:
                    print(f"❌ FDA CRASH: {e}")
                    
            # ---> THE FIX: Gated behind the self.use_fda_lr flag
            elif not is_hr_file and self.use_fda_lr:
                if random.random() < 0.5:
                    try:
                        lr_mixer = FourierLRtoLRMixup(
                            lr_image_pool=self.lr_pool, 
                            beta_range=(0.001, 0.02)
                        )
                        img_raw = lr_mixer.apply(img_raw)
                    except Exception as e:
                        print(f"❌ LR-to-LR Mixup CRASH: {e}")

        # Final Resizes for the Student
        img_resized = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)

        # =========================================================
        # 6. TEACHER SOFTENING & JOINT GEOMETRIC AUGMENTATION
        # =========================================================
        if self.aug and not self.test:
            if self.return_hr:
                # Stream A: What the Teacher actually looks at
                img_hr_teacher_input = img_hr_clean.copy()
                
                if len(self.lr_pool) > 0 and random.random() < 0.5:
                    try:
                        mild_degrader = FourierCCTVDegradation(
                            lr_image_pool=self.lr_pool, 
                            beta_range=(0.001, 0.02),
                            apply_jpeg=False          
                        )
                        img_hr_teacher_input = mild_degrader.apply(img_hr_teacher_input)
                    except Exception as e:
                        pass
                
                # Resize the softened image for the Teacher's network
                img_hr_resized = cv2.resize(img_hr_teacher_input, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
                
                # ---> THE FIX 1: Temporarily resize the Ground Truth to perfectly match the base dimensions!
                img_sr_base = cv2.resize(img_hr_clean, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)

                # ---> THE FIX 2: Safely apply the joint rotation to identical shapes!
                augmented = self.geo_aug(image=img_resized, image_hr=img_hr_resized, image_sr=img_sr_base)
                
                img_resized = augmented['image']
                img_hr_resized = augmented['image_hr']
                
                # ---> THE FIX 3: Scale the perfectly-aligned SR Ground Truth up to its final resolution!
                # Note: Multiply imgW and imgH by whatever your LatentUpsampler upscale_factor is (e.g., 2)
                upscale_factor = 1 
                sr_W, sr_H = self.imgW * upscale_factor, self.imgH * upscale_factor
                img_sr_gt = cv2.resize(augmented['image_sr'], (sr_W, sr_H), interpolation=cv2.INTER_CUBIC)
                
            else:
                augmented = self.geo_aug(image=img_resized)
                img_resized = augmented['image']

        # =========================================================
        # 7. TENSOR CONVERSION & LR-SAFE AUGMENTATION
        # =========================================================        
        
        t_lr = ToTensor()(img_resized.copy())
        
        if self.aug and not self.test:
            t_lr = self.color_aug(t_lr)
            
        # 3. Normalize to (-1.0 to 1.0) FIRST
        t_lr = self.normalize(t_lr)
        
        # ---> THE FIX: Apply Token-Aware Masking AFTER normalization
        # so dropped patches become 0.0 (Neutral Grey) instead of -1.0 (Black)
        # if self.aug and not self.test:
        #     if random.random() < self.eraser_prob:
        #         token_masker = TokenAwareMasking(patch_size=8, mask_ratio_range=(0.1, 0.25), p=1.0)
        #         t_lr = token_masker(t_lr)
        
        out_dict = {
            'lr': t_lr,       
            'gt': plate_gt,
            'name': filename,
            'is_hr': is_hr_file
        }
        
        if self.return_hr:
            out_dict['hr'] = self.normalize(ToTensor()(img_hr_resized.copy()))
            out_dict['hr_gt'] = self.normalize(ToTensor()(img_sr_gt.copy()))
            
        return out_dict
    def collate_fn(self, batch):
        out_batch = {
            'lr': torch.stack([b['lr'] for b in batch]),
            'gt': [b['gt'] for b in batch],
            'name': [b['name'] for b in batch],
            'is_hr': torch.tensor([b['is_hr'] for b in batch], dtype=torch.bool) 
        }
        
        if 'hr' in batch[0]:
            out_batch['hr'] = torch.stack([b['hr'] for b in batch])
            out_batch['hr_gt'] = torch.stack([b['hr_gt'] for b in batch]) 
            
        return out_batch

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
        self.in_images = in_images 
        
        assert dataset is not None, "Dataset is None"

        from collections import defaultdict
        self.sequences = defaultdict(list)
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            path_str = str(item['img_path'])
            try:
                track_id = path_str.split('track_')[1].split('/')[0] 
            except IndexError:
                track_id = Path(path_str).parent.name 
                
            self.sequences[track_id].append(item)
            
        self.grouped_dataset = list(self.sequences.values())

    def __len__(self):
        return len(self.grouped_dataset)

    def __getitem__(self, idx):
        raw_sequence_items = self.grouped_dataset[idx]
        
        sequence_items = [item for item in raw_sequence_items if not item['name'].startswith('hr-')]
        # ---> NEW: Grab the HR image for validation!
        hr_items = [item for item in raw_sequence_items if item['name'].startswith('hr-')]
        
        if len(sequence_items) == 0:
            raise ValueError(f"Track index {idx} has no LR images!")
        
        if len(sequence_items) < self.in_images:
            sequence_items.extend([sequence_items[-1]] * (self.in_images - len(sequence_items)))
        elif len(sequence_items) > self.in_images:
            sequence_items = sequence_items[:self.in_images]

        lr_tensors = []
        gts = []
        names = []
        
        for item in sequence_items:
            img_raw = item['img_raw'].copy()
            lr_img = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
            lr_tensors.append(self.normalize(ToTensor()(lr_img.copy())))
            gts.append(item['gt'])
            names.append(item['name'])

        # ---> NEW: Process the HR image
        if len(hr_items) > 0:
            hr_img_raw = hr_items[0]['img_raw'].copy()
        else:
            # Fallback to the LR image if no HR exists for this track
            hr_img_raw = sequence_items[0]['img_raw'].copy()
            
        hr_img_resized = cv2.resize(hr_img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
        t_hr = self.normalize(ToTensor()(hr_img_resized.copy()))
        
        # Duplicate the HR image to match the sequence length expected by the model
        hr_tensors = [t_hr] * len(lr_tensors)

        return {
            'lr_seq': torch.stack(lr_tensors),       
            'hr_seq': torch.stack(hr_tensors), # <--- NEW
            'gt': gts[0], 
            'names': names
        }

    def collate_fn(self, batch):
        return {
            'lr_seq': torch.stack([b['lr_seq'] for b in batch]),
            'hr_seq': torch.stack([b['hr_seq'] for b in batch]), # <--- NEW
            'gt': [b['gt'] for b in batch],
            'names': [b['names'] for b in batch]
        }
