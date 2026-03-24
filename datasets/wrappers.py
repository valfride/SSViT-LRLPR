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

# --- NEW: Dynamically add synEngine to the Python Path ---
CURRENT_DIR = Path(__file__).resolve().parent
SYN_ENGINE_DIR = CURRENT_DIR / "synEngine"
sys.path.append(str(SYN_ENGINE_DIR))
try:
    from PhysicalPlateGenerator import PhysicalPlateGenerator
except ImportError:
    print("⚠️ Warning: PhysicalPlateGenerator not found. Synthetic generation disabled.")

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
    """Transfers style/degradation of a real LR crop to an HR image."""
    def __init__(self, lr_image_pool, beta_range=(0.05, 0.10), jpeg_range=(55, 65), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.lr_image_pool = lr_image_pool 
        self.beta_range = beta_range
        self.jpeg_range = jpeg_range

    def apply(self, img, **params):
        target_lr_np = random.choice(self.lr_image_pool)
        beta = random.uniform(self.beta_range[0], self.beta_range[1])
        jpeg_quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
        blur_sigma = random.uniform(2.0, 3.0)

        img_hr = TF.to_tensor(img).unsqueeze(0)
        img_lr = TF.to_tensor(target_lr_np).unsqueeze(0)

        img_hr_blurred = T.GaussianBlur(kernel_size=(7, 7), sigma=(blur_sigma, blur_sigma))(img_hr)
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
        mask = torch.exp(-dist_sq / (2 * (sigma**2))).unsqueeze(0).unsqueeze(0)
        
        amp_hr_mixed = (amp_lr * mask) + (amp_hr * (1 - mask))
        fft_mixed = amp_hr_mixed * torch.exp(1j * phase_hr)
        fft_mixed_unshifted = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
        degraded_y = torch.real(torch.fft.ifftn(fft_mixed_unshifted, dim=(-2, -1)))
        
        deg_mean, deg_std = degraded_y.mean(dim=(-2, -1), keepdim=True), degraded_y.std(dim=(-2, -1), keepdim=True)
        hr_mean, hr_std = hr_y.mean(dim=(-2, -1), keepdim=True), hr_y.std(dim=(-2, -1), keepdim=True)
        degraded_y = ((degraded_y - deg_mean) / (deg_std + 1e-8) * hr_std) + hr_mean
        degraded_y = degraded_y.clamp(0, 1)
        
        hr_ycbcr_mixed = hr_ycbcr.clone()
        hr_ycbcr_mixed[:, 0:1, :, :] = degraded_y
        final_rgb = _ycbcr_to_rgb(hr_ycbcr_mixed).clamp(0, 1)

        pil_img = TF.to_pil_image(final_rgb.squeeze(0))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=jpeg_quality)
        
        return np.array(Image.open(buffer))

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
                use_cache=False, skip_low_res=False, dataset=None, **kwargs):
        
        self.imgW = imgW      
        self.imgH = imgH      
        self.aug = aug 
        self.dataset = dataset
        self.test = test
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        assert self.dataset is not None, "Dataset is None"

        self.geo_aug = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=(-0.02, 0.02), 
                rotate_limit=5,    
                p=0.5, 
                border_mode=cv2.BORDER_REPLICATE
            ),
        ])

        # 1. Gather a diverse pool of LR styles (ONE per track!)
        self.lr_pool = []
        target_pool_size = 5000
        seen_tracks = set() 
        
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        for idx in indices:
            if len(self.lr_pool) >= target_pool_size:
                break 
                
            item = self.dataset[idx]
            filename = item.get('name', '')
            
            if filename.startswith('lr-'):
                path_str = str(item.get('img_path', ''))
                try:
                    track_id = path_str.split('track_')[1].split('/')[0] 
                except IndexError:
                    track_id = Path(path_str).parent.name 
                
                if track_id not in seen_tracks:
                    img_raw = item['img_raw']
                    if img_raw is not None and len(img_raw.shape) == 3:
                        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                        self.lr_pool.append(img_rgb)
                        seen_tracks.add(track_id) 
        
        if len(self.lr_pool) == 0:
            print("⚠️ WARNING: Could not find any unique 'lr-' tracks for the FDA pool!")
        else:
            print(f"📊 Successfully built FDA Style Pool using {len(self.lr_pool)} unique tracks.")

        self.syn_prob = synthetic_prob
        if self.syn_prob > 0.0:
            asset_path = str(SYN_ENGINE_DIR / "assets")
            print(f"🚀 Booting Synthetic Engine (Chance: {self.syn_prob * 100}%)")
            self.syn_engine = PhysicalPlateGenerator(asset_dir=asset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_raw = item['img_raw'].copy()   
        plate_gt = item['gt']          
        filename = item['name']
        
        is_hr_file = filename.startswith("hr-")

        if getattr(self, 'syn_prob', 0.0) > 0 and random.random() < self.syn_prob:
            try:
                import string
                if random.random() < 0.5:
                    letters = ''.join(random.choices(string.ascii_uppercase, k=3))
                    numbers = ''.join(random.choices(string.digits, k=4))
                    plate_gt = f"{letters}{numbers}"
                
                img_bgr, standard_label = self.syn_engine.generate(plate_gt)
                plate_gt = standard_label
                img_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                is_hr_file = True 
                filename = f"syn_hr_{plate_gt}.jpg" 
                
            except Exception as e:
                print(f"❌ ENGINE CRASH on '{plate_gt}': {repr(e)}")

        # --- STEP A: Augmentations via Domain Transfer ---
        if self.aug and not self.test and len(self.lr_pool) > 0:
            
            # STEP A.1: HR -> LR Degradation
            if is_hr_file:
                try:
                    degrader = FourierCCTVDegradation(
                        lr_image_pool=self.lr_pool, 
                        beta_range=(0.05, 0.10), 
                        jpeg_range=(55, 65)
                    )
                    img_raw = degrader.apply(img_raw)
                    
                    if random.random() < 0.3:
                        bc = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=(-0.1, 0.1), p=1.0)
                        img_raw = bc(image=img_raw)['image']
                except Exception as e:
                    print(f"❌ FDA CRASH: {e}")
                    
            # STEP A.2: LR -> LR Style Swap (30% Chance)
            elif not is_hr_file:
                if random.random() < 0.5:
                    try:
                        lr_mixer = FourierLRtoLRMixup(
                            lr_image_pool=self.lr_pool, 
                            beta_range=(0.02, 0.08) 
                        )
                        img_raw = lr_mixer.apply(img_raw)
                    except Exception as e:
                        print(f"❌ LR-to-LR Mixup CRASH: {e}")

        # STEP B: Final Resize (Always happens to guarantee tensor shape)
        img_resized = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)

        # STEP C: Geometric Augmentation (Shift/Scale/Rotate)
        if self.aug and not self.test:
            augmented = self.geo_aug(image=img_resized)
            img_resized = augmented['image']

        # Convert to Tensor and Normalize [-1, 1]
        t_lr = self.normalize(ToTensor()(img_resized.copy()))
        
        return {
            'lr': t_lr,       
            'gt': plate_gt,
            'name': filename,
            'is_hr': is_hr_file
        }

    def collate_fn(self, batch):
        return {
            'lr': torch.stack([b['lr'] for b in batch]),
            'gt': [b['gt'] for b in batch],
            'name': [b['name'] for b in batch],
            'is_hr': torch.tensor([b['is_hr'] for b in batch], dtype=torch.bool) 
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
        
        if len(sequence_items) == 0:
            raise ValueError(f"Track index {idx} has no LR images! (Only found HR files or it was empty)")
        
        if len(sequence_items) < self.in_images:
            sequence_items.extend([sequence_items[-1]] * (self.in_images - len(sequence_items)))
        elif len(sequence_items) > self.in_images:
            sequence_items = sequence_items[:self.in_images]

        lr_tensors = []
        gts = []
        names = []
        
        for item in sequence_items:
            img_raw = item['img_raw']
            lr_img = cv2.resize(img_raw, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
            t_lr = self.normalize(ToTensor()(lr_img.copy()))
            
            lr_tensors.append(t_lr)
            gts.append(item['gt'])
            names.append(item['name'])

        sequence_tensor = torch.stack(lr_tensors)
        
        return {
            'lr_seq': sequence_tensor,       
            'gt': gts[0], 
            'names': names
        }

    def collate_fn(self, batch):
        return {
            'lr_seq': torch.stack([b['lr_seq'] for b in batch]),
            'gt': [b['gt'] for b in batch],
            'names': [b['names'] for b in batch]
        }