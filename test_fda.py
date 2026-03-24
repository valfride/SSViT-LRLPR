import argparse
import torch
import torch.fft
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
import torchvision.transforms as T
import io

def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r, g, b = image.unbind(dim=-3)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    return torch.stack((y, cb, cr), dim=-3)

def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    y, cb, cr = image.unbind(dim=-3)
    cb = cb - 0.5
    cr = cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=-3)

def match_color_and_contrast(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Matches the global lighting, contrast, and color tint of the target image."""
    src_mean, src_std = src.mean(dim=(-2, -1), keepdim=True), src.std(dim=(-2, -1), keepdim=True)
    tgt_mean, tgt_std = target.mean(dim=(-2, -1), keepdim=True), target.std(dim=(-2, -1), keepdim=True)
    
    matched = (src - src_mean) / (src_std + 1e-8)
    matched = (matched * tgt_std) + tgt_mean
    return matched.clamp(0, 1)

def fourier_style_transfer(hr_img, lr_img, beta=0.05):
    # 1. Resize LR to match HR exactly
    lr_resized = F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bilinear', align_corners=False)
    
    # 2. Convert to YCbCr and extract Luminance (Y)
    hr_ycbcr = rgb_to_ycbcr(hr_img)
    lr_ycbcr = rgb_to_ycbcr(lr_resized)
    
    hr_y = hr_ycbcr[:, 0:1, :, :]
    lr_y = lr_ycbcr[:, 0:1, :, :]
    
    # 3. Compute 2D FFT
    fft_hr = torch.fft.fftn(hr_y, dim=(-2, -1))
    fft_lr = torch.fft.fftn(lr_y, dim=(-2, -1))
    
    # 4. Shift frequencies to center
    fft_hr_shifted = torch.fft.fftshift(fft_hr, dim=(-2, -1))
    fft_lr_shifted = torch.fft.fftshift(fft_lr, dim=(-2, -1))
    
    # 5. Extract Amplitude and Phase
    amp_hr, phase_hr = torch.abs(fft_hr_shifted), torch.angle(fft_hr_shifted)
    amp_lr = torch.abs(fft_lr_shifted)
    
    # 6. Create Gaussian Mask (Ensure center uses integer division)
    _, _, H, W = hr_img.shape
    c_h, c_w = H // 2, W // 2
    sigma = min(H, W) * beta
    
    Y, X = torch.meshgrid(
        torch.arange(H, device=hr_img.device), 
        torch.arange(W, device=hr_img.device), 
        indexing='ij'
    )
    dist_sq = (X - c_w)**2 + (Y - c_h)**2
    mask = torch.exp(-dist_sq / (2 * (sigma**2)))
    mask = mask.unsqueeze(0).unsqueeze(0) 
    
    # 7. Blend Amplitudes
    amp_hr_mixed = (amp_lr * mask) + (amp_hr * (1 - mask))
    
    # 8. Reconstruct the Y channel
    fft_mixed = amp_hr_mixed * torch.exp(1j * phase_hr)
    fft_mixed_unshifted = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
    degraded_y = torch.real(torch.fft.ifftn(fft_mixed_unshifted, dim=(-2, -1)))
    
    # --- THE ARTIFACT KILLER ---
    # Force the new Y channel to match the exact statistical distribution of the original.
    # This prevents the brightness from decoupling from the color channels!
    deg_mean, deg_std = degraded_y.mean(dim=(-2, -1), keepdim=True), degraded_y.std(dim=(-2, -1), keepdim=True)
    hr_mean, hr_std = hr_y.mean(dim=(-2, -1), keepdim=True), hr_y.std(dim=(-2, -1), keepdim=True)
    
    degraded_y = (degraded_y - deg_mean) / (deg_std + 1e-8)
    degraded_y = (degraded_y * hr_std) + hr_mean
    
    # Strictly clamp the Luminance before mixing back with colors
    degraded_y = degraded_y.clamp(0, 1)
    # ---------------------------
    
    # 9. Recombine with the original HR Color Channels (Cb, Cr)
    hr_ycbcr_mixed = hr_ycbcr.clone()
    hr_ycbcr_mixed[:, 0:1, :, :] = degraded_y
    
    # 10. Convert back to RGB and final clamp
    final_rgb = ycbcr_to_rgb(hr_ycbcr_mixed)
    return final_rgb.clamp(0, 1)

def main():
    parser = argparse.ArgumentParser(description="Fourier Domain Adaptation for Plate Degradation")
    parser.add_argument("--hrpath", type=str, required=True, help="Path to the clean High-Res image")
    parser.add_argument("--lrpath", type=str, required=True, help="Path to the blurry/noisy Low-Res image")
    parser.add_argument("--savepath", type=str, default="degraded_output.png", help="Where to save the result")
    parser.add_argument("--beta", type=float, default=0.05, help="Strength of the degradation transfer (0.01 to 0.15)")
    # --- NEW: Command line control for blockiness ---
    parser.add_argument("--jpeg", type=int, default=90, help="JPEG compression quality (1-100). Lower = more blocky.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_hr = TF.to_tensor(Image.open(args.hrpath).convert("RGB")).unsqueeze(0).to(device)
    img_lr = TF.to_tensor(Image.open(args.lrpath).convert("RGB")).unsqueeze(0).to(device)

    # 1. OPTICAL LENS SIMULATION (Slightly stronger blur)
    blur_transform = T.GaussianBlur(kernel_size=(7, 7), sigma=(2.5, 2.5))
    img_hr_blurred = blur_transform(img_hr)

    # 2. SENSOR INTEGRATION (Changed 'area' to 'bilinear' for a smoother optical smear)
    lr_h, lr_w = img_lr.shape[-2:]
    img_hr_physical = F.interpolate(img_hr_blurred, size=(lr_h, lr_w), mode='bicubic', align_corners=False).clamp(0, 1)

    # 3. COLOR & LIGHTING MATCH
    img_hr_color_matched = match_color_and_contrast(img_hr_physical, img_lr)

    # 4. ENVIRONMENTAL NOISE (FDA)
    degraded_lowres = fourier_style_transfer(img_hr_color_matched, img_lr, beta=args.beta)

    # 5. DIGITAL COMPRESSION (Controlled by args.jpeg)
    pil_img = TF.to_pil_image(degraded_lowres.squeeze(0).cpu())
    buffer = io.BytesIO()
    
    # Use the new argument here
    pil_img.save(buffer, format="JPEG", quality=args.jpeg) 
    
    final_img = Image.open(buffer)
    final_img.save(args.savepath)
    
    print(f"✅ Success! Saved to: {args.savepath} (Beta: {args.beta}, JPEG Quality: {args.jpeg})")

if __name__ == "__main__":
    main()