import argparse
import torch
import torch.fft
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os

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

def fourier_lr_mixup(content_img, style_img, beta=0.05):
    """
    Swaps the lighting, color, and sensor noise of style_img into content_img.
    """
    # 1. Ensure dimensions match perfectly
    style_resized = F.interpolate(style_img, size=content_img.shape[-2:], mode='bilinear', align_corners=False)

    # 2. Match global color
    matched_content = _match_color_and_contrast(content_img, style_resized)

    # 3. YCbCr Conversion
    content_ycbcr = _rgb_to_ycbcr(matched_content)
    style_ycbcr = _rgb_to_ycbcr(style_resized)
    
    content_y = content_ycbcr[:, 0:1, :, :]
    style_y = style_ycbcr[:, 0:1, :, :]
    
    # 4. FFT
    fft_c = torch.fft.fftn(content_y, dim=(-2, -1))
    fft_s = torch.fft.fftn(style_y, dim=(-2, -1))
    
    fft_c_shifted = torch.fft.fftshift(fft_c, dim=(-2, -1))
    fft_s_shifted = torch.fft.fftshift(fft_s, dim=(-2, -1))
    
    amp_c, phase_c = torch.abs(fft_c_shifted), torch.angle(fft_c_shifted)
    amp_s = torch.abs(fft_s_shifted)
    
    # 5. Gaussian Mask
    _, _, H, W = content_img.shape
    c_h, c_w = H // 2, W // 2
    sigma = min(H, W) * beta
    
    Y, X = torch.meshgrid(torch.arange(H, device=content_img.device), torch.arange(W, device=content_img.device), indexing='ij')
    dist_sq = (X - c_w)**2 + (Y - c_h)**2
    mask = torch.exp(-dist_sq / (2 * (sigma**2))).unsqueeze(0).unsqueeze(0)
    
    # 6. Blend Amplitudes
    amp_mixed = (amp_s * mask) + (amp_c * (1 - mask))
    fft_mixed = amp_mixed * torch.exp(1j * phase_c)
    fft_mixed_unshifted = torch.fft.ifftshift(fft_mixed, dim=(-2, -1))
    degraded_y = torch.real(torch.fft.ifftn(fft_mixed_unshifted, dim=(-2, -1)))
    
    # 7. Statistical Lock
    deg_mean, deg_std = degraded_y.mean(dim=(-2, -1), keepdim=True), degraded_y.std(dim=(-2, -1), keepdim=True)
    c_mean, c_std = content_y.mean(dim=(-2, -1), keepdim=True), content_y.std(dim=(-2, -1), keepdim=True)
    degraded_y = ((degraded_y - deg_mean) / (deg_std + 1e-8) * c_std) + c_mean
    degraded_y = degraded_y.clamp(0, 1)
    
    # 8. Recombine RGB
    content_ycbcr_mixed = content_ycbcr.clone()
    content_ycbcr_mixed[:, 0:1, :, :] = degraded_y
    return _ycbcr_to_rgb(content_ycbcr_mixed).clamp(0, 1)

def main():
    parser = argparse.ArgumentParser(description="LR to LR Fourier Mixup Test")
    parser.add_argument("--content", type=str, required=True, help="Path to Content LR image (provides the text geometry)")
    parser.add_argument("--style", type=str, required=True, help="Path to Style LR image (provides the noise/lighting)")
    parser.add_argument("--savepath", type=str, default="mixup_demo.png", help="Where to save the side-by-side grid")
    parser.add_argument("--beta", type=float, default=0.05, help="Strength of the noise transfer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images
    img_content = TF.to_tensor(Image.open(args.content).convert("RGB")).unsqueeze(0).to(device)
    img_style = TF.to_tensor(Image.open(args.style).convert("RGB")).unsqueeze(0).to(device)

    # Apply the math
    img_mixed = fourier_lr_mixup(img_content, img_style, beta=args.beta)

    # Create a side-by-side comparison image: [Content] | [Style Target] | [Mixed Output]
    style_resized_for_grid = F.interpolate(img_style, size=img_content.shape[-2:], mode='bilinear', align_corners=False)
    
    # Concatenate along the width (dim=3)
    grid = torch.cat([img_content, style_resized_for_grid, img_mixed], dim=3)
    
    out_pil = TF.to_pil_image(grid.squeeze(0).cpu())
    out_pil.save(args.savepath)
    
    print(f"✅ Success! Saved side-by-side demo (Content | Style | Output) to: {args.savepath}")

if __name__ == "__main__":
    main()