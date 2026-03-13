import numpy as np
import cv2
import random
import os
import string
import albumentations as A
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

# ==============================================================================
# I. Advanced Physics Motion Blur (Transplanted from wrappers.py)
# ==============================================================================
class AdvancedPhysicsMotionBlur(A.ImageOnlyTransform):
    """
    Simulates physically accurate vehicle motion blur as defined in wrappers.py.
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
        start_x, end_x = center - velocity // 2, center + velocity // 2
        
        blur_type = random.random()
        if blur_type < 0.33:
            kernel[center, start_x:end_x] = 1.0
        elif blur_type < 0.66:
            fade = np.linspace(1.0, 0.1, end_x - start_x)
            if random.random() < 0.5: fade = fade[::-1] 
            kernel[center, start_x:end_x] = fade
        else:
            x_coords = np.linspace(-3, 3, end_x - start_x)
            kernel[center, start_x:end_x] = np.exp(-0.5 * (x_coords ** 2))
            
        rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (ksize, ksize))
        return cv2.filter2D(img, -1, kernel / np.sum(kernel))

# ==============================================================================
# II. Organic Plate Sequence Generator
# ==============================================================================
class OrganicPlateSequenceGenerator:
    def __init__(self, fonts_dir="./fonts"):
        self.fonts_dir = Path(fonts_dir)
        
        # 🟢 1. GLOBAL RATIO CONFIGS (Tuned for Scenario-B)
        self.configs = {
            "grey": {
                "bg": self.fonts_dir / 'brazilian.png',
                "font": self.fonts_dir / 'mandatory' / 'MANDATOR.ttf',
                "ratio_anchor": (0.35, 0.46, 0.3, 0.3),
                "ratio_char": (0.2, 0.49),
                "ratio_spacing": -0.075,
                "split": 3,
                "ratio_gap": 0.08 
            },
            "mercosur": {
                "bg": self.fonts_dir / 'mercosur.png',
                "font": self.fonts_dir / 'fe_font' / 'FE-FONT.ttf',
                "ratio_anchor": (0.38, 0.5, 0.3, 0.3),
                "ratio_char": (0.11, 0.48),
                "ratio_spacing": 0.01,
                "split": 7,
                "ratio_gap": 0.0
            }
        }

        # 🟢 2. VALIDATION BOUNDS
        self.val_bounds = {"min_w": 90, "max_w": 160, "min_ar": 2.6, "max_ar": 3.1}

        # 🟢 3. ORGANIC JITTER RATIOS
        self.jitter = {"img_w": 0.01, "img_h": 0.01, "char_w": 0.008, "char_h": 0.008}

        # 🟢 4. BALANCED HR REALISM
        self.hr_realism = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.6),
            A.ISONoise(color_shift=(0.01, 0.01), intensity=(0.01, 0.03), p=0.4),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2), 
            A.ImageCompression(quality_range=(95, 100), p=0.5),
        ])

        # 🟢 5. LR DEGRADATION PIPELINE (Matches wrappers.py)
        self.lr_degradation = A.Compose([
            A.Downscale(scale_range=[0.3, 0.3], p=1.0),
            AdvancedPhysicsMotionBlur(velocity_range=(3, 8), angle_range=(-10, 10), p=1.0),
            A.ImageCompression(quality_range=[40, 60], p=1.0),
        ])

    def _generate_random_text(self, style):
        """Generates random strings following official patterns."""
        L = lambda: random.choice(string.ascii_uppercase)
        N = lambda: random.choice(string.digits)
        if style == "grey":
            return f"{L()}{L()}{L()}{N()}{N()}{N()}{N()}"
        return f"{L()}{L()}{L()}{N()}{random.choice('ABCDEFGHIJ')}{N()}{N()}"

    def _apply_environmental_effects(self, pil_img, env_state):
        """Applies consistent, moving environmental lighting and shadows."""
        w, h = pil_img.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Moving shadow
        s_x = int(env_state['shadow_x'] * w)
        s_w = int(env_state['shadow_w'] * w)
        draw.rectangle([s_x, 0, s_x + s_w, h], fill=(0, 0, 0, env_state['shadow_alpha']))

        # Dynamic Light Focus
        l_x = int(env_state['light_x'] * w)
        l_r = int(w * 0.4)
        for r in range(l_r, 0, -5):
            alpha = int((1 - r/l_r) * env_state['light_alpha'])
            draw.ellipse([l_x - r, h//2 - r, l_x + r, h//2 + r], fill=(255, 255, 255, alpha))

        mask = overlay.filter(ImageFilter.GaussianBlur(radius=w*0.05))
        pil_img.paste(mask, (0, 0), mask)
        return pil_img

    def _draw_char_to_cell(self, ch, font_path, cell_size, padding=0, render_scale=4):
        """Renders individual characters with resolution-independent scaling."""
        cell_w, cell_h = cell_size
        R = max(cell_w, cell_h) * render_scale
        canvas_size = R * 2 
        tmp = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp)
        try: font = ImageFont.truetype(str(font_path), size=R)
        except: font = ImageFont.load_default()
        try:
            l, t, r, b = draw.textbbox((0, 0), ch, font=font)
            w, h = r - l, b - t
        except: w, h = draw.textsize(ch, font=font); l, t = 0, 0
        draw.text(((canvas_size - w) // 2 - l, (canvas_size - h) // 2 - t), ch, font=font, fill=(0, 0, 0, 255))
        
        # Fixed Image.crop() Type Error
        bbox = tmp.getbbox()
        if bbox is None: return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        glyph = tmp.crop(bbox)

        ratio = min((cell_w - 2 * padding) / glyph.width, (cell_h - 2 * padding) / glyph.height)
        new_w, new_h = int(glyph.width * ratio), int(glyph.height * ratio)
        scaled = glyph.resize((new_w, new_h), Image.LANCZOS)
        cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        cell.paste(scaled, ((cell_w - new_w)//2, (cell_h - new_h)//2), scaled)
        return cell

    def _generate_raw_plate(self, text, style, target_w, target_h, char_jitter=(0,0)):
        """Builds the plate image before environmental effects."""
        cfg = self.configs[style]
        char_w = int(target_w * cfg['ratio_char'][0]) + char_jitter[0]
        char_h = int(target_h * cfg['ratio_char'][1]) + char_jitter[1]
        spacing = int(target_w * cfg['ratio_spacing'])
        gap = int(target_w * cfg['ratio_gap'])
        
        bg = Image.open(cfg['bg']).convert("RGBA").resize((target_w, target_h), Image.LANCZOS)
        ax, ay, aw, ah = [int(target_w*cfg['ratio_anchor'][0]), int(target_h*cfg['ratio_anchor'][1]), 
                          int(target_w*cfg['ratio_anchor'][2]), int(target_h*cfg['ratio_anchor'][3])]
        
        total_w = (len(text) * char_w) + ((len(text)-1) * spacing) + gap
        curr_x, curr_y = ax + (aw - total_w) // 2, ay + (ah - char_h) // 2
        
        for i, char in enumerate(text):
            if cfg['split'] > 0 and i == cfg['split']:
                if gap > 0: # Draw central dot for Grey style
                    draw = ImageDraw.Draw(bg)
                    r = max(1, int(char_h * 0.075))
                    draw.ellipse([curr_x + gap//2 - r, curr_y + char_h//2 - r, 
                                  curr_x + gap//2 + r, curr_y + char_h//2 + r], fill=(80,80,80,255))
                curr_x += gap
            bg.alpha_composite(self._draw_char_to_cell(char, cfg['font'], (char_w, char_h)), (curr_x, curr_y))
            curr_x += char_w + spacing
        return bg.convert("RGB")

    # 🟢 NEW: In-Memory Generation Method (Crucial for online training)
    def generate_sequence_in_memory(self, style, n_frames):
        """
        Generates a sequence and returns it as a list of (HR, LR) numpy arrays.
        Does NOT write to disk.
        """
        text = self._generate_random_text(style)
        
        # 1. Sample Trajectory & Environment
        w_start = random.randint(self.val_bounds['min_w'], self.val_bounds['max_w'])
        w_end = random.randint(self.val_bounds['min_w'], self.val_bounds['max_w'])
        ar_start = random.uniform(self.val_bounds['min_ar'], self.val_bounds['max_ar'])
        ar_end = random.uniform(self.val_bounds['min_ar'], self.val_bounds['max_ar'])
        
        env_start = {'shadow_x': random.uniform(-0.5, 1.0), 'shadow_w': random.uniform(0.05, 0.2), 
                     'shadow_alpha': random.randint(40, 100), 'light_x': random.uniform(0.0, 1.0), 'light_alpha': random.randint(30, 70)}
        
        # Interpolation
        widths = np.linspace(w_start, w_end, n_frames).astype(int)
        ratios = np.linspace(ar_start, ar_end, n_frames)
        
        sequence_data = []

        for i in range(n_frames):
            # Jitter
            f_w = int(widths[i] * (1 + random.uniform(-self.jitter['img_w'], self.jitter['img_w'])))
            f_h = int((widths[i] / ratios[i]) * (1 + random.uniform(-self.jitter['img_h'], self.jitter['img_h'])))
            c_j = (int(f_w * random.uniform(-self.jitter['char_w'], self.jitter['char_w'])), 
                   int(f_h * random.uniform(-self.jitter['char_h'], self.jitter['char_h'])))
            
            # Env Params for this frame (simplified constant env for speed, or interpolate if needed)
            curr_env = env_start 

            # HR Generation
            raw_plate = self._generate_raw_plate(text, style, f_w, f_h, char_jitter=c_j)
            hr_pil = self._apply_environmental_effects(raw_plate, curr_env)
            hr_np = self.hr_realism(image=np.array(hr_pil))['image'] 
            
            # LR Generation
            lr_degraded = self.lr_degradation(image=hr_np)['image']
            lr_scale = random.uniform(2.6, 3.0)
            lr_img_np = cv2.resize(lr_degraded, (int(f_w/lr_scale), int(f_h/lr_scale)), interpolation=cv2.INTER_AREA)
            
            sequence_data.append((hr_np, lr_img_np))
            
        return sequence_data, text

    def generate_track(self, style, n_frames=5, output_dir="./synthetic_dataset"):
        """Generates a full track folder matching Scenario-B validation distribution (For offline use)."""
        text = self._generate_random_text(style)
        w_start, w_end = random.randint(self.val_bounds['min_w'], self.val_bounds['max_w']), random.randint(self.val_bounds['min_w'], self.val_bounds['max_w'])
        ar_start, ar_end = random.uniform(self.val_bounds['min_ar'], self.val_bounds['max_ar']), random.uniform(self.val_bounds['min_ar'], self.val_bounds['max_ar'])
        
        # Environmental Trajectory Sampling
        env_start = {'shadow_x': random.uniform(-0.5, 1.0), 'shadow_w': random.uniform(0.05, 0.2), 
                     'shadow_alpha': random.randint(40, 100), 'light_x': random.uniform(0.0, 1.0), 'light_alpha': random.randint(30, 70)}
        env_end = {'shadow_x': env_start['shadow_x'] + random.uniform(0.2, 0.5), 'shadow_w': env_start['shadow_w'], 
                   'shadow_alpha': env_start['shadow_alpha'], 'light_x': env_start['light_x'] + random.uniform(-0.3, 0.3), 'light_alpha': env_start['light_alpha']}

        track_folder = Path(output_dir) / style / f"track_{random.randint(10000, 99999)}"
        track_folder.mkdir(parents=True, exist_ok=True)
        with open(track_folder / "gt.txt", "w") as f: f.write(text)

        widths, ratios = np.linspace(w_start, w_end, n_frames).astype(int), np.linspace(ar_start, ar_end, n_frames)
        e_shadow_x, e_shadow_w, e_light_x = [np.linspace(env_start[k], env_end[k], n_frames) for k in ['shadow_x', 'shadow_w', 'light_x']]

        for i in range(n_frames):
            f_w = int(widths[i] * (1 + random.uniform(-self.jitter['img_w'], self.jitter['img_w'])))
            f_h = int((widths[i] / ratios[i]) * (1 + random.uniform(-self.jitter['img_h'], self.jitter['img_h'])))
            c_j = (int(f_w * random.uniform(-self.jitter['char_w'], self.jitter['char_w'])), 
                   int(f_h * random.uniform(-self.jitter['char_h'], self.jitter['char_h'])))
            
            curr_env = {'shadow_x': e_shadow_x[i], 'shadow_w': e_shadow_w[i], 'shadow_alpha': env_start['shadow_alpha'],
                        'light_x': e_light_x[i], 'light_alpha': env_start['light_alpha']}

            # HR Image
            raw_plate = self._generate_raw_plate(text, style, f_w, f_h, char_jitter=c_j)
            hr_pil = self._apply_environmental_effects(raw_plate, curr_env)
            hr_np = self.hr_realism(image=np.array(hr_pil))['image'] 
            
            # LR Image
            lr_degraded = self.lr_degradation(image=hr_np)['image']
            lr_scale = random.uniform(2.6, 3.0)
            lr_img_np = cv2.resize(lr_degraded, (int(f_w/lr_scale), int(f_h/lr_scale)), interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(str(track_folder / f"hr-{i+1:03d}.jpg"), cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(track_folder / f"lr-{i+1:03d}.jpg"), cv2.cvtColor(lr_img_np, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Track Generated: {text} ({style})")

if __name__ == "__main__":
    gen = OrganicPlateSequenceGenerator()
    for _ in range(5):
        gen.generate_track("grey")
        gen.generate_track("mercosur")