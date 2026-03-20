import os
import cv2
import random
import numpy as np
import glob
import json
import string
import argparse
from pathlib import Path

class PhysicalPlateGenerator:
    def __init__(self, asset_dir="./datasets/synEngine/assets", target_size=(103, 32)):
        self.asset_dir = Path(asset_dir)
        self.font_dir = self.asset_dir / "real_fonts" / "Brazilian"
        self.bg_dir = self.asset_dir / "inpainted_backgrounds" / "Brazilian"
        
        self.target_width, self.target_height = target_size
        self.assets = {"bgs": [], "chars": {}}
        self.layout = None
        
        # MERCOSUR -> BRAZILIAN MAPPING
        self.m_to_b = {chr(65 + i): str(i) for i in range(10)} 
        
        self._index_assets()
        self._load_layout()

    def to_brazilian_label(self, text):
        # Clean string: Only keep pure alphanumeric characters
        clean_text = "".join(c for c in str(text).upper() if c.isalnum())
        chars = list(clean_text)
        
        if len(chars) == 7 and chars[4] in self.m_to_b:
            chars[4] = self.m_to_b[chars[4]]
        return "".join(chars)

    def _index_assets(self):
        alphabet = string.digits + string.ascii_uppercase
        self.assets["bgs"] = glob.glob(str(self.bg_dir / "*.jpg"))
        for char in alphabet:
            self.assets["chars"][char] = glob.glob(str(self.font_dir / char / "*.jpg"))

    def _load_layout(self):
        layout_file = self.asset_dir / "layout_Brazilian.json"
        if layout_file.exists():
            with open(layout_file, 'r') as f:
                self.layout = json.load(f)

            # --- 📏 1. THE "PERFECT BASELINE" FIX ---
            # Calculate a single, global Y coordinate and Height for ALL characters
            global_y = int(np.mean([m['y'] for m in self.layout]))
            global_h = int(np.mean([m['h'] for m in self.layout]))
            
            # Ensure the characters don't bleed off the bottom of the plate (prevents chopping)
            if global_y + global_h > self.target_height:
                global_y = self.target_height - global_h
                
            # Force every slot to use this exact same vertical alignment
            for m in self.layout:
                m['y'] = max(0, global_y)
                m['h'] = global_h

    def _post_process(self, plate):
        lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(2,2))
        plate = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        
        quality = random.randint(75, 95)
        _, encoded_img = cv2.imencode('.jpg', plate, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encoded_img, 1)

    def generate(self, text):
        render_text = text.upper()
        standard_label = self.to_brazilian_label(render_text)

        bg_path = random.choice(self.assets["bgs"])
        plate = cv2.imread(bg_path)
        plate = cv2.resize(plate, (self.target_width, self.target_height))
        
        # --- 🎨 1. GLOBAL INK EQUALIZATION ---
        # Calculate a uniform ink color for the ENTIRE plate.
        # Ink is usually ~25% the brightness of the plate's ambient color.
        plate_median_color = np.median(plate, axis=(0, 1))
        global_ink_color = plate_median_color * 0.25
        
        for idx, char in enumerate(render_text):
            char_paths = self.assets["chars"].get(char, [])
            if not char_paths: continue
            
            char_crop = cv2.imread(random.choice(char_paths))
            
            # Since extract_real_assets.py already standardized sizes, 
            # we just do one clean snap to the layout bounds.
            m = self.layout[idx]
            cw, ch = max(1, m['w']), max(1, m['h'])
            char_crop = cv2.resize(char_crop, (cw, ch), interpolation=cv2.INTER_AREA)
            
            y, x = m['y'], m['x']
            y2, x2 = min(self.target_height, y+ch), min(self.target_width, x+cw)
            
            roi_h, roi_w = y2 - y, x2 - x
            if roi_h <= 0 or roi_w <= 0: continue
            
            bg_roi = plate[y:y2, x:x2].astype(float)
            patch_f = char_crop[:roi_h, :roi_w].astype(float)

            # --- 🎭 2. STENCIL MASK EXTRACTION ---
            gray_patch = cv2.cvtColor(patch_f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
            
            bg_val = (np.mean(gray_patch[0, :]) + np.mean(gray_patch[-1, :]) + 
                    np.mean(gray_patch[:, 0]) + np.mean(gray_patch[:, -1])) / 4
            ink_val = np.percentile(gray_patch, 5) # Darkest 5% is the ink
            
            # Safety check: If the crop is a blank error patch, skip it
            if bg_val - ink_val < 10: 
                continue
                
            # Normalize mask: Background = 0.0, Ink = 1.0
            mask = (bg_val - gray_patch) / (bg_val - ink_val)
            mask = np.clip(mask, 0, 1)
            
            # --- 📈 3. ANTI-THINNING (GAMMA CORRECTION) ---
            # Using a power < 1 thickens the midtones of the mask, plumping up thin letters
            # without making them look like blocky, jagged pixels.
            mask = np.power(mask, 0.65) 
            
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            mask_3c = np.stack([mask]*3, axis=-1)

            # --- 🖌️ 4. APPLY UNIFORM INK ---
            ink_layer = np.full_like(bg_roi, global_ink_color)
            # Inject a tiny bit of the plate's physical grain into the ink for realism
            noise = (bg_roi - np.median(bg_roi)) * 0.15
            ink_layer = np.clip(ink_layer + noise, 0, 255)

            # Blend the uniform ink layer using the extracted stencil
            blended = (ink_layer * mask_3c) + (bg_roi * (1.0 - mask_3c))
            
            plate[y:y2, x:x2] = blended.astype(np.uint8)

        return self._post_process(plate), standard_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    
    engine = PhysicalPlateGenerator()
    out_dir = Path("./test_synthetic_outputs")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for i in range(args.n):
        is_merc = random.choice([True, False])
        letters = "".join(random.choices(string.ascii_uppercase, k=3))
        
        if is_merc:
            n1 = random.choice(string.digits)
            l1 = random.choice(string.ascii_uppercase[0:10])  
            n2 = "".join(random.choices(string.digits, k=2))
            input_txt = letters + n1 + l1 + n2
        else:
            input_txt = letters + "".join(random.choices(string.digits, k=4))
            
        try:
            hr_img, label = engine.generate(input_txt)
            cv2.imwrite(str(out_dir / f"syn_{label}.jpg"), hr_img)
            print(f"[{i+1}/{args.n}] ✅ Rendered: {input_txt} -> Saved as: syn_{label}.jpg")
        except Exception as e:
            print(f"[{i+1}/{args.n}] ❌ Failed on {input_txt}: {e}")