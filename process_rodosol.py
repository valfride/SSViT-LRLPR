import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="RodoSol-ALPR Expanded Crop Processor")
    parser.add_argument("--input", required=True, help="Path to the original split txt file")
    parser.add_argument("--output", required=True, help="Path to save the new flattened_rodosol.txt")
    parser.add_argument("--dest", required=True, help="Folder to save the cropped license plates")
    return parser.parse_args()

def process_rodosol():
    args = get_args()
    input_file = Path(args.input)
    output_path = Path(args.output)
    dest_dir = Path(args.dest)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()

    flattened_entries = []

    print(f"🚀 Processing RodoSol-ALPR from {input_file} (15% Expanded Crop Mode)...")
    
    for line in tqdm(lines):
        line = line.strip()
        if not line: continue
        
        parts = line.split(';')
        if len(parts) != 2: continue
        img_rel_path = parts[0]
        
        # Only process cars
        if 'cars-' not in img_rel_path:
            continue
            
        img_path = input_file.parent / img_rel_path
        label_path = img_path.with_suffix('.txt')
        
        if not img_path.exists() or not label_path.exists():
            continue

        label_data = {}
        with open(label_path, 'r') as lf:
            for l in lf:
                if ': ' in l:
                    k, v = l.strip().split(': ', 1)
                    label_data[k] = v
        
        plate_text = label_data.get('plate', '')
        corners_str = label_data.get('corners')
        
        if not corners_str or not plate_text: 
            continue

        try:
            points = [tuple(map(int, p.split(','))) for p in corners_str.split(' ')]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # --- CALCULATE 15% EXPANSION ---
            box_width = max_x - min_x
            box_height = max_y - min_y
            
            pad_x = int(box_width * 0.15)
            pad_y = int(box_height * 0.15)
            
            # Apply padding while keeping the box centered and within image bounds
            exp_min_x = max(0, min_x - pad_x)
            exp_min_y = max(0, min_y - pad_y)
            exp_max_x = min(w, max_x + pad_x)
            exp_max_y = min(h, max_y + pad_y)
            
            # --- PURE ARRAY SLICING ---
            cropped = img[exp_min_y:exp_max_y, exp_min_x:exp_max_x]
            
            if cropped.size == 0:
                continue

            # --- ADDED 'hr-' PREFIX HERE ---
            save_name = f"hr-{plate_text}_{img_path.stem}.jpg"
            save_path = dest_dir / save_name
            
            cv2.imwrite(str(save_path), cropped)
            
            # Append as 'training'
            flattened_entries.append(f"{plate_text};{save_path.absolute()};training")
            
        except Exception as e:
            continue

    with open(output_path, 'w') as f:
        f.write('\n'.join(flattened_entries))
        
    print(f"✅ Finished! Processed {len(flattened_entries)} expanded car plates.")
    print(f"📂 Flattened file saved to: {output_path}")

if __name__ == "__main__":
    process_rodosol()