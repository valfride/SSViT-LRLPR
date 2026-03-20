import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm

def parse_ufpr_txt(txt_path):
    data = {'chars': []}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("type:"):
                data['type'] = line.split(":")[1].strip().lower()
            elif line.startswith("make:"):
                data['make'] = line.split(":")[1].strip().lower()
            elif line.startswith("model:"):
                data['model'] = line.split(":")[1].strip().lower()
            elif line.startswith("plate:"):
                data['text'] = line.split(":")[1].strip()
            elif line.startswith("corners:"):
                pts = line.split(":")[1].strip().split()
                data['corners'] = [list(map(int, pt.split(","))) for pt in pts]
            elif line.startswith("char"):
                parts = line.split(":")[1].strip().split()
                data['chars'].append(list(map(int, parts)))
    return data

def extract_ufpr(data_root, out_dir, max_chars, max_bgs):
    data_root, out_dir = Path(data_root), Path(out_dir)
    txt_files = list(data_root.rglob("*.txt"))
    
    # --- 🚀 PASS 1: INFER CANONICAL PLATE SIZE ---
    print("📊 Pre-scanning dataset to infer canonical metrics...")
    raw_widths, raw_heights = [], []
    valid_data = []

    for txt_path in txt_files:
        data = parse_ufpr_txt(txt_path)
        plate_str = data.get('text', "")
        make, model = data.get('make', ""), data.get('model', "")
        
        # Filter unwanted plates
        if any(f in make or f in model for f in ['atego', 'neobus', 'mercedes-benz']):
            continue
        if data.get('type') != 'car' or len(plate_str) != 7:
            continue
            
        pts = np.array(data['corners'])
        w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
        h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
        
        raw_widths.append(w)
        raw_heights.append(h)
        
        # Save parsed data and path to avoid re-reading files
        data['txt_path'] = txt_path 
        valid_data.append(data)

    if not raw_widths:
        print("❌ Error: No valid car plates found!")
        return

    W, H = int(np.mean(raw_widths)), int(np.mean(raw_heights))
    dst_pts = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")

    # --- 🚀 PASS 2: INFER UNIVERSAL CHARACTER SIZE & LAYOUT ---
    slot_metrics = {i: {'x': [], 'y': []} for i in range(7)}
    all_char_w, all_char_h = [], []

    for data in valid_data:
        src_pts = np.array(data['corners'], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        for idx in range(7):
            cx, cy, cw, ch = data['chars'][idx]
            char_box = np.array([[[cx, cy]], [[cx + cw, cy]], [[cx + cw, cy + ch]], [[cx, cy + ch]]], dtype="float32")
            warped_box = cv2.perspectiveTransform(char_box, M)

            x1, y1 = np.min(warped_box[:, 0, 0]), np.min(warped_box[:, 0, 1])
            x2, y2 = np.max(warped_box[:, 0, 0]), np.max(warped_box[:, 0, 1])

            slot_metrics[idx]['x'].append(x1)
            slot_metrics[idx]['y'].append(y1)
            all_char_w.append(x2 - x1)
            all_char_h.append(y2 - y1)

    # The mathematically perfect dimension for every character in the dataset
    canon_char_w = int(np.mean(all_char_w))
    canon_char_h = int(np.mean(all_char_h))
    print(f"✅ Inferred Plate Size: {W}x{H}")
    print(f"✅ Inferred Universal Character Size: {canon_char_w}x{canon_char_h}")

    # Build the strict layout. All slots now share the exact same W and H.
    canonical_layout = []
    for i in range(7):
        canonical_layout.append({
            'x': int(np.mean(slot_metrics[i]['x'])),
            'y': int(np.mean(slot_metrics[i]['y'])),
            'w': canon_char_w,
            'h': canon_char_h
        })

    # --- 🚀 PASS 3: EXTRACT & STANDARDIZE ---
    layout_name = "Brazilian"
    font_dir = out_dir / "real_fonts" / layout_name
    bg_dir = out_dir / "inpainted_backgrounds" / layout_name
    font_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    char_counts = {}
    bg_counts = 0

    for data in tqdm(valid_data, desc="Extracting & Standardizing Assets"):
        txt_path = data['txt_path']
        text = data.get('text', "").upper()

        img_path = txt_path.with_suffix(".png")
        if not img_path.exists(): img_path = txt_path.with_suffix(".jpg")
        img = cv2.imread(str(img_path))
        if img is None: continue

        src_pts = np.array(data['corners'], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)

        mask = np.zeros((H, W), dtype=np.uint8)
        needs_bg = bg_counts < max_bgs

        for idx, char in enumerate(text):
            cx, cy, cw, ch = data['chars'][idx]
            char_box = np.array([[[cx, cy]], [[cx + cw, cy]], [[cx + cw, cy + ch]], [[cx, cy + ch]]], dtype="float32")
            warped_box = cv2.perspectiveTransform(char_box, M)

            # 1 pixel extraction padding
            x1, y1 = np.min(warped_box[:, 0, 0]), np.min(warped_box[:, 0, 1]) 
            x2, y2 = np.max(warped_box[:, 0, 0]), np.max(warped_box[:, 0, 1]) 

            wx, wy = max(0, int(x1)), max(0, int(y1))
            ww, wh = min(W - wx, int(x2 - x1)), min(H - wy, int(y2 - y1))

            if ww <= 0 or wh <= 0: continue

            # Raw image crop
            char_crop = warped[wy:wy+wh, wx:wx+ww]

            # --- 🎨 HEIGHT-LOCKED STANDARDIZATION (Anti-Smear Padding) ---
            if char_counts.get(char, 0) < max_chars:
                orig_h, orig_w = char_crop.shape[:2]
                
                # 1. Scale proportionally so the HEIGHT perfectly matches the canonical height
                new_w = max(1, int(orig_w * (canon_char_h / orig_h)))
                char_crop_scaled = cv2.resize(char_crop, (new_w, canon_char_h), interpolation=cv2.INTER_AREA)

                # 2. Fit into the canonical width slot
                if new_w < canon_char_w:
                    # Character is narrower than the slot (e.g., '1', 'I'). Pad the sides.
                    pad_x = canon_char_w - new_w
                    pad_left = pad_x // 2
                    pad_right = pad_x - pad_left
                    
                    # SMART PADDING: Sample safe background from top/bottom to avoid smearing side ink
                    top_edge = char_crop_scaled[0:2, :].reshape(-1, 3)
                    bottom_edge = char_crop_scaled[-2:, :].reshape(-1, 3)
                    safe_bg_pixels = np.concatenate([top_edge, bottom_edge])
                    bg_color = [int(c) for c in np.median(safe_bg_pixels, axis=0)]
                    
                    # Use BORDER_CONSTANT with the calculated plate color
                    standardized_crop = cv2.copyMakeBorder(
                        char_crop_scaled, 0, 0, pad_left, pad_right, 
                        cv2.BORDER_CONSTANT, value=bg_color
                    )
                else:
                    # Character is wider than the standard slot, squeeze it gently
                    standardized_crop = cv2.resize(char_crop_scaled, (canon_char_w, canon_char_h), interpolation=cv2.INTER_AREA)

                c_folder = font_dir / char
                c_folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(c_folder / f"{txt_path.stem}_{idx}.jpg"), standardized_crop)
                char_counts[char] = char_counts.get(char, 0) + 1

            # --- MASK FOR BACKGROUND INPAINTING ---
            # We use the RAW char_crop for the mask to strictly match the plate's physical coordinates
            if needs_bg:
                gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
                _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                mask[wy:wy+wh, wx:wx+ww] = cv2.bitwise_or(mask[wy:wy+wh, wx:wx+ww], cv2.dilate(ink, np.ones((3,3))))

        if needs_bg:
            inpainted = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(str(bg_dir / f"{txt_path.stem}.jpg"), inpainted)
            bg_counts += 1

    # Save the highly standardized layout JSON
    with open(out_dir / f"layout_{layout_name}.json", 'w') as f:
        json.dump(canonical_layout, f, indent=4)
    print("✅ Assets ready. All characters standardized without distortion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./datasets/synEngine/assets")
    args = parser.parse_args()
    extract_ufpr(args.data_root, args.out_dir, max_chars=500, max_bgs=300)