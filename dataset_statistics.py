import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CSV_FILE = "final_competition_split.txt"   # Put your csv filename here
DATA_ROOT = "."          # Change if your csv paths need a prefix (e.g. "D:/Data")

def get_image_dims(path):
    """
    Fast way to get dimensions without loading pixel data.
    """
    img = cv2.imread(str(path))
    if img is not None:
        return img.shape[:2] # Height, Width
    return None

def analyze_dataset(csv_path, root_dir):
    print(f"📂 Loading {csv_path}...")
    
    # 1. Parse CSV
    # Format: Plate;Path;Split
    data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    print(f"🔍 Found {len(lines)} entries. Scanning Scenario-B...")
    
    hr_stats = []
    lr_stats = []
    
    for line in tqdm(lines):
        line = line.strip()
        if not line: continue
        
        parts = line.split(';')
        if len(parts) < 2: continue
        
        plate, rel_path, split = parts[0], parts[1], parts[2]
        
        # FILTER: Only Scenario-B
        if "Scenario-B" not in rel_path:
            continue
            
        # Construct full path
        track_path = Path(root_dir) / rel_path
        
        if not track_path.exists():
            continue
            
        # Scan for all images in the track
        # We look for ALL hr-*.jpg and lr-*.jpg patterns
        hr_files = sorted(list(track_path.glob("hr-*.jpg")))
        lr_files = sorted(list(track_path.glob("lr-*.jpg")))
        
        # Collect HR Stats
        for f in hr_files:
            dims = get_image_dims(f)
            if dims:
                h, w = dims
                hr_stats.append({'h': h, 'w': w, 'split': split, 'type': 'HR'})

        # Collect LR Stats
        for f in lr_files:
            dims = get_image_dims(f)
            if dims:
                h, w = dims
                lr_stats.append({'h': h, 'w': w, 'split': split, 'type': 'LR'})

    # Convert to DataFrames
    df_hr = pd.DataFrame(hr_stats)
    df_lr = pd.DataFrame(lr_stats)
    
    return df_hr, df_lr

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_distributions(df_hr, df_lr):
    plt.figure(figsize=(18, 6))
    
    # --- 1. HR Scatter (Width vs Height) ---
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df_hr, x='w', y='h', hue='split', alpha=0.3)
    plt.title(f"HR Resolution Distribution (N={len(df_hr)})")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.grid(True, alpha=0.3)
    
    # --- 2. LR Scatter (Width vs Height) ---
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_lr, x='w', y='h', hue='split', alpha=0.3)
    plt.title(f"LR Resolution Distribution (N={len(df_lr)})")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.grid(True, alpha=0.3)
    
    # --- 3. Aspect Ratio Histogram ---
    plt.subplot(1, 3, 3)
    df_hr['aspect_ratio'] = df_hr['w'] / df_hr['h']
    df_lr['aspect_ratio'] = df_lr['w'] / df_lr['h']
    
    sns.kdeplot(data=df_hr, x='aspect_ratio', label='HR', fill=True, alpha=0.3)
    sns.kdeplot(data=df_lr, x='aspect_ratio', label='LR', fill=True, alpha=0.3)
    plt.title("Aspect Ratio (W/H) Density")
    plt.xlabel("Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("dataset_distribution_scenario_b.png")
    print("\n✅ Plot saved to dataset_distribution_scenario_b.png")

if __name__ == "__main__":
    df_hr, df_lr = analyze_dataset(CSV_FILE, DATA_ROOT)
    
    if not df_hr.empty:
        print("\n📊 HR STATISTICS (Scenario-B)")
        print(df_hr[['h', 'w']].describe())
        print(f"\nMost Common HR Size: {df_hr.groupby(['w', 'h']).size().idxmax()}")
        
        print("\n📊 LR STATISTICS (Scenario-B)")
        print(df_lr[['h', 'w']].describe())
        print(f"\nMost Common LR Size: {df_lr.groupby(['w', 'h']).size().idxmax()}")
        
        plot_distributions(df_hr, df_lr)
    else:
        print("⚠️ No Scenario-B data found. Check your paths.")