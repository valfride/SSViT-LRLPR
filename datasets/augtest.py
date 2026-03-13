import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from wrappers import Sequential_lr_sr
import os
from pathlib import Path

def find_image_tracks(root_dir, num_tracks=4):
    """
    Scans the dataset to find valid track folders, and extracts all pairs inside them.
    Returns a list of tracks, where each track is a list of (lr_path, hr_path) pairs.
    """
    root = Path(root_dir)
    if not root.exists():
        return []
    
    tracks = []
    # Find all directories that contain hr-* images
    for track_folder in root.rglob("*"):
        if not track_folder.is_dir(): continue
        
        hr_files = sorted(list(track_folder.glob("hr-*.*")))
        if len(hr_files) == 0: continue
        
        # Filter for image extensions
        hr_files = [f for f in hr_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if len(hr_files) == 0: continue
        
        track_pairs = []
        for hr_path in hr_files:
            lr_name = hr_path.name.replace('hr-', 'lr-', 1)
            lr_path = track_folder / lr_name
            if lr_path.exists():
                track_pairs.append((str(lr_path), str(hr_path)))
                
        if len(track_pairs) > 0:
            tracks.append(track_pairs)
            
        if len(tracks) >= num_tracks:
            break
            
    return tracks

def check_real_batches():
    print("🧪 Initializing Degradation Engine Inspection...")

    possible_roots = [
        "./CompetitionDataset",
        "../CompetitionDataset",
        "/content/CompetitionDataset",
        "./synthetic_dataset" 
    ]
    
    tracks = []
    for r in possible_roots:
        found = find_image_tracks(r, num_tracks=5) # Fetch 5 different plates
        if len(found) > 0:
            print(f"✅ Found {len(found)} track folders in: {r}")
            tracks = found
            break
            
    if not tracks:
        print("❌ ERROR: Could not find any dataset folders. Please check the 'possible_roots' list.")
        return

    # Create dummy dataset list mapping
    dataset_list = []
    for track in tracks:
        for lr_path, hr_path in track:
            dataset_list.append({'imgs': lr_path, 'gt': 'TEST'})
            dataset_list.append({'imgs': hr_path, 'gt': 'TEST'})

    # Configure Loader
    dataset = Sequential_lr_sr(
        imgW=192, imgH=64,        
        aug=True,                 
        image_aspect_ratio=3.0, 
        background=None,
        test=False,               
        dataset=dataset_list, 
        in_images=1
    )

    print(f"✅ Loader Initialized. Fetching {len(tracks)} tracks...")

    all_tensors = []
    idx = 0

    for t_idx, track in enumerate(tracks):
        track_real_lr = []
        track_synth_lr = []
        track_clean_hr = []
        
        print(f"  Processing Track {t_idx+1}...")
        for i in range(len(track)):
            try:
                lr_sample = dataset[idx]
                hr_sample = dataset[idx + 1]
                idx += 2
                
                track_real_lr.append(lr_sample['lr'][0])
                track_synth_lr.append(hr_sample['lr'][0])
                track_clean_hr.append(hr_sample['hr'][0])
            except Exception as e:
                print(f"    Frame {i+1} Failed: {e}")
                
        # Force exactly 5 frames per track for a perfect grid
        while len(track_real_lr) < 5:
            # Pad with empty tensors if track has less than 5 frames
            track_real_lr.append(torch.zeros_like(track_real_lr[0]))
            track_synth_lr.append(torch.zeros_like(track_synth_lr[0]))
            track_clean_hr.append(torch.zeros_like(track_clean_hr[0]))
            
        track_real_lr = track_real_lr[:5]
        track_synth_lr = track_synth_lr[:5]
        track_clean_hr = track_clean_hr[:5]
        
        # Append rows sequentially: Row 1 (Real), Row 2 (Synth), Row 3 (Clean)
        # Because we feed make_grid with nrow=5, these will perfectly form 3 rows!
        all_tensors.extend(track_real_lr)
        all_tensors.extend(track_synth_lr)
        all_tensors.extend(track_clean_hr)

    # VISUALIZE & SAVE
    if len(all_tensors) > 0:
        # Stack & Denormalize
        comparison_batch = (torch.stack(all_tensors) * 0.5) + 0.5

        # Figure height scales with number of tracks (3 rows per track)
        plt.figure(figsize=(14, 3 * len(tracks))) 
        
        # nrow=5 forces exactly 5 columns (1 column per frame)
        padding = 4
        grid = vutils.make_grid(comparison_batch, nrow=5, padding=padding, normalize=False) 
        
        img_np = grid.permute(1, 2, 0).numpy()
        
        plt.imshow(img_np)
        
        # Draw horizontal separating lines to distinguish tracks visually
        imgH = 64
        for i in range(1, len(tracks)):
            # Calculate exact Y pixel coordinate to draw a line between track blocks
            y_line = i * (3 * imgH + 4 * padding) + (padding // 2)
            plt.axhline(y=y_line, color='red', linewidth=2, alpha=0.7)
            
        plt.title(f"Degradation Pipeline Tuning ({len(tracks)} Tracks)\nEach Block: Row 1: Real LR | Row 2: Synth LR | Row 3: Clean HR", fontsize=14, pad=15)
        plt.axis("off")
        plt.tight_layout()
        
        save_path = "augmentation_tuning.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n📸 Grid successfully saved to: {os.path.abspath(save_path)}")
        plt.close()

if __name__ == "__main__":
    check_real_batches()