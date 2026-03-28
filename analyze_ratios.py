import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def analyze_aspect_ratios(dataset_dir):
    aspect_ratios = []
    
    # 1. Grab all images in your dataset folder safely
    print(f"Scanning '{dataset_dir}' for images...")
    
    # Convert string to Path object
    base_path = Path(dataset_dir)
    if not base_path.exists():
        print(f"❌ Error: The directory {dataset_dir} does not exist!")
        return

    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    # Recursively find all files and filter by our allowed extensions
    image_paths = [
        p for p in base_path.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    
    if not image_paths:
        print(f"⚠️ No .jpg, .jpeg, or .png images found in {dataset_dir}!")
        return
        
    print(f"✅ Found {len(image_paths)} images. Calculating aspect ratios...")
    
    # 2. Loop through and calculate R = Width / Height
    for img_path in tqdm(image_paths, desc="Processing Images"):
        # cv2.imread works with strings, so convert Path object back to string
        img = cv2.imread(str(img_path))
        if img is None:
            # OpenCV fails silently if an image is corrupted, so we just skip it
            continue
            
        height, width = img.shape[:2]
        
        # Prevent division by zero just in case of an empty/corrupted 0x0 file
        if height > 0: 
            r = width / height
            aspect_ratios.append(r)
            
    if not aspect_ratios:
        print("⚠️ Could not read any valid image data (files might be corrupted).")
        return

    # 3. Plot the Histogram
    plt.figure(figsize=(10, 6))
    
    # We use 50 bins to get a nice, detailed curve
    counts, bins, patches = plt.hist(aspect_ratios, bins=50, color='skyblue', edgecolor='black') 
    
    plt.title(f'License Plate Aspect Ratio Distribution\n{dataset_dir}')
    plt.xlabel('Aspect Ratio (Width / Height)')
    plt.ylabel('Number of Images')
    
    # Add some helpful grid lines and a Mean line
    mean_ratio = sum(aspect_ratios) / len(aspect_ratios)
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(mean_ratio, color='red', linestyle='dashed', linewidth=2, label=f'Mean Ratio: {mean_ratio:.2f}')
    plt.legend()
    
    # 4. Print the exact statistics to the terminal
    print("\n" + "="*40)
    print("📊 ASPECT RATIO STATISTICS")
    print("="*40)
    print(f"Total Valid Images: {len(aspect_ratios)}")
    print(f"Minimum Ratio:      {min(aspect_ratios):.2f}")
    print(f"Maximum Ratio:      {max(aspect_ratios):.2f}")
    print(f"Average Ratio:      {mean_ratio:.2f}")
    print("="*40 + "\n")
    
    # 5. Display the graph
    plt.show()

if __name__ == "__main__":
    # Your base dataset path
    base_dataset_path = "/home/vwnascimento/doc2025/CompetitionDataset/train/"
    
    # ---------------------------------------------------------
    # OPTION 1: Run the analysis on the entire dataset at once
    # ---------------------------------------------------------
    print("▶️ STARTING FULL DATASET ANALYSIS")
    analyze_aspect_ratios(base_dataset_path)
    
    # ---------------------------------------------------------
    # OPTION 2: Run them separately to compare the layouts
    # (Uncomment the lines below if you want to analyze them individually)
    # ---------------------------------------------------------
    
    # print("▶️ ANALYZING OLD BRAZILIAN PLATES ONLY")
    # br_path = os.path.join(base_dataset_path, "Scenario-A", "Brazilian")
    # analyze_aspect_ratios(br_path)
    # 
    # print("▶️ ANALYZING MERCOSUR PLATES ONLY")
    # mercosur_path = os.path.join(base_dataset_path, "Scenario-A", "Mercosur")
    # analyze_aspect_ratios(mercosur_path)