import os
import json
from pathlib import Path

def generate_test_list(dataset_root, output_file):
    # Define the root path to the test folder
    test_dir = Path(dataset_root) / "test"
    
    # Check if directory exists
    if not test_dir.exists():
        print(f"Error: Directory {test_dir} not found.")
        return

    # List to store the formatted lines
    lines = []
    
    # Iterate through all items in the test directory
    for track_folder in sorted(test_dir.iterdir()):
        if track_folder.is_dir():
            relative_path = track_folder.as_posix()
            
            # =========================================================
            # ---> THE UPGRADE: Dynamic Ground Truth Extraction
            # =========================================================
            gt_text = "UNKNOWN" # Fallback just in case
            
            # Search for any .json or .txt annotation file in the folder
            annotation_files = list(track_folder.glob("*.json")) + list(track_folder.glob("*.txt"))
            
            for anno_file in annotation_files:
                try:
                    with open(anno_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # If we found the plate_text, grab it and stop searching this folder
                        if "plate_text" in data:
                            gt_text = data["plate_text"].strip()
                            break 
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If the file isn't valid JSON, skip it and keep looking
                    continue
            
            # The phase is strictly 'testing' or 'validation' 
            # (Note: keeping it "testing" is fine, but you will use --mode val to evaluate it)
            phase = "testing"
            
            # Format: GT;PATH;PHASE
            line = f"{gt_text};{relative_path};{phase}"
            lines.append(line)

    # Write the collected lines to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
            
    print(f"Successfully generated {output_file} with {len(lines)} entries.")
    print("Sample entries:")
    for l in lines[:5]:
        print(l)

if __name__ == "__main__":
    # Configuration
    DATASET_ROOT = "./CompetitionDataset" 
    OUTPUT_FILE = "test_split.txt"
    
    generate_test_list(DATASET_ROOT, OUTPUT_FILE)