import random
import os

# CONFIG
FAILURE_FILE = "validation_failures.txt"
CURRENT_SPLIT = "final_competition_split.txt"  # Your current full split file
NEW_SPLIT = "split_hard_mining.txt"
MOVE_RATIO = 0.5  # Move 50% of failures

def extract_track_id(path_str):
    # Extracts 'track_XXXXX' from the path
    parts = path_str.split('/')
    for p in parts:
        if "track_" in p:
            return p
    return None

def main():
    print(f"🚀 Parsing failures from {FAILURE_FILE}...")
    
    # 1. Parse Failures
    failures = []
    with open(FAILURE_FILE, 'r') as f:
        for line in f:
            if "|" not in line: continue
            
            # Format: /path/to/img | Pred 'XX' vs GT 'YY' | Conf 0.XX
            parts = line.strip().split('|')
            path_part = parts[0].strip()
            conf_part = parts[2].strip()
            
            track_id = extract_track_id(path_part)
            confidence = float(conf_part.split(' ')[1])
            
            if track_id:
                failures.append({'id': track_id, 'conf': confidence})

    # Sort by confidence (Highest first) -> We want to train on the "confident mistakes"
    failures.sort(key=lambda x: x['conf'], reverse=True)
    
    # Select the top 50% to move
    num_to_move = int(len(failures) * MOVE_RATIO)
    ids_to_move = set([f['id'] for f in failures[:num_to_move]])
    
    print(f"🔍 Found {len(failures)} failures.")
    print(f"🎯 Moving {len(ids_to_move)} high-confidence errors to Training Set.")

    # 2. Process Split File
    moved_count = 0
    new_lines = []
    
    with open(CURRENT_SPLIT, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Format: PLATE;PATH;SPLIT
            # Example: AUZ7696;./CompetitionDataset/.../track_10019;validation
            parts = line.split(';')
            plate = parts[0]
            path = parts[1]
            split_type = parts[2]
            
            track_id = extract_track_id(path)
            
            # Check if this is one of our targets
            if split_type == 'validation' and track_id in ids_to_move:
                new_lines.append(f"{plate};{path};training") # <--- FLIP TO TRAINING
                moved_count += 1
            else:
                new_lines.append(line) # Keep as is

    # 3. Save
    with open(NEW_SPLIT, 'w') as f:
        f.write("\n".join(new_lines))
        
    print(f"✅ Success! Moved {moved_count} tracks.")
    print(f"📂 New split file saved as: {NEW_SPLIT}")
    print("⚠️  Remember to update your config to point to this new file!")

if __name__ == "__main__":
    main()