import argparse
import os
import lmdb
import cv2
import numpy as np
import pickle
from tqdm import tqdm

def make_lmdb_dataset(input_txt, output_dir, map_size=1099511627776): # Default map_size is 1TB (it's virtual, not actual disk space)
    """
    Creates an LMDB database from a flat text file.
    The text file format must be: GT_LABEL;IMAGE_PATH;SPLIT
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 1. Open the LMDB Environment
    # writemap=True and map_async=True significantly speed up the writing process
    env = lmdb.open(output_dir, map_size=map_size, writemap=True, map_async=True)

    # 2. Parse the Input Text File
    print(f"Reading dataset list from {input_txt}...")
    with open(input_txt, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(lines)} total entries. Starting conversion...")

    # We will also create a new metadata list that we can easily load in our PyTorch dataset
    metadata_list = []
    failed_images = 0

    # 3. Write Images to Database in Transactions
    # We use a transaction (txn) and commit periodically so we don't hold the whole DB in RAM
    txn = env.begin(write=True)
    write_frequency = 5000 

    for idx, line in enumerate(tqdm(lines)):
        parts = line.split(';')
        if len(parts) != 3:
            print(f"⚠️ Warning: Skipping malformed line at index {idx}: {line}")
            failed_images += 1
            continue
            
        gt_label, img_path, split = parts

        # Read the image using OpenCV
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image not found: {img_path}")
            failed_images += 1
            continue

        # We read as raw bytes to store it optimally. We do NOT decode it to numpy yet!
        # This keeps the LMDB file much smaller (storing the compressed JPEG/PNG bytes)
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        # The key must be a unique byte string. We use a formatted index.
        key = f"image_{idx:08d}".encode('ascii')

        # Store the raw bytes in the database
        txn.put(key, image_bytes)

        # Append metadata (We store the LMDB key so our PyTorch dataset knows what to ask for)
        metadata_list.append({
            'lmdb_key': key,
            'gt': gt_label,
            'split': split,
            'original_path': img_path # Kept for debugging/logging purposes
        })

        # Commit the transaction periodically
        if (idx + 1) % write_frequency == 0:
            txn.commit()
            txn = env.begin(write=True)

    # Final commit for any remaining entries
    txn.commit()

    # 4. Save the Metadata
    # We save the metadata list as a pickle file inside the LMDB directory
    # so the dataloader can easily load the keys and ground truths.
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_list, f)

    # Close the environment
    env.close()

    print("\n✅ LMDB Database Creation Complete!")
    print(f"Successfully wrote {len(metadata_list)} images to {output_dir}")
    print(f"Saved metadata to {metadata_path}")
    if failed_images > 0:
        print(f"⚠️ Failed to write {failed_images} images (check warnings above).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert an image dataset to an LMDB database.")
    parser.add_argument('--input', type=str, required=True, help="Path to the flat txt file (e.g., flattened_competition_dataset.txt)")
    parser.add_argument('--output', type=str, required=True, help="Path to the directory where the LMDB will be created")
    args = parser.parse_args()

    make_lmdb_dataset(args.input, args.output)