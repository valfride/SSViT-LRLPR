import argparse
import os
import glob

def convert_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found.")
        return

    output_lines = []
    missing_tracks = 0
    total_images = 0

    print(f"Reading track definitions from: {input_file}")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split(';')
        if len(parts) != 3:
            print(f"⚠️ Warning: Skipping malformed line: {line}")
            continue

        plate_text = parts[0]
        track_folder = parts[1]
        split_type = parts[2]

        if not os.path.isdir(track_folder):
            missing_tracks += 1
            continue

        # Grab absolutely every image in the folder (both lr-*.png/jpg and hr-*.png/jpg)
        search_pattern = os.path.join(track_folder, "*.*")
        all_files = glob.glob(search_pattern)

        for file_path in all_files:
            # Simple check to ensure we only grab images, not hidden files or text logs
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Format: PLATE;IMAGE_PATH;SPLIT
                new_line = f"{plate_text};{file_path};{split_type}\n"
                output_lines.append(new_line)
                total_images += 1

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"\n✅ Conversion Complete!")
    print(f"   - Processed {len(lines)} track folders.")
    print(f"   - Generated {total_images} individual image entries.")
    if missing_tracks > 0:
        print(f"   - ⚠️ Missed {missing_tracks} track folders (path not found).")
    print(f"   - Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten a track-based dataset list into single independent image paths.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file (e.g., final_competition_split.txt)")
    parser.add_argument("--output", type=str, required=True, help="Path to the output text file (e.g., flattened_split.txt)")

    args = parser.parse_args()
    convert_dataset(args.input, args.output)