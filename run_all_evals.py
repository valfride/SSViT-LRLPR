import os
import subprocess
import re

def run_evaluations():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    # Add your PROPOSED_MODEL here when ready!
    models = [
        "CPPD_BASELINE",
        "IGTR_BASELINE",
        "LISTER_BASELINE",
        "MDIFF_BASELINE",
        "OTE_BASELINE",
        "SVTRV2_BASELINE"
    ]

    frame_counts = [1, 3, 5]
    split_path = "./CompetitionDataset_LMDB_TEST_3k" # Adjust if your path is different
    
    regex_full = re.compile(r"Full Sequence \(7/7\):\s*([\d\.]+)%")
    regex_6plus = re.compile(r"Partial Match \(≥6/7\):\s*([\d\.]+)%")
    regex_5plus = re.compile(r"Partial Match \(≥5/7\):\s*([\d\.]+)%")
    
    # ---> NEW: Catch the Confidence Gap!
    regex_gap = re.compile(r"Confidence Gap:\s+([-\d\.]+)")

    results = []

    print(f"{'='*60}")
    print("🚀 STARTING SILENT TEMPORAL ABLATION")
    print(f"{'='*60}\n")

    for model in models:
        for frames in frame_counts:
            print(f"⏳ Evaluating {model} [Frames: {frames}]... ", end="", flush=True)
            
            config_path = f"./experiments/baselines/{model}/{model}_14-04-2026/config_snapshot.yaml"
            ckpt_path = f"./experiments/baselines/{model}/{model}_14-04-2026/"
            
            cmd = [
                "python3", "test_competition.py",
                "--config", config_path,
                "--checkpoints", ckpt_path,
                "--split", split_path,
                "--mode", "val",
                "--in_images", str(frames)
            ]
            print(cmd)
            try:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                output = result.stdout
                
                full_match = regex_full.search(output)
                plus6_match = regex_6plus.search(output)
                plus5_match = regex_5plus.search(output)
                gap_match = regex_gap.search(output) # <--- Extract the gap
                
                results.append({
                    "Model": model.replace("_BASELINE", ""),
                    "Frames": frames,
                    "Full": full_match.group(1) if full_match else "N/A",
                    ">=6": plus6_match.group(1) if plus6_match else "N/A",
                    ">=5": plus5_match.group(1) if plus5_match else "N/A",
                    "ConfGap": gap_match.group(1) if gap_match else "N/A" # <--- Store it
                })
                
                print("✅ Done!") 
                
            except subprocess.CalledProcessError as e:
                # If it crashes, drop down to the next line and report it
                print("❌ ERROR!")
                print(f"--- CRASH LOG ---\n{e.stderr}\n-----------------") # <--- ADD THIS
                results.append({
                    "Model": model.replace("_BASELINE", ""),
                    "Frames": frames,
                    "Full": "ERROR", ">=6": "ERROR", ">=5": "ERROR", "ConfGap": "ERROR"
                })

    # ==============================================================================
    # THE ORIGINAL GROUPED TABLE (Now with the Confidence Gap column)
    # ==============================================================================
    print("\n" + "="*95)
    print("🏆 TEMPORAL ABLATION RESULTS TABLE")
    print("="*95)
    print(f"| {'Model Architecture':<20} | {'Frames':<6} | {'Full Seq (7/7)':<15} | {'≥ 6 Chars (N-1)':<15} | {'≥ 5 Chars (N-2)':<15} | {'Conf Gap':<10} |")
    print(f"|{'-'*22}|{'-'*8}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*12}|")
    
    current_model = ""
    for res in results:
        display_model = res['Model'] if res['Model'] != current_model else ""
        current_model = res['Model']
        
        f = f"{res['Full']}%" if res['Full'] not in ["ERROR", "N/A"] else res['Full']
        s6 = f"{res['>=6']}%" if res['>=6'] not in ["ERROR", "N/A"] else res['>=6']
        s5 = f"{res['>=5']}%" if res['>=5'] not in ["ERROR", "N/A"] else res['>=5']
        
        # Format the gap to 4 decimal places if it's a valid number
        try:
            gap_val = float(res['ConfGap'])
            gap_str = f"{gap_val:.4f}"
        except ValueError:
            gap_str = res['ConfGap']

        print(f"| {display_model:<20} | {res['Frames']:<6} | {f:<15} | {s6:<15} | {s5:<15} | {gap_str:<10} |")
        
        # Add a visual separator between different models for readability
        if res['Frames'] == 5:
            print(f"|{'-'*22}|{'-'*8}|{'-'*17}|{'-'*17}|{'-'*17}|{'-'*12}|")

if __name__ == "__main__":
    run_evaluations()