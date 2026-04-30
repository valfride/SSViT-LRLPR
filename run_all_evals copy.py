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
    split_path = "./CompetitionDataset_LMDB_TEST_3k/"
    
    regex_full = re.compile(r"Full Sequence \(7/7\):\s*([\d\.]+)%")
    regex_6plus = re.compile(r"Partial Match \(≥6/7\):\s*([\d\.]+)%")
    regex_5plus = re.compile(r"Partial Match \(≥5/7\):\s*([\d\.]+)%")

    results = []

    print(f"{'='*60}")
    print("🚀 STARTING SILENT TEMPORAL ABLATION")
    print(f"{'='*60}\n")

    for model in models:
        for frames in frame_counts:
            # ---> THE TRICK: end="" prevents a new line, flush=True forces it to render immediately
            print(f"⏳ Evaluating {model} [Frames: {frames}]... ", end="", flush=True)
            
            config_path = f"./experiments/baselines/{model}/{model}_14-04-2026/config_snapshot.yaml"
            ckpt_path = f"./experiments/baselines/{model}/{model}_14-04-2026/"
            
            cmd = [
                "python", 
                "test_competition.py",
                "--config", 
                config_path,
                "--checkpoints", 
                ckpt_path,
                "--split", 
                split_path,
                "--mode", 
                "val",
                # "--swa", 
                "--in_images", str(frames)
            ]
            
            try:
                # capture_output=True swallows all tqdm bars and pyTorch warnings silently
                process = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
                output = process.stdout
                
                full_match = regex_full.search(output)
                plus6_match = regex_6plus.search(output)
                plus5_match = regex_5plus.search(output)
                
                results.append({
                    "Model": model.replace("_BASELINE", ""),
                    "Frames": frames,
                    "Full": full_match.group(1) if full_match else "N/A",
                    ">=6": plus6_match.group(1) if plus6_match else "N/A",
                    ">=5": plus5_match.group(1) if plus5_match else "N/A"
                })
                
                # Prints on the exact same line when finished!
                print("✅ Done!") 
                
            except subprocess.CalledProcessError as e:
                # If it crashes, drop down to the next line and report it
                print("❌ ERROR! Check logs.")
                results.append({
                    "Model": model.replace("_BASELINE", ""),
                    "Frames": frames,
                    "Full": "ERROR", ">=6": "ERROR", ">=5": "ERROR"
                })

    # Print the Grouped Markdown Table
    print("\n" + "="*80)
    print("🏆 TEMPORAL ABLATION RESULTS TABLE")
    print("="*80)
    print(f"| {'Model Architecture':<20} | {'Frames':<6} | {'Full Seq (7/7)':<15} | {'≥ 6 Chars (N-1)':<15} | {'≥ 5 Chars (N-2)':<15} |")
    print(f"|{'-'*22}|{'-'*8}|{'-'*17}|{'-'*17}|{'-'*17}|")
    
    current_model = ""
    for res in results:
        display_model = res['Model'] if res['Model'] != current_model else ""
        current_model = res['Model']
        
        f = f"{res['Full']}%" if res['Full'] not in ["ERROR", "N/A"] else res['Full']
        s6 = f"{res['>=6']}%" if res['>=6'] not in ["ERROR", "N/A"] else res['>=6']
        s5 = f"{res['>=5']}%" if res['>=5'] not in ["ERROR", "N/A"] else res['>=5']
        
        print(f"| {display_model:<20} | {res['Frames']:<6} | {f:<15} | {s6:<15} | {s5:<15} |")

if __name__ == "__main__":
    run_evaluations()