import os
import subprocess
import time
from pathlib import Path

def run_ablations():
    # 1. Define the environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 2. Define the ablation jobs
    # The key is the config file name, the value is the unique tag
    ablations = [
        {"cfg": "nohcg_config.yaml",  "tag": "nohcg_27-04-2026"},
        {"cfg": "nosfb_config.yaml",  "tag": "nosfb_27-04-2026"},
        {"cfg": "norope_config.yaml", "tag": "norope_27-04-2026"},
    ]

    print(f"{'='*60}")
    print("🚀 STARTING ARCHITECTURAL ABLATION PIPELINE")
    print("⚠️  Warning: Each run may take a long time. Use 'tmux' or 'screen'.")
    print(f"{'='*60}\n")

    for job in ablations:
        config_name = job["cfg"]
        tag = job["tag"]
        log_file = f"train_log_{tag}.txt"
        
        print(f"⏳ Training: {config_name} ... ", end="", flush=True)
        
        # Build the command exactly as your manual setup
        cmd = [
            "python3", "train_gan.py",
            "--config", f"./ablation_configs/{config_name}",
            "--save", "./experiments/ablation",
            "--tag", tag
        ]
        
        start_time = time.time()
        
        try:
            # Open a unique log file for this specific ablation
            with open(log_file, "w") as f:
                # Use subprocess.run to wait for the training to finish
                subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            
            elapsed = (time.time() - start_time) / 3600 # Convert to hours
            print(f"✅ Done! ({elapsed:.2f} hrs) | Logs: {log_file}")
            
        except subprocess.CalledProcessError:
            print(f"❌ CRASHED! Check logs: {log_file}")
            # Optional: continue to next ablation if one fails
            continue

    print(f"\n{'='*60}")
    print("🏆 ALL ABLATION RUNS COMPLETED")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_ablations()