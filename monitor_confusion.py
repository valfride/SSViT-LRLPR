import time
import json
import os
from pathlib import Path

# Config
FILE_PATH = "./confusion_stats.json"
REFRESH_RATE = 5 # Check every 5 seconds

def load_json():
    try:
        with open(FILE_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def main():
    print(f"👀 Monitoring changes in: {FILE_PATH}...\n")
    
    last_state = {}
    
    while True:
        current_state = load_json()
        
        # If the file hasn't changed, just wait
        if current_state == last_state:
            time.sleep(REFRESH_RATE)
            continue
            
        # CLEAR SCREEN
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"🔄 UPDATED: {time.strftime('%H:%M:%S')}")
        print("="*40)
        
        # 1. NEW ENEMIES (Did not exist before)
        new_items = {k: v for k, v in current_state.items() if k not in last_state}
        if new_items:
            print("\n🔥 NEW CONFUSIONS (The network just got confused here):")
            for k, v in new_items.items():
                print(f"   + {k} vs {v}")
                
        # 2. RESOLVED ENEMIES (Existed before, now gone)
        resolved_items = {k: v for k, v in last_state.items() if k not in current_state}
        if resolved_items:
            print("\n✅ RESOLVED (The network fixed these):")
            for k, v in resolved_items.items():
                print(f"   - {k} vs {v}")

        # 3. CURRENT FULL LIST
        print("\n📋 CURRENT ACTIVE TARGETS:")
        print(json.dumps(current_state, indent=4))
        
        last_state = current_state
        time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")