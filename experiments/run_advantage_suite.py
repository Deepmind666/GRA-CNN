import sys
import subprocess
import time
from datetime import datetime

PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\gra311\python.exe"
SCRIPT_PATH = r"C:\GRA-CNN\experiments\run_real_pruning.py"

import sys
import subprocess
import time
from datetime import datetime
import concurrent.futures
import uuid
import os
import pandas as pd
import glob

PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\gra311\python.exe"
SCRIPT_PATH = r"C:\GRA-CNN\experiments\run_real_pruning.py"
MAX_PARALLEL = 7 # Push it a bit more
TEMP_DIR = r"C:\GRA-CNN\experiments\temp_results"
os.makedirs(TEMP_DIR, exist_ok=True)

def run_experiment(args):
    """Execution wrapper for thread pool with unique output"""
    cmd_base, desc = args
    
    # Generate unique output file for this worker
    unique_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(TEMP_DIR, f"res_{unique_id}.csv")
    
    # Append --output argument
    cmd_full = cmd_base + ['--output', output_file]
    
    cmd_str = ' '.join(cmd_full)
    print(f"[Worker] Starting: {desc}")
    
    start = time.time()
    result = subprocess.run(cmd_full, capture_output=True, text=True)
    duration = (time.time() - start) / 60
    
    if result.returncode == 0:
        print(f"[Worker] SUCCESS ({duration:.1f}m): {desc}")
        return output_file
    else:
        print(f"[Worker] FAILED ({duration:.1f}m): {desc}")
        print(result.stderr[-500:])
        return None

def merge_results():
    print("Merging temporary results...")
    master_file = r'C:\GRA-CNN\experiments\supplementary_results.csv'
    
    # Load master
    if os.path.exists(master_file):
        try:
            master = pd.read_csv(master_file, on_bad_lines='skip', low_memory=False)
        except:
            master = pd.DataFrame()
    else:
        master = pd.DataFrame()
        
    # Load all temps
    temp_files = glob.glob(os.path.join(TEMP_DIR, "*.csv"))
    new_data = []
    
    for f in temp_files:
        try:
            df = pd.read_csv(f)
            new_data.append(df)
        except Exception as e:
            print(f"Skipping bad temp file {f}: {e}")
            
    if new_data:
        combined_new = pd.concat(new_data, ignore_index=True)
        final = pd.concat([master, combined_new], ignore_index=True)
        final.to_csv(master_file, index=False)
        print(f"Merged {len(combined_new)} new rows. Total: {len(final)}")
        
        # Cleanup
        for f in temp_files:
            try:
                os.remove(f)
            except: pass
    else:
        print("No new data to merge.")

def main():
    print(f"=== GRA-CNN Phase 4: Safe Parallel Suite (Workers={MAX_PARALLEL}) ===")
    
    tasks = []

    # 1. Depth Scaling (ResNet-20..110)
    for arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        for method in ['l1', 'gra']:
            cmd = [PYTHON_EXE, SCRIPT_PATH, '--arch', arch, '--dataset', 'cifar10', '--method', method, '--ratio', '0.8', '--epochs', '30']
            desc = f"{arch} {method} 0.8"
            tasks.append((cmd, desc))

    # 2. Rho Sensitivity
    for rho in [0.3, 0.8]:
        cmd = [PYTHON_EXE, SCRIPT_PATH, '--arch', 'resnet110', '--dataset', 'cifar10', '--method', 'gra', '--ratio', '0.7', '--rho', str(rho), '--epochs', '30']
        desc = f"resnet110 gra 0.7 rho={rho}"
        tasks.append((cmd, desc))

    # 3. Extreme Sparsity
    for ratio in [0.85, 0.9]:
        for method in ['l1', 'gra']:
            cmd = [PYTHON_EXE, SCRIPT_PATH, '--arch', 'resnet110', '--dataset', 'cifar10', '--method', method, '--ratio', str(ratio), '--epochs', '40']
            desc = f"resnet110 {method} {ratio}"
            tasks.append((cmd, desc))

    print(f"Queue: {len(tasks)} experiments")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_experiment, t): t for t in tasks}
        concurrent.futures.wait(futures)
        
    merge_results()
    print(f"\n=== Suite Completed ===")



if __name__ == "__main__":
    main()
