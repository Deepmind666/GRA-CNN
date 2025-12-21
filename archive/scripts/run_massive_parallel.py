import subprocess
import time
import sys
import os

# Define python executable (use the one from the correct environment)
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"

# Define experiments for MASSIVE PARALLEL EXECUTION
experiments = []

# 1. CIFAR-100 Ablation (Different Pruning Ratios) - GRA
ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
for r in ratios:
    experiments.append((r, 0.5, "gra"))

# 2. CIFAR-100 GRA Rho Ablation (Fix ratio at 0.5)
# We already have 0.5 in the list above (which implies rho=0.5).
# Add other rhos.
rhos = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
for rho in rhos:
    experiments.append((0.5, rho, "gra"))

# 3. CIFAR-100 L1-Norm Comparison
l1_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
for r in l1_ratios:
    experiments.append((r, 0.5, "l1")) # rho is ignored for L1

# Remove duplicates (if any)
experiments = sorted(list(set(experiments)))

MAX_WORKERS = 12 # AGGRESSIVE PARALLELISM for RTX 5090

print(f"Starting {len(experiments)} experiments on RTX 5090 with MAX_WORKERS={MAX_WORKERS}...")

def run_batch(batch):
    procs = []
    for ratio, rho, method in batch:
        save_dir = f"./experiments/cifar100_r20_{method}_{ratio}_{rho}"
        csv_filename = f'results_cifar100_{method}_{ratio}_{rho}.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        
        if os.path.exists(csv_path):
            print(f"Skipping finished experiment: {method} ratio={ratio}, rho={rho}")
            continue

        cmd = [
            PYTHON_EXE, "run_cifar100_resnet20.py",
            "--ratio", str(ratio),
            "--rho", str(rho),
            "--method", method,
            "--save_dir", save_dir,
            "--workers", "0" # Set workers to 0 to avoid Windows shared event error
        ]
        
        print(f"Launching: CIFAR-100 | method={method} | ratio={ratio} | rho={rho}")
        p = subprocess.Popen(cmd)
        procs.append(p)
        time.sleep(2) # Stagger start to avoid IO spike
    
    # Wait for batch
    for p in procs:
        p.wait()

# Split into batches
for i in range(0, len(experiments), MAX_WORKERS):
    batch = experiments[i:i+MAX_WORKERS]
    print(f"Running batch {i//MAX_WORKERS + 1}...")
    run_batch(batch)

print("All experiments completed.")
