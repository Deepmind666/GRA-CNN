import subprocess
import time
import os
import sys

PYTHON_EXE = sys.executable
MAX_WORKERS = 6 # Adjust based on GPU memory

experiments = []

# 1. CIFAR-10 + ResNet-20
for method in ['l1', 'fpgm', 'hrank', 'gra']:
    for ratio in [0.3, 0.5, 0.7]:
        experiments.append({
            'dataset': 'cifar10',
            'model': 'resnet20',
            'method': method,
            'ratio': ratio,
            'epochs': 160,
            'finetune_epochs': 100
        })

# 2. CIFAR-100 + ResNet-20
for method in ['l1', 'fpgm', 'hrank', 'gra']:
    experiments.append({
        'dataset': 'cifar100',
        'model': 'resnet20',
        'method': method,
        'ratio': 0.5,
        'epochs': 160,
        'finetune_epochs': 100
    })

# 3. Tiny-ImageNet + ResNet-18
for method in ['l1', 'fpgm', 'hrank', 'gra']:
    for ratio in [0.4, 0.5]:
        experiments.append({
            'dataset': 'tinyimagenet',
            'model': 'resnet18',
            'method': method,
            'ratio': ratio,
            'epochs': 60, # Shorter for Tiny
            'finetune_epochs': 40
        })

def run_batch(batch):
    procs = []
    for exp in batch:
        # Check if done
        save_dir = f"experiments/{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['ratio']}"
        if os.path.exists(os.path.join(save_dir, "results_comprehensive.csv")):
             print(f"Skipping finished: {exp}")
             continue
             
        cmd = [
            PYTHON_EXE, "run_experiments.py",
            "--dataset", str(exp['dataset']),
            "--model", str(exp['model']),
            "--method", str(exp['method']),
            "--prune_ratio", str(exp['ratio']),
            "--epochs", str(exp['epochs']),
            "--finetune_epochs", str(exp['finetune_epochs']),
            "--workers", "0" # Safer for Windows
        ]
        
        print(f"Launching: {exp['dataset']} | {exp['method']} | {exp['ratio']}")
        p = subprocess.Popen(cmd)
        procs.append(p)
        time.sleep(2) # Stagger
        
    for p in procs:
        p.wait()

def main():
    # Chunk experiments
    for i in range(0, len(experiments), MAX_WORKERS):
        batch = experiments[i:i+MAX_WORKERS]
        print(f"--- Running Batch {i//MAX_WORKERS + 1} ---")
        run_batch(batch)

if __name__ == "__main__":
    main()
