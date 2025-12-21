import subprocess
import sys
import time
import os

# Define experiments
experiments = []

# 1. CIFAR-100 ResNet-56 (GRA vs L1)
for r in [0.3, 0.5, 0.7]:
    experiments.append({
        'dataset': 'cifar100', 'model': 'resnet56', 'method': 'l1', 'ratio': r
    })
    experiments.append({
        'dataset': 'cifar100', 'model': 'resnet56', 'method': 'gra', 'ratio': r
    })

# 2. CIFAR-10 ResNet-110 (Scalability)
experiments.append({
    'dataset': 'cifar10', 'model': 'resnet110', 'method': 'l1', 'ratio': 0.5
})
experiments.append({
    'dataset': 'cifar10', 'model': 'resnet110', 'method': 'gra', 'ratio': 0.5
})

# 3. Smoothing Curves (ResNet-20/56 CIFAR-10)
for r in [0.4, 0.6, 0.8]:
    experiments.append({
        'dataset': 'cifar10', 'model': 'resnet20', 'method': 'gra', 'ratio': r
    })
    experiments.append({
        'dataset': 'cifar10', 'model': 'resnet56', 'method': 'gra', 'ratio': r
    })

MAX_CONCURRENT = 4
procs = []

def run_exp(exp):
    cmd = [
        sys.executable, "run_experiments.py",
        "--dataset", exp['dataset'],
        "--model", exp['model'],
        "--method", exp['method'],
        "--prune_ratio", str(exp['ratio']),
        "--epochs", "160", # Baseline training if needed (usually cached or skipped)
        "--finetune_epochs", "40",
        "--batch_size", "128",
        "--workers", "0"
    ]
    print(f"Starting: {exp['dataset']} {exp['model']} {exp['method']} {exp['ratio']}")
    # log output to file
    log_name = f"log_{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['ratio']}.txt"
    with open(log_name, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p

# Main loop
idx = 0
while idx < len(experiments) or len(procs) > 0:
    # Check finished
    for p in procs[:]:
        if p.poll() is not None:
            procs.remove(p)
            
    # Fill queue
    while len(procs) < MAX_CONCURRENT and idx < len(experiments):
        p = run_exp(experiments[idx])
        procs.append(p)
        idx += 1
        time.sleep(2) # Stagger starts
        
    time.sleep(5)

print("All experiments completed!")
