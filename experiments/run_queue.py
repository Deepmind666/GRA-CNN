"""
Sequential Experiment Queue
============================
Runs remaining experiments one-by-one to avoid GPU memory overload.
Safe to run while other experiments are finishing.
"""

import subprocess
import os
import sys
import time
from datetime import datetime

PYTHON = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
SCRIPT = r"C:\GRA-CNN\experiments\run_comprehensive.py"
SAVE_DIR = r"C:\GRA-CNN\experiments\comprehensive"
LOG_FILE = r"C:\GRA-CNN\experiments\queue_log.txt"

# Experiment queue - will run sequentially
EXPERIMENT_QUEUE = [
    # ResNet-56 CIFAR-10 (core)
    {
        "archs": "resnet56",
        "datasets": "cifar10",
        "methods": "l1 fpgm hrank gra",
        "ratios": "0.3 0.5 0.7",
        "seeds": "0 1 2",
        "desc": "ResNet-56 CIFAR-10 core"
    },
    # ResNet-20 CIFAR-10 (core)
    {
        "archs": "resnet20",
        "datasets": "cifar10",
        "methods": "l1 fpgm hrank gra",
        "ratios": "0.3 0.5 0.7",
        "seeds": "0 1 2",
        "desc": "ResNet-20 CIFAR-10 core"
    },
    # ResNet-56 CIFAR-100
    {
        "archs": "resnet56",
        "datasets": "cifar100",
        "methods": "l1 fpgm hrank gra",
        "ratios": "0.3 0.5 0.7",
        "seeds": "0 1 2",
        "desc": "ResNet-56 CIFAR-100"
    },
    # ResNet-20 CIFAR-100
    {
        "archs": "resnet20",
        "datasets": "cifar100",
        "methods": "l1 fpgm hrank gra",
        "ratios": "0.3 0.5 0.7",
        "seeds": "0 1 2",
        "desc": "ResNet-20 CIFAR-100"
    },
]

def log(msg):
    """Log message to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def run_experiment(config):
    """Run a single experiment configuration."""
    cmd = [
        PYTHON, SCRIPT,
        "--archs", config["archs"],
        "--datasets", config["datasets"],
        "--methods", *config["methods"].split(),
        "--ratios", *config["ratios"].split(),
        "--seeds", *config["seeds"].split(),
        "--save-dir", SAVE_DIR
    ]
    
    log(f"Starting: {config['desc']}")
    log(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log(f"✅ Completed: {config['desc']} in {elapsed/60:.1f} min")
        else:
            log(f"❌ Failed: {config['desc']} (exit code {result.returncode})")
            log(f"Error: {result.stderr[-500:] if result.stderr else 'No error output'}")
    except subprocess.TimeoutExpired:
        log(f"⏰ Timeout: {config['desc']} exceeded 2 hours")
    except Exception as e:
        log(f"❌ Error: {config['desc']} - {e}")

def main():
    log("="*60)
    log("Starting Sequential Experiment Queue")
    log(f"Total experiments in queue: {len(EXPERIMENT_QUEUE)}")
    log("="*60)
    
    for i, config in enumerate(EXPERIMENT_QUEUE, 1):
        log(f"\n[{i}/{len(EXPERIMENT_QUEUE)}] {config['desc']}")
        run_experiment(config)
        
        # Brief pause between experiments
        time.sleep(5)
    
    log("\n" + "="*60)
    log("Queue completed!")
    log("="*60)

if __name__ == "__main__":
    main()
