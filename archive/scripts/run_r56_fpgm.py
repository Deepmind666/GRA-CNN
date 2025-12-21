import os
import subprocess
import sys

PYTHON = sys.executable

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    # Ratios to test
    ratios = [0.3, 0.5, 0.7]
    
    # Run FPGM
    for r in ratios:
        print(f"\n--- Running ResNet-56 FPGM Ratio {r} ---")
        cmd = [
            PYTHON, "run_experiments.py",
            "--dataset", "cifar10",
            "--model", "resnet56",
            "--method", "fpgm",
            "--prune_ratio", str(r),
            "--epochs", "160",
            "--finetune_epochs", "40", # Faster finetune for baselines as requested
            "--batch_size", "128",
            "--workers", "0"
        ]
        try:
            run_cmd(cmd)
        except Exception as e:
            print(f"Error running FPGM {r}: {e}")

if __name__ == "__main__":
    main()
