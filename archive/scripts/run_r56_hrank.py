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
    
    # Run HRank
    for r in ratios:
        print(f"\n--- Running ResNet-56 HRank Ratio {r} ---")
        cmd = [
            PYTHON, "run_experiments.py",
            "--dataset", "cifar10",
            "--model", "resnet56",
            "--method", "hrank",
            "--prune_ratio", str(r),
            "--epochs", "160",
            "--finetune_epochs", "40",
            "--batch_size", "128",
            "--workers", "0"
        ]
        try:
            run_cmd(cmd)
        except Exception as e:
            print(f"Error running HRank {r}: {e}")

if __name__ == "__main__":
    main()
