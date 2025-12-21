import os
import subprocess
import sys

# Configuration
PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, 'code')
EXP_DIR = os.path.join(BASE_DIR, 'experiments', 'day2')
os.makedirs(EXP_DIR, exist_ok=True)

def run_command(cmd, log_file=None):
    print(f"Running: {cmd}")
    if log_file:
        with open(log_file, 'w') as f:
            subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, shell=True)

def check_baseline(model, dataset):
    baseline_path = os.path.join(BASE_DIR, 'experiments', f'baseline_{dataset}_{model}.pth')
    if not os.path.exists(baseline_path):
        print(f"Baseline {baseline_path} not found. Please train it first or wait for other scripts.")
        # Try to find any best checkpoint
        potential = os.path.join(BASE_DIR, 'checkpoints', f'{model}_best.pth')
        if os.path.exists(potential):
            import shutil
            shutil.copy(potential, baseline_path)
            return True
        return False
    return True

def task_gra_comparative_missing_metrics():
    print("\n=== Task 1: Comparative Metrics (Cosine, Correlation, MI) ===")
    model = 'resnet56'
    dataset = 'cifar10'
    if not check_baseline(model, dataset):
        return
    
    baseline_path = os.path.join(BASE_DIR, 'experiments', f'baseline_{dataset}_{model}.pth')
    
    # L1 and GRA are already running in run_multi_seed.py
    metrics = ['cosine', 'correlation', 'mi'] 
    ratio = 0.5
    
    for metric in metrics:
        print(f"--- Metric: {metric} ---")
        work_dir = os.path.join(EXP_DIR, f'comparative_{metric}')
        os.makedirs(work_dir, exist_ok=True)
        
        pruned_ckpt = os.path.join(work_dir, 'pruned.pth')
        
        # Prune
        if not os.path.exists(pruned_ckpt):
            cmd = f"{PYTHON} {os.path.join(CODE_DIR, 'prune_resnet20.py')} --depth 56 --resume {baseline_path} --save {pruned_ckpt} --ratio {ratio} --method {metric}"
            run_command(cmd, log_file=os.path.join(work_dir, 'prune.log'))
        
        # Finetune
        finetuned_log = os.path.join(work_dir, 'training_log.csv')
        if not os.path.exists(finetuned_log):
            cmd = f"{PYTHON} {os.path.join(CODE_DIR, 'finetune_resnet20.py')} --depth 56 --pruned-model {pruned_ckpt} --save-dir {work_dir} --epochs 40"
            run_command(cmd, log_file=os.path.join(work_dir, 'finetune.log'))
        else:
            print(f"Already finetuned {metric}.")

def main():
    task_gra_comparative_missing_metrics()

if __name__ == "__main__":
    main()
