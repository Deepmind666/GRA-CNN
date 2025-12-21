import subprocess
import sys
import time

def run_cmd(cmd, log_file):
    with open(log_file, "w") as f:
        print(f"Starting: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    procs = []
    
    # Parallel Rho Ablation
    configs = [
        ('resnet20', '0.5'),
        ('resnet56', '0.5')
    ]
    
    for model, ratio in configs:
        log_name = f"log_rho_{model}_{ratio}.txt"
        cmd = [sys.executable, "run_rho_ablation_detailed.py", "--model", model, "--ratio", ratio]
        procs.append(run_cmd(cmd, log_name))
        
    # Monitor
    while procs:
        for p in procs[:]:
            if p.poll() is not None:
                procs.remove(p)
        time.sleep(5)
        
    print("All Rho Ablation experiments finished.")

if __name__ == "__main__":
    main()
