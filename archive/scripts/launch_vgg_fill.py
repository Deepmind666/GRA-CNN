import subprocess
import sys
import time

def run_cmd(cmd, log_file):
    with open(log_file, "w") as f:
        print(f"Starting: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    procs = []
    
    # Parallel VGG experiments
    configs = [
        ('l1', '0.5'), ('l1', '0.7'),
        ('gra', '0.5'), ('gra', '0.7')
    ]
    
    for method, ratio in configs:
        log_name = f"log_vgg_{method}_{ratio}.txt"
        cmd = [sys.executable, "run_vgg_cifar.py", "--method", method, "--ratio", ratio]
        procs.append(run_cmd(cmd, log_name))
        time.sleep(2) # Stagger
        
    # Monitor
    while procs:
        for p in procs[:]:
            if p.poll() is not None:
                procs.remove(p)
        time.sleep(5)
        
    print("All VGG experiments finished.")

if __name__ == "__main__":
    main()
