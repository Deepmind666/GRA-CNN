import subprocess
import sys
import time

def run_cmd(cmd, log_file):
    with open(log_file, "w") as f:
        print(f"Starting: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    procs = []
    
    # 1. Latency Benchmark (Runs quickly)
    p = run_cmd([sys.executable, "benchmark_latency.py"], "log_latency.txt")
    p.wait() # Wait for this one as it uses GPU for timing and we don't want interference
    
    # 2. ResNet-110 (Medium load)
    procs.append(run_cmd([sys.executable, "run_resnet110_cifar10.py"], "log_r110.txt"))
    
    # 3. Tiny-ImageNet Extended (Heavy load)
    procs.append(run_cmd([sys.executable, "run_tinyimagenet_resnet18_extended.py"], "log_tiny_ext.txt"))
    
    # 4. Multi-Seed Experiments (Parallelized)
    # Compare L1 vs GRA on ResNet-56 Ratio 0.5 with 3 seeds
    cmd_base = [sys.executable, "run_multi_seed.py", "--dataset", "cifar10", "--model", "resnet56", "--prune_ratio", "0.5", "--epochs", "40"]
    
    procs.append(run_cmd(cmd_base + ["--method", "l1"], "log_seed_l1.txt"))
    procs.append(run_cmd(cmd_base + ["--method", "gra"], "log_seed_gra.txt"))
    
    # Monitor
    while procs:
        for p in procs[:]:
            if p.poll() is not None:
                procs.remove(p)
        time.sleep(5)
        
    print("All massive experiments finished.")

if __name__ == "__main__":
    main()
