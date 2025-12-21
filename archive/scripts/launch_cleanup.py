import subprocess
import sys
import os
import time

# Configuration
MAX_PARALLEL = 2 # Safe parallelism for cleanup

def run_cmd(cmd, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        print(f"Starting: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    python_exe = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
    script_path = "run_multi_seed.py"
    
    experiments = [
        # ResNet110 Missing/Failed
        {'dataset': 'cifar10', 'model': 'resnet110', 'method': 'l1', 'prune_ratio': '0.3'},
        {'dataset': 'cifar10', 'model': 'resnet110', 'method': 'l1', 'prune_ratio': '0.5'},
        {'dataset': 'cifar10', 'model': 'resnet110', 'method': 'fpgm', 'prune_ratio': '0.5'},
        {'dataset': 'cifar10', 'model': 'resnet110', 'method': 'gra', 'prune_ratio': '0.3'},
        
        # VGG16 L1 (Failed due to bug)
        {'dataset': 'cifar10', 'model': 'vgg16', 'method': 'l1', 'prune_ratio': '0.3'},
        {'dataset': 'cifar10', 'model': 'vgg16', 'method': 'l1', 'prune_ratio': '0.5'},
        {'dataset': 'cifar10', 'model': 'vgg16', 'method': 'l1', 'prune_ratio': '0.7'},
    ]
    
    print(f"Cleanup: {len(experiments)} experiments to retry.")
    
    running_procs = []
    queue = experiments[:]
    
    while queue or running_procs:
        # Check running
        for p_info in running_procs[:]:
            p, log = p_info
            if p.poll() is not None:
                running_procs.remove(p_info)
                if p.returncode != 0:
                    print(f"[FAILED] {log}")
                else:
                    print(f"[DONE] {log}")
        
        # Launch new
        while len(running_procs) < MAX_PARALLEL and queue:
            exp = queue.pop(0)
            
            cmd = [python_exe, '-u', script_path, 
                   '--dataset', exp['dataset'],
                   '--model', exp['model'],
                   '--method', exp['method'],
                   '--prune_ratio', exp['prune_ratio'],
                   '--seeds', '0', '1', '2']
            
            log_name = f"logs/exp_{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['prune_ratio']}.log"
            
            print(f"Retrying: {log_name}")
            p = run_cmd(cmd, log_name)
            running_procs.append((p, log_name))
        
        time.sleep(2)
        
    print("Cleanup completed.")
    
    # Generate Charts & PDF again
    print("Regenerating PDF...")
    subprocess.run([python_exe, "vis/draw_auto.py"])
    subprocess.run(["pdflatex", "-jobname=GRA-CNN_Final_Report", "manuscript_apin.tex"], cwd="APIN_Submission")

if __name__ == "__main__":
    main()
