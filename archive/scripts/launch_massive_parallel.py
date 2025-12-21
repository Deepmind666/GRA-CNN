import subprocess
import sys
import time
import os

# Configuration
MAX_PARALLEL = 5  # Increased to 5 to fully utilize RTX 5090

def run_cmd(cmd, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        print(f"Starting: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    python_exe = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe" # Explicit path
    script_path = "run_multi_seed.py"
    
    # --- Experiment Definition (Same as v3) ---
    experiments_r110 = []
    for r in [0.3, 0.5, 0.7]:
        for m in ['l1', 'fpgm', 'gra']:
            experiments_r110.append({
                'dataset': 'cifar10', 'model': 'resnet110', 'method': m, 'prune_ratio': str(r)
            })

    experiments_vgg = []
    for r in [0.3, 0.5, 0.7]:
        for m in ['l1', 'fpgm', 'gra']:
            experiments_vgg.append({
                'dataset': 'cifar10', 'model': 'vgg16', 'method': m, 'prune_ratio': str(r)
            })

    experiments_rho = []
    rhos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for model in ['resnet20', 'resnet56']:
        for rho in rhos:
            experiments_rho.append({
                'dataset': 'cifar10', 'model': model, 'method': 'gra', 'prune_ratio': '0.5', 'rho': str(rho), 'seeds': '0'
            })

    experiments_extra = []
    for m in ['l1', 'fpgm', 'gra']:
        experiments_extra.append({'dataset': 'cifar100', 'model': 'resnet56', 'method': m, 'prune_ratio': '0.5'})
    for m in ['l1', 'fpgm', 'gra']:
        experiments_extra.append({'dataset': 'tinyimagenet', 'model': 'resnet18', 'method': m, 'prune_ratio': '0.5'})

    all_experiments = experiments_r110 + experiments_vgg + experiments_rho + experiments_extra
    print(f"Total experiments scheduled: {len(all_experiments)}")
    
    # --- Parallel Execution Loop ---
    running_procs = [] # List of (popen_obj, log_file)
    queue = all_experiments[:] # Copy
    completed_count = 0
    
    while queue or running_procs:
        # 1. Check completed processes
        for p_info in running_procs[:]:
            p, log = p_info
            if p.poll() is not None: # Finished
                running_procs.remove(p_info)
                completed_count += 1
                if p.returncode != 0:
                    print(f"[FAILED] {log} (Exit Code: {p.returncode})")
                else:
                    print(f"[DONE] {log}")

        # 2. Start new processes if slots available
        while len(running_procs) < MAX_PARALLEL and queue:
            exp = queue.pop(0)
            
            # Use -u for unbuffered output to see logs in real-time
            cmd = [python_exe, '-u', script_path, 
                   '--dataset', exp['dataset'],
                   '--model', exp['model'],
                   '--method', exp['method'],
                   '--prune_ratio', exp['prune_ratio']]
            
            if 'rho' in exp:
                cmd.extend(['--rho', exp['rho']])
            
            if 'seeds' in exp:
                cmd.extend(['--seeds', exp['seeds']])
            else:
                cmd.extend(['--seeds', '0', '1', '2'])
            
            log_name = f"logs/exp_{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['prune_ratio']}.log"
            if 'rho' in exp:
                 log_name = f"logs/exp_{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['prune_ratio']}_rho{exp['rho']}.log"
            
            # Check if already done (simple check: if log exists and contains "Final Result")
            # This allows resuming if interrupted
            if os.path.exists(log_name):
                with open(log_name, 'r') as f:
                    content = f.read()
                    if "Final Result" in content:
                        print(f"[SKIP] Already finished: {log_name}")
                        completed_count += 1
                        continue

            p = run_cmd(cmd, log_name)
            running_procs.append((p, log_name))
            print(f"[Running] {len(running_procs)}/{MAX_PARALLEL} | Pending: {len(queue)} | Done: {completed_count}")
        
        time.sleep(2) # Polling interval

    print("All experiments completed.")
    
    # Generate Charts
    print("Generating Charts...")
    subprocess.run([python_exe, "vis/draw_auto.py"])
    
    # Compile PDF
    print("Compiling PDF...")
    subprocess.run(["pdflatex", "-jobname=GRA-CNN_Final_Report", "manuscript_apin.tex"], cwd="APIN_Submission")

if __name__ == "__main__":
    main()
