import subprocess
import sys
import time
import os

def run_cmd(cmd, log_file):
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        print(f"Starting: {' '.join(cmd)}")
        # Use Popen to run in background (managed by this script)
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

def main():
    python_exe = sys.executable
    script_path = "run_multi_seed.py"
    
    # Priority 1: ResNet-110 on CIFAR-10 (Fig 8)
    # Methods: l1, fpgm, gra. Ratios: 0.3, 0.5, 0.7
    experiments_r110 = []
    for r in [0.3, 0.5, 0.7]:
        for m in ['l1', 'fpgm', 'gra']:
            experiments_r110.append({
                'dataset': 'cifar10', 'model': 'resnet110', 'method': m, 'prune_ratio': str(r)
            })

    # Priority 2: VGG-16 on CIFAR-10 (Fig 9)
    experiments_vgg = []
    for r in [0.3, 0.5, 0.7]:
        for m in ['l1', 'fpgm', 'gra']:
            experiments_vgg.append({
                'dataset': 'cifar10', 'model': 'vgg16', 'method': m, 'prune_ratio': str(r)
            })

    # Priority 3: Rho Sensitivity (Fig 10)
    # ResNet-20 and ResNet-56, Ratio 0.5, varying Rho
    experiments_rho = []
    rhos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for model in ['resnet20', 'resnet56']:
        for rho in rhos:
            experiments_rho.append({
                'dataset': 'cifar10', 'model': model, 'method': 'gra', 'prune_ratio': '0.5', 'rho': str(rho), 'seeds': '0' # Single seed for sensitivity
            })

    # Priority 4: CIFAR-100 & Tiny-ImageNet (Tables)
    experiments_extra = []
    # CIFAR-100 ResNet-56
    for m in ['l1', 'fpgm', 'gra']:
        experiments_extra.append({'dataset': 'cifar100', 'model': 'resnet56', 'method': m, 'prune_ratio': '0.5'})
    
    # Tiny-ImageNet ResNet-18
    for m in ['l1', 'fpgm', 'gra']:
        experiments_extra.append({'dataset': 'tinyimagenet', 'model': 'resnet18', 'method': m, 'prune_ratio': '0.5'})

    # Combine all (You can adjust order)
    all_experiments = experiments_r110 + experiments_vgg + experiments_rho + experiments_extra
    
    print(f"Total experiments scheduled: {len(all_experiments)}")
    
    # Execution Loop
    # We can run 1 experiment at a time to be safe with GPU memory (especially for Tiny/R110)
    # If you have multiple GPUs, you could parallelize. Assuming 1 GPU (RTX 5090 is powerful but VRAM is shared).
    
    for i, exp in enumerate(all_experiments):
        print(f"=== Running Experiment {i+1}/{len(all_experiments)} ===")
        
        cmd = [python_exe, script_path, 
               '--dataset', exp['dataset'],
               '--model', exp['model'],
               '--method', exp['method'],
               '--prune_ratio', exp['prune_ratio']]
        
        if 'rho' in exp:
            cmd.extend(['--rho', exp['rho']])
            
        if 'seeds' in exp:
            cmd.extend(['--seeds', exp['seeds']])
        else:
            # Default 3 seeds
            cmd.extend(['--seeds', '0', '1', '2'])
            
        log_name = f"logs/exp_{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['prune_ratio']}.log"
        if 'rho' in exp:
             log_name = f"logs/exp_{exp['dataset']}_{exp['model']}_{exp['method']}_{exp['prune_ratio']}_rho{exp['rho']}.log"
             
        p = run_cmd(cmd, log_name)
        exit_code = p.wait()
        
        if exit_code != 0:
            print(f"Experiment failed! Check {log_name}")
        else:
            print("Experiment finished successfully.")
            
    print("All experiments completed.")
    
    # Generate Charts
    print("Generating Charts...")
    subprocess.run([python_exe, "vis/draw_auto.py"])
    
    # Compile PDF
    print("Compiling PDF...")
    subprocess.run(["pdflatex", "-jobname=GRA-CNN_Final_Report", "manuscript_apin.tex"], cwd="APIN_Submission")

if __name__ == "__main__":
    main()
