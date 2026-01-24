import os
import time
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================================
# 实验矩阵配置 (全量覆盖论文图表)
# ============================================================================

# 架构列表
ARCHS = ['resnet20', 'resnet56', 'resnet110', 'vgg16', 'mobilenetv2', 'resnet18']
# 数据集列表
DATASETS = ['cifar10', 'cifar100', 'tinyimagenet']
# 方法列表
METHODS = ['gra', 'l1', 'fpgm', 'hrank']
# 剪枝率列表
RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7] # 细化剪枝率以支持平滑曲线

experiments = []

# 1. CIFAR 全量跑测 (用于主表和性能曲线)
for arch in ['resnet20', 'resnet56', 'resnet110', 'vgg16', 'mobilenetv2']:
    for dataset in ['cifar10', 'cifar100']:
        for method in METHODS:
            for ratio in [0.5, 0.7]: # 典型点
                experiments.append({'arch': arch, 'dataset': dataset, 'method': method, 'ratio': ratio})

# 2. 详细剪枝率曲线数据 (仅 ResNet-56 & VGG-16)
for arch in ['resnet56', 'vgg16']:
    for dataset in ['cifar10']:
        for method in ['gra', 'l1']:
            for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
                experiments.append({'arch': arch, 'dataset': dataset, 'method': method, 'ratio': ratio})

# 3. Tiny-ImageNet 泛化性验证 (ResNet-18 & MobileNetV2)
for arch in ['resnet18', 'mobilenetv2']:
    for dataset in ['tinyimagenet']:
        for method in ['gra', 'l1', 'fpgm']:
            for ratio in [0.5]:
                experiments.append({'arch': arch, 'dataset': dataset, 'method': method, 'ratio': ratio})

# 去重
unique_exps = []
seen = set()
for e in experiments:
    key = f"{e['arch']}_{e['dataset']}_{e['method']}_{e['ratio']}"
    if key not in seen:
        unique_exps.append(e)
        seen.add(key)

total_exps = len(unique_exps)

# ============================================================================
# 执行逻辑 (并行化)
# ============================================================================

LOG_FILE = r'C:\GRA-CNN\experiments\master_run_log.txt'
MAX_WORKERS = 8 # RTX 5090 Blackwell Turbo Mode: 8路并行以最大化吞吐量

def log(msg):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{t}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except:
        pass

def run_exp(exp):
    # 直接使用 gra311 的 python.exe 避免 conda run 在并行时的临时文件冲突
    PYTHON_EXE = r'C:\Users\admin\anaconda3\envs\gra311\python.exe'
    
    cmd = [
        PYTHON_EXE, r'C:\GRA-CNN\experiments\run_real_pruning.py',
        '--arch', exp['arch'],
        '--dataset', exp['dataset'],
        '--method', exp['method'],
        '--ratio', str(exp['ratio']),
        '--epochs', '40'
    ]
    
    start_time = time.time()
    try:
        # 运行实验
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        return True, exp, duration, ""
    except subprocess.CalledProcessError as e:
        return False, exp, 0, e.stderr

def main():
    log(f"=== Parallel Master Runner Started (Workers: {MAX_WORKERS}) ===")
    log(f"Total experiments in queue: {total_exps}")
    
    start_all = time.time()
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_exp, exp): exp for exp in unique_exps}
        
        for future in as_completed(futures):
            success, exp, duration, error = future.result()
            if success:
                completed += 1
                log(f"SUCCESS [{completed+failed}/{total_exps}]: {exp['arch']}@{exp['dataset']} {exp['method']} {exp['ratio']} | Time: {duration/60:.2f}m")
            else:
                failed += 1
                log(f"FAILED [{completed+failed}/{total_exps}]: {exp['arch']}@{exp['dataset']} {exp['method']} {exp['ratio']} | Error: {error[:200]}...")
            
            # 计算进度与 ETA
            elapsed = time.time() - start_all
            avg_per_exp = elapsed / (completed + failed)
            remaining_tasks = total_exps - (completed + failed)
            # 由于并行执行，剩余时间需除以并行度估计
            eta_seconds = (remaining_tasks * avg_per_exp) / 1.0 # 粗略估计
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            log(f"Progress: {completed+failed}/{total_exps} | ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    log(f"\n=== Master Runner Finished ===")
    log(f"Completed: {completed} | Failed: {failed}")
    log(f"Total Time: {(time.time()-start_all)/3600:.2f} hours")

if __name__ == '__main__':
    # 确保日志文件清空
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"--- High Intensity Run {datetime.now()} ---\n")
    main()
