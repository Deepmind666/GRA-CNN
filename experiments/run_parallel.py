"""
并行实验启动器 - 充分利用RTX 5090 GPU
======================================
同时运行4个方法的实验
"""

import subprocess
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = r'C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe'
SCRIPT = r'C:\GRA-CNN\experiments\run_real_pruning.py'
WORK_DIR = r'C:\GRA-CNN'

# 实验配置
CONFIGS = [
    ('ResNet-20', 'CIFAR-10'),
    ('ResNet-56', 'CIFAR-10'),
    ('ResNet-20', 'CIFAR-100'),
    ('ResNet-56', 'CIFAR-100'),
]
METHODS = ['GRA', 'L1', 'FPGM', 'HRank']
RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]

def run_experiment(arch, dataset, method, ratio):
    """运行单个实验"""
    cmd = [
        PYTHON, SCRIPT,
        '--arch', arch,
        '--dataset', dataset,
        '--method', method,
        '--ratio', str(ratio),
        '--epochs', '40'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )
        return f"{arch}/{dataset}/{method}/{ratio}: OK"
    except subprocess.TimeoutExpired:
        return f"{arch}/{dataset}/{method}/{ratio}: TIMEOUT"
    except Exception as e:
        return f"{arch}/{dataset}/{method}/{ratio}: ERROR - {e}"

def main():
    print("="*60)
    print("并行实验启动器")
    print("充分利用 RTX 5090 GPU")
    print("="*60)
    
    # 生成所有实验任务
    tasks = []
    for arch, dataset in CONFIGS:
        for method in METHODS:
            for ratio in RATIOS:
                tasks.append((arch, dataset, method, ratio))
    
    print(f"总共 {len(tasks)} 个实验任务")
    print(f"并行进程数: 4")
    print()
    
    # 使用4个并行进程
    completed = 0
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_experiment, *task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed += 1
            print(f"[{completed}/{len(tasks)}] {result}")
    
    print()
    print("="*60)
    print(f"完成! {completed} 个实验")
    print("="*60)

if __name__ == '__main__':
    main()
