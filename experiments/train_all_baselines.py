"""
并行基线模型训练启动器
========================
同时训练缺失的基线模型以节省时间
"""

import subprocess
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = r'C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe'
SCRIPT = r'C:\GRA-CNN\experiments\train_baseline.py'
WORK_DIR = r'C:\GRA-CNN'

# 缺失的基线模型
TASKS = [
    ('resnet32', 'cifar10'),
    ('resnet32', 'cifar100'),
    ('resnet44', 'cifar10'),
    ('resnet44', 'cifar100'),
    ('resnet110', 'cifar100'),
    ('vgg16', 'cifar10'),
    ('vgg16', 'cifar100'),
]

def run_train(arch, dataset):
    print(f"Starting training: {arch} on {dataset}")
    cmd = [
        PYTHON, SCRIPT,
        '--arch', arch,
        '--dataset', dataset,
        '--epochs', '200'
    ]
    
    try:
        # 使用 stdout=PIPE 捕获输出以防控制台过载
        result = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            return f"{arch}/{dataset}: SUCCESS"
        else:
            return f"{arch}/{dataset}: FAILED - {result.stderr[-500:]}"
    except Exception as e:
        return f"{arch}/{dataset}: ERROR - {str(e)}"

def main():
    print("="*60)
    print("并行基线模型训练启动器")
    print(f"总计任务: {len(TASKS)}")
    print("="*60)
    
    # RTX 5090 够强，可以跑4个并行
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_train, arch, ds): (arch, ds) for arch, ds in TASKS}
        
        completed = 0
        for future in as_completed(futures):
            arch, ds = futures[future]
            res = future.result()
            completed += 1
            print(f"[{completed}/{len(TASKS)}] {res}")

if __name__ == '__main__':
    main()
