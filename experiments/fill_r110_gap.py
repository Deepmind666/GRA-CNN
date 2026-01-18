"""
补充ResNet-110/CIFAR-10实验
"""
import subprocess
import os
import sys

PYTHON = r'C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe'
SCRIPT = r'C:\GRA-CNN\experiments\run_real_pruning.py'

tasks = []
for method in ['GRA', 'L1', 'FPGM', 'HRank']:
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
        tasks.append(('ResNet-110', 'CIFAR-10', method, ratio))

print(f'补充 ResNet-110/CIFAR-10: {len(tasks)} 个实验')

for i, (arch, ds, method, ratio) in enumerate(tasks, 1):
    print(f'[{i}/{len(tasks)}] {method}/{ratio}...', end=' ', flush=True)
    cmd = [PYTHON, SCRIPT, '--arch', arch, '--dataset', ds, '--method', method, '--ratio', str(ratio), '--epochs', '40']
    try:
        result = subprocess.run(cmd, cwd=r'C:\GRA-CNN', capture_output=True, text=True, timeout=3600)
        print('OK' if result.returncode == 0 else 'FAILED')
    except Exception as e:
        print(f'ERROR: {e}')

print('完成!')
