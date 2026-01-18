"""
完整12面板实验矩阵运行器
========================
使用所有已验证的基线模型运行完整对比实验
"""

import subprocess
import os
import sys
import time
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = r'C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe'
SCRIPT = r'C:\GRA-CNN\experiments\run_real_pruning.py'
WORK_DIR = r'C:\GRA-CNN'
RESULTS_FILE = r'C:\GRA-CNN\experiments\final_12panel_results.csv'

# 完整12面板配置
CONFIGS = [
    ('ResNet-20', 'CIFAR-10'),
    ('ResNet-32', 'CIFAR-10'),
    ('ResNet-44', 'CIFAR-10'),
    ('ResNet-56', 'CIFAR-10'),
    ('ResNet-110', 'CIFAR-10'),
    ('VGG-16', 'CIFAR-10'),
    ('ResNet-20', 'CIFAR-100'),
    ('ResNet-32', 'CIFAR-100'),
    ('ResNet-44', 'CIFAR-100'),
    ('ResNet-56', 'CIFAR-100'),
    ('ResNet-110', 'CIFAR-100'),
    ('VGG-16', 'CIFAR-100'),
]

METHODS = ['GRA', 'L1', 'FPGM', 'HRank']
RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]

def run_experiment(arch, dataset, method, ratio):
    cmd = [
        PYTHON, SCRIPT,
        '--arch', arch,
        '--dataset', dataset,
        '--method', method,
        '--ratio', str(ratio),
        '--epochs', '40'
    ]
    
    task_id = f"{arch}/{dataset}/{method}/{ratio}"
    
    try:
        result = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True, timeout=2400)
        return {'task_id': task_id, 'status': 'OK', 'arch': arch, 'dataset': dataset, 'method': method, 'ratio': ratio}
    except subprocess.TimeoutExpired:
        return {'task_id': task_id, 'status': 'TIMEOUT', 'arch': arch, 'dataset': dataset, 'method': method, 'ratio': ratio}
    except Exception as e:
        return {'task_id': task_id, 'status': f'ERROR: {e}', 'arch': arch, 'dataset': dataset, 'method': method, 'ratio': ratio}

def main():
    print("="*70)
    print("完整12面板实验矩阵运行器")
    print(f"开始时间: {datetime.now()}")
    print("="*70)
    
    # 生成所有任务
    tasks = []
    for arch, dataset in CONFIGS:
        for method in METHODS:
            for ratio in RATIOS:
                tasks.append((arch, dataset, method, ratio))
    
    total = len(tasks)
    print(f"总任务数: {total}")
    print(f"配置: {len(CONFIGS)} archs × {len(METHODS)} methods × {len(RATIOS)} ratios")
    print()
    
    # 检查已完成的实验
    completed_ids = set()
    if os.path.exists(RESULTS_FILE):
        try:
            existing = pd.read_csv(RESULTS_FILE)
            for _, row in existing.iterrows():
                tid = f"{row['architecture']}/{row['dataset']}/{row['method']}/{row['ratio']}"
                completed_ids.add(tid)
            print(f"已完成: {len(completed_ids)} 个实验")
        except:
            pass
    
    # 过滤已完成
    pending = [(a, d, m, r) for a, d, m, r in tasks 
               if f"{a}/{d}/{m}/{r}" not in completed_ids]
    
    print(f"待运行: {len(pending)} 个实验")
    print()
    
    if not pending:
        print("所有实验已完成!")
        return
    
    # 并行执行
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_experiment, *task): task for task in pending}
        
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            status_icon = "✓" if res['status'] == 'OK' else "✗"
            print(f"[{completed}/{len(pending)}] {status_icon} {res['task_id']}")
            results.append(res)
            
            # 每10个保存一次
            if completed % 10 == 0:
                save_results(results)
    
    save_results(results)
    print()
    print("="*70)
    print(f"完成! 共 {len(results)} 个实验")
    print(f"结束时间: {datetime.now()}")
    print("="*70)

def save_results(results):
    if not results:
        return
    
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
    else:
        existing = pd.DataFrame()
    
    new_rows = []
    for r in results:
        if r['status'] == 'OK':
            new_rows.append({
                'architecture': r['arch'],
                'dataset': r['dataset'],
                'method': r['method'],
                'ratio': r['ratio'],
                'status': r['status'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(RESULTS_FILE, index=False)

if __name__ == '__main__':
    main()
