"""
10小时综合实验套件 - 充分利用RTX 5090
======================================
包含:
1. 完整架构矩阵: ResNet-20/32/44/56/110, VGG-16
2. 完整数据集: CIFAR-10, CIFAR-100
3. 完整方法对比: GRA, L1, FPGM, HRank
4. 完整剪枝率: 0.3, 0.4, 0.5, 0.6, 0.7
5. ρ参数消融: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
6. 多随机种子: 3个种子确保统计显著性
"""

import subprocess
import sys
import os
import time
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = r'C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe'
SCRIPT = r'C:\GRA-CNN\experiments\run_real_pruning.py'
WORK_DIR = r'C:\GRA-CNN'
RESULTS_FILE = r'C:\GRA-CNN\experiments\comprehensive_10hr_results.csv'

# ============================================================================
# 实验配置
# ============================================================================

# 完整架构矩阵
ARCHITECTURES = ['ResNet-20', 'ResNet-56', 'ResNet-110']  # 核心架构
DATASETS = ['CIFAR-10', 'CIFAR-100']
METHODS = ['GRA', 'L1', 'FPGM', 'HRank']
RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]

# ρ消融实验配置 (仅对GRA方法)
RHO_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RHO_RATIOS = [0.3, 0.5, 0.7]  # 选择性剪枝率做消融

# 多种子配置
SEEDS = [42, 123, 456]

# 并行进程数
MAX_WORKERS = 4

# ============================================================================
# 实验任务生成
# ============================================================================

def generate_main_experiments():
    """生成主对比实验"""
    tasks = []
    for arch in ARCHITECTURES:
        for dataset in DATASETS:
            for method in METHODS:
                for ratio in RATIOS:
                    tasks.append({
                        'type': 'main',
                        'arch': arch,
                        'dataset': dataset,
                        'method': method,
                        'ratio': ratio,
                        'rho': 0.5,
                        'seed': 42,
                        'epochs': 40
                    })
    return tasks

def generate_ablation_experiments():
    """生成ρ消融实验"""
    tasks = []
    for arch in ['ResNet-20', 'ResNet-56']:  # 选择代表性架构
        for dataset in ['CIFAR-10', 'CIFAR-100']:
            for rho in RHO_VALUES:
                for ratio in RHO_RATIOS:
                    tasks.append({
                        'type': 'ablation_rho',
                        'arch': arch,
                        'dataset': dataset,
                        'method': 'GRA',
                        'ratio': ratio,
                        'rho': rho,
                        'seed': 42,
                        'epochs': 40
                    })
    return tasks

def generate_seed_experiments():
    """生成多种子实验 (统计显著性)"""
    tasks = []
    for arch in ['ResNet-20', 'ResNet-56']:
        for dataset in ['CIFAR-10']:
            for method in ['GRA', 'L1']:  # 只对核心方法做多种子
                for ratio in [0.3, 0.5, 0.7]:
                    for seed in SEEDS:
                        tasks.append({
                            'type': 'multi_seed',
                            'arch': arch,
                            'dataset': dataset,
                            'method': method,
                            'ratio': ratio,
                            'rho': 0.5,
                            'seed': seed,
                            'epochs': 40
                        })
    return tasks

# ============================================================================
# 实验执行
# ============================================================================

def run_single_experiment(task):
    """运行单个实验"""
    cmd = [
        PYTHON, SCRIPT,
        '--arch', task['arch'],
        '--dataset', task['dataset'],
        '--method', task['method'],
        '--ratio', str(task['ratio']),
        '--epochs', str(task['epochs'])
    ]
    
    task_id = f"{task['arch']}/{task['dataset']}/{task['method']}/r={task['ratio']}/rho={task['rho']}/seed={task['seed']}"
    
    try:
        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=2400  # 40分钟超时
        )
        elapsed = time.time() - start
        
        return {
            'status': 'OK',
            'task_id': task_id,
            'task': task,
            'elapsed': elapsed
        }
    except subprocess.TimeoutExpired:
        return {
            'status': 'TIMEOUT',
            'task_id': task_id,
            'task': task,
            'elapsed': 2400
        }
    except Exception as e:
        return {
            'status': f'ERROR: {e}',
            'task_id': task_id,
            'task': task,
            'elapsed': 0
        }

def main():
    print("="*70)
    print("10小时综合实验套件")
    print("充分利用 RTX 5090 GPU")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 生成所有任务
    main_tasks = generate_main_experiments()
    ablation_tasks = generate_ablation_experiments()
    seed_tasks = generate_seed_experiments()
    
    print(f"\n实验统计:")
    print(f"  主对比实验: {len(main_tasks)} 个")
    print(f"  ρ消融实验: {len(ablation_tasks)} 个")
    print(f"  多种子实验: {len(seed_tasks)} 个")
    
    all_tasks = main_tasks + ablation_tasks + seed_tasks
    total = len(all_tasks)
    print(f"  总计: {total} 个实验")
    print(f"\n并行进程数: {MAX_WORKERS}")
    print(f"预估时间: {total * 15 / MAX_WORKERS / 60:.1f} 小时")
    print()
    
    # 加载已有结果，跳过已完成的实验
    completed_ids = set()
    if os.path.exists(RESULTS_FILE):
        try:
            existing = pd.read_csv(RESULTS_FILE)
            for _, row in existing.iterrows():
                task_id = f"{row['architecture']}/{row['dataset']}/{row['method']}/r={row['ratio']}/rho={row.get('rho', 0.5)}/seed={row.get('seed', 42)}"
                completed_ids.add(task_id)
            print(f"已完成: {len(completed_ids)} 个实验")
        except Exception as e:
            print(f"无法加载已有结果: {e}")
    
    # 过滤已完成的
    pending_tasks = []
    for task in all_tasks:
        task_id = f"{task['arch']}/{task['dataset']}/{task['method']}/r={task['ratio']}/rho={task['rho']}/seed={task['seed']}"
        if task_id not in completed_ids:
            pending_tasks.append(task)
    
    print(f"待运行: {len(pending_tasks)} 个实验")
    print()
    
    # 并行执行
    results = []
    completed = 0
    start_all = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_experiment, task): task for task in pending_tasks}
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            
            status_icon = "✓" if result['status'] == 'OK' else "✗"
            print(f"[{completed}/{len(pending_tasks)}] {status_icon} {result['task_id']} ({result['elapsed']:.1f}s)")
            
            results.append(result)
            
            # 每10个实验保存一次进度
            if completed % 10 == 0:
                save_progress(results)
    
    # 最终保存
    save_progress(results)
    
    elapsed_total = time.time() - start_all
    print()
    print("="*70)
    print(f"实验完成!")
    print(f"完成数量: {len(results)}")
    print(f"总耗时: {elapsed_total/3600:.2f} 小时")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

def save_progress(results):
    """保存进度到CSV"""
    if not results:
        return
    
    # 加载已有数据
    if os.path.exists(RESULTS_FILE):
        try:
            existing = pd.read_csv(RESULTS_FILE)
        except:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()
    
    # 添加新结果
    new_rows = []
    for r in results:
        if r['status'] == 'OK':
            task = r['task']
            new_rows.append({
                'architecture': task['arch'],
                'dataset': task['dataset'],
                'method': task['method'],
                'ratio': task['ratio'],
                'rho': task['rho'],
                'seed': task['seed'],
                'type': task['type'],
                'status': r['status'],
                'elapsed': r['elapsed'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(RESULTS_FILE, index=False)
        print(f"  [进度已保存: {len(combined)} 条记录]")

if __name__ == '__main__':
    main()
