"""
GRA 10小时通宵深度优化实验套件
==============================

理论创新:
1. Sample-Weighted Fisher: 困难样本权重更高
2. Channel Orthogonality: 去除冗余通道
3. Class-Balanced Fisher: 类别平衡
4. Gradient Stability: 梯度稳定性

权重扫描:
- F50: Fisher 50% + GRA 30% + L1 20% (当前)
- F60: Fisher 60% + GRA 25% + L1 15%
- F70: Fisher 70% + GRA 20% + L1 10%
- F40: Fisher 40% + GRA 40% + L1 20%
- F55: Fisher 55% + GRA 30% + L1 15%
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import subprocess
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys

PROJECT_ROOT = Path(r"C:\GRA-CNN")
PYTHON = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"

# ============================================================================
# 算法变体定义
# ============================================================================

WEIGHT_CONFIGS = {
    'F50': {'fisher': 0.50, 'gra': 0.30, 'l1': 0.20},  # 当前最优
    'F60': {'fisher': 0.60, 'gra': 0.25, 'l1': 0.15},  # Fisher主导
    'F70': {'fisher': 0.70, 'gra': 0.20, 'l1': 0.10},  # Fisher极端
    'F40': {'fisher': 0.40, 'gra': 0.40, 'l1': 0.20},  # 平衡
    'F55': {'fisher': 0.55, 'gra': 0.30, 'l1': 0.15},  # 微调
    'F65': {'fisher': 0.65, 'gra': 0.25, 'l1': 0.10},  # 高Fisher
    'F45': {'fisher': 0.45, 'gra': 0.35, 'l1': 0.20},  # 高GRA
}

ALGORITHM_VARIANTS = [
    'fisher_weighted',      # 样本加权Fisher
    'fisher_orthogonal',    # 通道正交性
    'fisher_class_balanced', # 类别平衡
    'fisher_stable',        # 梯度稳定性
]

CORE_CONFIGS = [
    ('ResNet-56', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.3),
    ('VGG-16', 'CIFAR-10', 0.5),
]

FULL_CONFIGS = [
    ('ResNet-20', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.3),
    ('ResNet-56', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.7),
    ('VGG-16', 'CIFAR-10', 0.5),
    ('ResNet-110', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-100', 0.5),
]

# ============================================================================
# 实验运行器
# ============================================================================

def run_single_experiment(arch, dataset, ratio, weight_config=None, variant=None, epochs=40):
    """运行单个实验"""
    script_path = PROJECT_ROOT / "experiments" / "run_gra_variant.py"
    
    cmd = [
        PYTHON, str(script_path),
        "--arch", arch,
        "--dataset", dataset,
        "--ratio", str(ratio),
        "--epochs", str(epochs)
    ]
    
    if weight_config:
        cmd.extend(["--fisher_weight", str(weight_config['fisher'])])
        cmd.extend(["--gra_weight", str(weight_config['gra'])])
        cmd.extend(["--l1_weight", str(weight_config['l1'])])
    
    if variant:
        cmd.extend(["--variant", variant])
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, 
            timeout=3600, cwd=str(PROJECT_ROOT)
        )
        return result.returncode == 0
    except Exception as e:
        return False


def run_weight_scan_phase():
    """第一阶段: 权重扫描"""
    print("\n" + "="*60)
    print("第一阶段: 权重配比扫描")
    print("="*60)
    
    results = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        for arch, dataset, ratio in CORE_CONFIGS:
            exp_id = f"{config_name}_{arch}_{dataset}_{ratio}".replace("-", "")
            print(f"运行: {exp_id}")
            
            success = run_single_experiment(arch, dataset, ratio, weights)
            results.append({
                'phase': 'weight_scan',
                'config': config_name,
                'arch': arch,
                'dataset': dataset,
                'ratio': ratio,
                'success': success
            })
            
            if success:
                print(f"  ✓ 成功")
            else:
                print(f"  ✗ 失败")
    
    return results


def run_variant_phase():
    """第二阶段: 算法变体验证"""
    print("\n" + "="*60)
    print("第二阶段: 算法变体验证")
    print("="*60)
    
    results = []
    for variant in ALGORITHM_VARIANTS:
        for arch, dataset, ratio in CORE_CONFIGS:
            exp_id = f"{variant}_{arch}_{dataset}_{ratio}".replace("-", "")
            print(f"运行: {exp_id}")
            
            success = run_single_experiment(arch, dataset, ratio, variant=variant)
            results.append({
                'phase': 'variant',
                'variant': variant,
                'arch': arch,
                'dataset': dataset,
                'ratio': ratio,
                'success': success
            })
            
            if success:
                print(f"  ✓ 成功")
            else:
                print(f"  ✗ 失败")
    
    return results


def run_full_coverage_phase(best_config):
    """第三阶段: 最优配置全覆盖"""
    print("\n" + "="*60)
    print(f"第三阶段: 最优配置全覆盖 ({best_config})")
    print("="*60)
    
    weights = WEIGHT_CONFIGS[best_config]
    results = []
    
    for arch, dataset, ratio in FULL_CONFIGS:
        exp_id = f"{best_config}_{arch}_{dataset}_{ratio}".replace("-", "")
        print(f"运行: {exp_id}")
        
        success = run_single_experiment(arch, dataset, ratio, weights)
        results.append({
            'phase': 'full_coverage',
            'config': best_config,
            'arch': arch,
            'dataset': dataset,
            'ratio': ratio,
            'success': success
        })
        
        if success:
            print(f"  ✓ 成功")
        else:
            print(f"  ✗ 失败")
    
    return results


def analyze_results():
    """分析实验结果"""
    print("\n" + "="*60)
    print("结果分析")
    print("="*60)
    
    try:
        df = pd.read_csv(PROJECT_ROOT / "experiments" / "overnight_deep_results.csv")
        
        # 按权重配置分组
        print("\n权重配置对比:")
        for config in WEIGHT_CONFIGS.keys():
            subset = df[df['weight_config'] == config]
            if len(subset) > 0:
                mean_acc = subset['final_acc'].mean()
                print(f"  {config}: {mean_acc:.2f}%")
        
        # 找最优
        best = df.groupby('weight_config')['final_acc'].mean().idxmax()
        print(f"\n最优权重配置: {best}")
        
        return best
    except Exception as e:
        print(f"分析失败: {e}")
        return 'F50'


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"GRA 10小时深度优化实验")
    print(f"开始时间: {start_time}")
    print(f"预计结束: {start_time.replace(hour=start_time.hour+10)}")
    print("="*60)
    
    all_results = []
    
    # 第一阶段
    phase1_results = run_weight_scan_phase()
    all_results.extend(phase1_results)
    
    # 分析得到最优权重
    best_config = analyze_results()
    
    # 第二阶段
    phase2_results = run_variant_phase()
    all_results.extend(phase2_results)
    
    # 第三阶段
    phase3_results = run_full_coverage_phase(best_config)
    all_results.extend(phase3_results)
    
    # 保存结果
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*60)
    print(f"实验完成!")
    print(f"结束时间: {end_time}")
    print(f"总耗时: {duration:.2f} 小时")
    print(f"成功率: {sum(r['success'] for r in all_results)}/{len(all_results)}")
    print("="*60)
    
    # 保存日志
    log = {
        'start': str(start_time),
        'end': str(end_time),
        'duration_hours': duration,
        'results': all_results
    }
    
    with open(PROJECT_ROOT / "experiments" / "overnight_deep_log.json", 'w') as f:
        json.dump(log, f, indent=2)
