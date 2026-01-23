"""
10小时通宵深度优化实验 - 简化版
================================

直接运行多个权重配比和算法变体实验
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path
import time

PROJECT_ROOT = Path(r"C:\GRA-CNN")
PYTHON = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
SCRIPT = str(PROJECT_ROOT / "experiments" / "run_gra_variant.py")

# 实验配置
EXPERIMENTS = []

# 第一阶段: 权重扫描 (核心配置)
WEIGHT_CONFIGS = [
    ('F50', 0.50, 0.30, 0.20),
    ('F55', 0.55, 0.30, 0.15),
    ('F60', 0.60, 0.25, 0.15),
    ('F65', 0.65, 0.25, 0.10),
    ('F70', 0.70, 0.20, 0.10),
    ('F45', 0.45, 0.35, 0.20),
    ('F40', 0.40, 0.40, 0.20),
]

CORE_ARCHS = [
    ('ResNet-56', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.3),
    ('VGG-16', 'CIFAR-10', 0.5),
]

for name, fw, gw, lw in WEIGHT_CONFIGS:
    for arch, dataset, ratio in CORE_ARCHS:
        EXPERIMENTS.append({
            'phase': 'weight_scan',
            'name': f"{name}_{arch}_{ratio}",
            'arch': arch,
            'dataset': dataset,
            'ratio': ratio,
            'fisher_weight': fw,
            'gra_weight': gw,
            'l1_weight': lw,
            'variant': None
        })

# 第二阶段: 算法变体
VARIANTS = ['fisher_weighted', 'fisher_orthogonal', 'fisher_class_balanced', 'fisher_stable']
for variant in VARIANTS:
    for arch, dataset, ratio in CORE_ARCHS:
        EXPERIMENTS.append({
            'phase': 'variant',
            'name': f"{variant}_{arch}_{ratio}",
            'arch': arch,
            'dataset': dataset,
            'ratio': ratio,
            'fisher_weight': 0.50,
            'gra_weight': 0.30,
            'l1_weight': 0.20,
            'variant': variant
        })

# 第三阶段: 最优配置全覆盖
FULL_ARCHS = [
    ('ResNet-20', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-10', 0.7),
    ('ResNet-110', 'CIFAR-10', 0.5),
    ('ResNet-56', 'CIFAR-100', 0.5),
]

for arch, dataset, ratio in FULL_ARCHS:
    EXPERIMENTS.append({
        'phase': 'full_coverage',
        'name': f"best_{arch}_{ratio}",
        'arch': arch,
        'dataset': dataset,
        'ratio': ratio,
        'fisher_weight': 0.55,  # 预设最优
        'gra_weight': 0.30,
        'l1_weight': 0.15,
        'variant': None
    })


def run_experiment(exp):
    """运行单个实验"""
    cmd = [
        PYTHON, SCRIPT,
        "--arch", exp['arch'],
        "--dataset", exp['dataset'],
        "--ratio", str(exp['ratio']),
        "--epochs", "40",
        "--fisher_weight", str(exp['fisher_weight']),
        "--gra_weight", str(exp['gra_weight']),
        "--l1_weight", str(exp['l1_weight']),
    ]
    
    if exp['variant']:
        cmd.extend(["--variant", exp['variant']])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(PROJECT_ROOT))
        return result.returncode == 0
    except Exception as e:
        print(f"    异常: {e}")
        return False


def main():
    start_time = datetime.now()
    print("="*70)
    print(f"GRA 10小时深度优化实验")
    print(f"开始时间: {start_time}")
    print(f"总实验数: {len(EXPERIMENTS)}")
    print("="*70)
    
    results = []
    success_count = 0
    
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp['phase']}: {exp['name']}")
        
        success = run_experiment(exp)
        exp['success'] = success
        results.append(exp)
        
        if success:
            success_count += 1
            print(f"    ✓ 成功")
        else:
            print(f"    ✗ 失败")
        
        # 每10个实验保存一次
        if (i + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            print(f"\n--- 进度: {i+1}/{len(EXPERIMENTS)}, 成功率: {success_count}/{i+1}, 耗时: {elapsed:.1f}h ---")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*70)
    print(f"实验完成!")
    print(f"结束时间: {end_time}")
    print(f"总耗时: {duration:.2f} 小时")
    print(f"成功率: {success_count}/{len(EXPERIMENTS)} ({100*success_count/len(EXPERIMENTS):.0f}%)")
    print("="*70)
    
    # 保存日志
    log = {
        'start': str(start_time),
        'end': str(end_time),
        'duration_hours': duration,
        'total': len(EXPERIMENTS),
        'success': success_count,
        'results': results
    }
    
    log_file = PROJECT_ROOT / "experiments" / "overnight_deep_log.json"
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"\n日志已保存: {log_file}")


if __name__ == "__main__":
    main()
