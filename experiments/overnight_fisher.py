"""
10小时通宵权重扫描实验 - 直接版
================================

直接调用run_real_pruning.py进行实验
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import time

PROJECT_ROOT = Path(r"C:\GRA-CNN")
RESULTS_FILE = PROJECT_ROOT / "experiments" / "supplementary_results.csv"

# 实验配置: 直接使用已有的run_real_pruning.py
EXPERIMENTS = [
    # 核心配置 - 多次运行收集数据
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.3, "epochs": 40},
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.7, "epochs": 40},
    {"arch": "VGG-16", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    {"arch": "ResNet-20", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    {"arch": "ResNet-110", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    {"arch": "ResNet-56", "dataset": "CIFAR-100", "ratio": 0.5, "epochs": 40},
    # 重复核心配置获取更多数据点
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.3, "epochs": 40},
    {"arch": "VGG-16", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    # 额外剪枝率
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.4, "epochs": 40},
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.6, "epochs": 40},
    {"arch": "VGG-16", "dataset": "CIFAR-10", "ratio": 0.3, "epochs": 40},
    {"arch": "VGG-16", "dataset": "CIFAR-10", "ratio": 0.7, "epochs": 40},
    # 更多架构覆盖
    {"arch": "ResNet-20", "dataset": "CIFAR-10", "ratio": 0.3, "epochs": 40},
    {"arch": "ResNet-20", "dataset": "CIFAR-10", "ratio": 0.7, "epochs": 40},
    {"arch": "ResNet-110", "dataset": "CIFAR-10", "ratio": 0.3, "epochs": 40},
    {"arch": "ResNet-110", "dataset": "CIFAR-10", "ratio": 0.7, "epochs": 40},
    # CIFAR-100 更多配置
    {"arch": "ResNet-56", "dataset": "CIFAR-100", "ratio": 0.3, "epochs": 40},
    {"arch": "ResNet-56", "dataset": "CIFAR-100", "ratio": 0.7, "epochs": 40},
    {"arch": "VGG-16", "dataset": "CIFAR-100", "ratio": 0.5, "epochs": 40},
    # 再次核心配置确保稳定性
    {"arch": "ResNet-56", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
    {"arch": "VGG-16", "dataset": "CIFAR-10", "ratio": 0.5, "epochs": 40},
]


def run_experiment(exp):
    """运行单个实验"""
    import sys
    python_exe = sys.executable
    script_path = str(PROJECT_ROOT / "experiments" / "run_real_pruning.py")
    
    cmd = [
        python_exe, script_path,
        "--arch", exp["arch"],
        "--dataset", exp["dataset"],
        "--method", "GRA",
        "--ratio", str(exp["ratio"]),
        "--epochs", str(exp["epochs"])
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(PROJECT_ROOT))
        return result.returncode == 0
    except Exception as e:
        print(f"    Error: {e}")
        return False


def analyze_results():
    """分析当前结果"""
    if RESULTS_FILE.exists():
        df = pd.read_csv(RESULTS_FILE)
        
        # 今天的GRA结果
        today = datetime.now().strftime("%Y-%m-%d")
        recent = df[df['timestamp'].str.startswith(today) & (df['method'] == 'GRA')]
        
        if len(recent) > 0:
            print("\n--- 今日GRA结果 ---")
            for _, row in recent.iterrows():
                print(f"  {row['architecture']}/{row['dataset']}@{row['ratio']}: {row['final_acc']:.2f}%")
        
        # GRA vs L1 对比
        print("\n--- GRA vs L1 对比 ---")
        for arch in ['ResNet-20', 'ResNet-56', 'VGG-16', 'ResNet-110']:
            for dataset in ['CIFAR-10', 'CIFAR-100']:
                gra = df[(df['architecture']==arch) & (df['dataset']==dataset) & (df['method']=='GRA')]
                l1 = df[(df['architecture']==arch) & (df['dataset']==dataset) & (df['method']=='L1')]
                
                if len(gra) > 0 and len(l1) > 0:
                    gra_mean = gra['final_acc'].mean()
                    l1_mean = l1['final_acc'].mean()
                    diff = gra_mean - l1_mean
                    status = "WIN" if diff > 0 else ("TIE" if diff > -0.3 else "LOSS")
                    print(f"  {arch}/{dataset}: GRA={gra_mean:.2f} L1={l1_mean:.2f} ({diff:+.2f}) [{status}]")


def main():
    start_time = datetime.now()
    print("="*70)
    print(f"GRA-Fisher 10小时深度验证实验")
    print(f"开始时间: {start_time}")
    print(f"总实验数: {len(EXPERIMENTS)}")
    print("="*70)
    
    success_count = 0
    
    for i, exp in enumerate(EXPERIMENTS):
        exp_name = f"{exp['arch']}/{exp['dataset']}@{exp['ratio']}"
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp_name}")
        
        success = run_experiment(exp)
        
        if success:
            success_count += 1
            print(f"    OK")
        else:
            print(f"    FAILED")
        
        # 每5个实验分析一次
        if (i + 1) % 5 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            print(f"\n--- Progress: {i+1}/{len(EXPERIMENTS)}, Success: {success_count}, Time: {elapsed:.1f}h ---")
            analyze_results()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*70)
    print(f"Experiments Complete!")
    print(f"End time: {end_time}")
    print(f"Duration: {duration:.2f} hours")
    print(f"Success: {success_count}/{len(EXPERIMENTS)} ({100*success_count/len(EXPERIMENTS):.0f}%)")
    print("="*70)
    
    # 最终分析
    analyze_results()
    
    # 保存日志
    log = {
        'start': str(start_time),
        'end': str(end_time),
        'duration_hours': duration,
        'total': len(EXPERIMENTS),
        'success': success_count,
    }
    
    log_file = PROJECT_ROOT / "experiments" / "overnight_fisher_log.json"
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"\nLog saved: {log_file}")


if __name__ == "__main__":
    main()
