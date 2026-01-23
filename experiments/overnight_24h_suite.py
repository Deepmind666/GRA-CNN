"""
GRA-Fisher 24小时全覆盖验证实验
================================

环境: gra311 (Python 3.11 + PyTorch cu121 + RTX 5090)
算法: GRA-Fisher (Fisher 50% + GRA 30% + L1 20%)

实验矩阵:
- 架构: ResNet-20, ResNet-56, ResNet-110, VGG-16
- 数据集: CIFAR-10, CIFAR-100
- 剪枝率: 0.3, 0.4, 0.5, 0.6, 0.7
- 对照方法: L1, FPGM
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"C:\GRA-CNN")
RESULTS_FILE = PROJECT_ROOT / "experiments" / "supplementary_results.csv"
LOG_FILE = PROJECT_ROOT / "experiments" / "24hour_experiment_log.json"

# 完整实验矩阵
EXPERIMENTS = []

# GRA-Fisher 全覆盖
ARCHS = ["ResNet-20", "ResNet-56", "ResNet-110", "VGG-16"]
DATASETS = ["CIFAR-10", "CIFAR-100"]
RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]
METHODS = ["GRA", "L1", "FPGM"]

for arch in ARCHS:
    for dataset in DATASETS:
        for ratio in RATIOS:
            # GRA-Fisher 每个配置跑2次确保稳定性
            EXPERIMENTS.append({"arch": arch, "dataset": dataset, "ratio": ratio, "method": "GRA", "epochs": 40})
            EXPERIMENTS.append({"arch": arch, "dataset": dataset, "ratio": ratio, "method": "GRA", "epochs": 40})
            
            # L1 和 FPGM 各 1 次用于对比
            EXPERIMENTS.append({"arch": arch, "dataset": dataset, "ratio": ratio, "method": "L1", "epochs": 40})
            EXPERIMENTS.append({"arch": arch, "dataset": dataset, "ratio": ratio, "method": "FPGM", "epochs": 40})

print(f"Total experiments: {len(EXPERIMENTS)}")


def run_experiment(exp):
    """运行单个实验"""
    script = str(PROJECT_ROOT / "experiments" / "run_real_pruning.py")
    cmd = [
        "python", script,
        "--arch", exp["arch"],
        "--dataset", exp["dataset"],
        "--method", exp["method"],
        "--ratio", str(exp["ratio"]),
        "--epochs", str(exp["epochs"])
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, 
            timeout=5400, cwd=str(PROJECT_ROOT)  # 90分钟超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"    Timeout!")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    start_time = datetime.now()
    print("="*70)
    print("GRA-Fisher 24小时全覆盖验证实验")
    print(f"开始时间: {start_time}")
    print(f"环境: gra311 (PyTorch cu121 + RTX 5090)")
    print(f"总实验数: {len(EXPERIMENTS)}")
    print("="*70)
    
    results = []
    success_count = 0
    
    for i, exp in enumerate(EXPERIMENTS):
        exp_name = f"{exp['method']}/{exp['arch']}/{exp['dataset']}@{exp['ratio']}"
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp_name}")
        
        exp_start = time.time()
        success = run_experiment(exp)
        exp_time = time.time() - exp_start
        
        exp['success'] = success
        exp['time_seconds'] = exp_time
        results.append(exp)
        
        if success:
            success_count += 1
            print(f"    OK ({exp_time/60:.1f} min)")
        else:
            print(f"    FAILED ({exp_time/60:.1f} min)")
        
        # 每10个实验显示进度
        if (i + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            remaining = len(EXPERIMENTS) - (i + 1)
            avg_time = elapsed / (i + 1) * 60  # minutes per exp
            eta = remaining * avg_time / 60  # hours
            print(f"\n--- Progress: {i+1}/{len(EXPERIMENTS)} | Success: {success_count} | Elapsed: {elapsed:.1f}h | ETA: {eta:.1f}h ---")
        
        # 定期保存日志
        if (i + 1) % 20 == 0:
            with open(LOG_FILE, 'w') as f:
                json.dump({
                    'start': str(start_time),
                    'current': str(datetime.now()),
                    'progress': f"{i+1}/{len(EXPERIMENTS)}",
                    'success': success_count,
                    'results': results
                }, f, indent=2)
    
    # 最终日志
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*70)
    print("实验完成!")
    print(f"结束时间: {end_time}")
    print(f"总耗时: {duration:.2f} 小时")
    print(f"成功率: {success_count}/{len(EXPERIMENTS)} ({100*success_count/len(EXPERIMENTS):.0f}%)")
    print("="*70)
    
    # 保存最终日志
    with open(LOG_FILE, 'w') as f:
        json.dump({
            'start': str(start_time),
            'end': str(end_time),
            'duration_hours': duration,
            'total': len(EXPERIMENTS),
            'success': success_count,
            'results': results
        }, f, indent=2)
    
    print(f"\nLog saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
