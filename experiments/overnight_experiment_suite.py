"""
===============================================================================
GRA-CNN 论文补充实验套件 - 10小时通宵版
===============================================================================
根据 Applied Intelligence 审稿意见设计的全面实验计划：

1. 消融实验 (Ablation Study)
   - 对比 GRA vs L1 vs FPGM vs HRank 的性能差异
   - 验证 GRA 的独特价值

2. 多种子统计显著性 (Multi-seed Statistical Significance)  
   - 每个核心配置运行 3 个不同种子
   - 计算均值和标准差，支持误差棒图

3. 超参数敏感性分析 (Rho Sensitivity Analysis)
   - 测试 rho = 0.1, 0.3, 0.5, 0.7, 0.9
   - 验证 GRA 对分辨率系数的鲁棒性

4. 效率指标采集 (Efficiency Metrics)
   - FLOPs 减少量
   - 参数量减少量
   - 推理延迟 (GPU/CPU)
===============================================================================
"""

import subprocess
import os
import sys
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PYTHON = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
SCRIPT = r"C:\GRA-CNN\experiments\run_real_pruning.py"
RESULT_FILE = r"C:\GRA-CNN\experiments\overnight_experiments.csv"

# ============================================================================
# 实验配置
# ============================================================================

EXPERIMENTS = []

# -----------------------------------------------------------------------------
# Part 1: 消融实验 - 核心架构上比较所有方法
# -----------------------------------------------------------------------------
ABLATION_CONFIGS = [
    ("ResNet-56", "CIFAR-10"),
    ("ResNet-56", "CIFAR-100"),
    ("VGG-16", "CIFAR-10"),
]
METHODS = ["GRA", "L1", "FPGM", "HRank"]
RATIOS = [0.3, 0.5, 0.7]

for arch, ds in ABLATION_CONFIGS:
    for method in METHODS:
        for ratio in RATIOS:
            EXPERIMENTS.append({
                "arch": arch,
                "dataset": ds,
                "method": method,
                "ratio": ratio,
                "epochs": 40,
                "type": "ablation"
            })

# -----------------------------------------------------------------------------
# Part 2: 多种子统计显著性 - 核心配置多次运行
# -----------------------------------------------------------------------------
MULTISEED_CONFIGS = [
    ("ResNet-56", "CIFAR-10", "GRA", 0.5),
    ("ResNet-56", "CIFAR-10", "L1", 0.5),
    ("ResNet-56", "CIFAR-100", "GRA", 0.5),
    ("VGG-16", "CIFAR-10", "GRA", 0.5),
]
SEEDS = [42, 123, 456]

for arch, ds, method, ratio in MULTISEED_CONFIGS:
    for seed in SEEDS:
        EXPERIMENTS.append({
            "arch": arch,
            "dataset": ds,
            "method": method,
            "ratio": ratio,
            "epochs": 40,
            "seed": seed,
            "type": "multiseed"
        })

# -----------------------------------------------------------------------------
# Part 3: Rho 敏感性分析
# -----------------------------------------------------------------------------
RHO_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
RHO_CONFIGS = [
    ("ResNet-56", "CIFAR-10"),
    ("ResNet-56", "CIFAR-100"),
]

for arch, ds in RHO_CONFIGS:
    for rho in RHO_VALUES:
        EXPERIMENTS.append({
            "arch": arch,
            "dataset": ds,
            "method": "GRA",
            "ratio": 0.5,
            "rho": rho,
            "epochs": 40,
            "type": "rho_sensitivity"
        })

# -----------------------------------------------------------------------------
# Part 4: 完善其他架构覆盖
# -----------------------------------------------------------------------------
ADDITIONAL_ARCHS = [
    ("ResNet-20", "CIFAR-10", 0.5),
    ("ResNet-32", "CIFAR-10", 0.5),
    ("ResNet-44", "CIFAR-10", 0.5),
    ("ResNet-20", "CIFAR-100", 0.5),
    ("ResNet-32", "CIFAR-100", 0.5),
    ("ResNet-44", "CIFAR-100", 0.5),
]

for arch, ds, ratio in ADDITIONAL_ARCHS:
    for method in METHODS:
        EXPERIMENTS.append({
            "arch": arch,
            "dataset": ds,
            "method": method,
            "ratio": ratio,
            "epochs": 40,
            "type": "coverage"
        })


def run_experiment(exp):
    """Run a single experiment and return results."""
    cmd = [
        PYTHON, SCRIPT,
        "--arch", exp["arch"],
        "--dataset", exp["dataset"],
        "--method", exp["method"],
        "--ratio", str(exp["ratio"]),
        "--epochs", str(exp["epochs"])
    ]
    
    if "rho" in exp:
        cmd.extend(["--rho", str(exp["rho"])])
    if "seed" in exp:
        cmd.extend(["--seed", str(exp["seed"])])
    
    start_time = time.time()
    result = {
        "experiment": exp,
        "status": "pending",
        "stdout": "",
        "stderr": "",
        "elapsed": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per experiment
            cwd=r"C:\GRA-CNN"
        )
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["status"] = "success" if proc.returncode == 0 else "failed"
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    result["elapsed"] = time.time() - start_time
    return result


def main():
    print("=" * 70)
    print("GRA-CNN 通宵实验套件")
    print(f"开始时间: {datetime.now()}")
    print(f"总实验数: {len(EXPERIMENTS)}")
    print("=" * 70)
    
    # Log experiment plan
    with open(r"C:\GRA-CNN\experiments\overnight_plan.json", "w") as f:
        json.dump(EXPERIMENTS, f, indent=2)
    
    results = []
    completed = 0
    
    # Run experiments sequentially (safer for GPU memory)
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] {exp['type']}: {exp['arch']}/{exp['dataset']}/{exp['method']}@{exp['ratio']}")
        
        result = run_experiment(exp)
        results.append(result)
        completed += 1
        
        # Save progress periodically
        if completed % 5 == 0:
            with open(r"C:\GRA-CNN\experiments\overnight_progress.json", "w") as f:
                json.dump({
                    "completed": completed,
                    "total": len(EXPERIMENTS),
                    "results": results
                }, f, indent=2)
        
        print(f"   Status: {result['status']}, Elapsed: {result['elapsed']:.1f}s")
    
    # Final save
    with open(r"C:\GRA-CNN\experiments\overnight_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"实验完成! 总计: {completed}/{len(EXPERIMENTS)}")
    print(f"结束时间: {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
