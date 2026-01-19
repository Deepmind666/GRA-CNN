"""
GRA-CNN 通宵实验套件 (修复版)
==============================
直接调用 run_real_pruning.py，使用正确的参数格式
"""

import os
import sys
import json
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path

# 配置
PROJECT_ROOT = Path(r"C:\GRA-CNN")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "overnight_enhanced_v2"
LOG_FILE = RESULTS_DIR / "overnight_log.json"
ERROR_FILE = RESULTS_DIR / "overnight_errors.json"
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
PRUNING_SCRIPT = str(EXPERIMENTS_DIR / "run_real_pruning.py")

# 创建结果目录
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 日志
experiment_log = []
error_log = []

def log_event(event_type, message, data=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
        "data": data
    }
    experiment_log.append(entry)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{event_type}] {message}")
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)

def log_error(exp_id, error_msg):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": exp_id,
        "error": error_msg[:500]
    }
    error_log.append(entry)
    print(f"[ERROR] {exp_id}: {error_msg[:100]}")
    
    with open(ERROR_FILE, 'w', encoding='utf-8') as f:
        json.dump(error_log, f, indent=2, ensure_ascii=False)

def get_experiments():
    """定义实验列表 (使用正确的参数格式)"""
    experiments = []
    
    # 核心实验配置
    configs = [
        # (dataset, arch, method, ratio, epochs)
        # Figure 3 核心数据
        ("CIFAR-10", "ResNet-56", "GRA", 0.3, 40),
        ("CIFAR-10", "ResNet-56", "GRA", 0.5, 40),
        ("CIFAR-10", "ResNet-56", "GRA", 0.7, 40),
        ("CIFAR-10", "ResNet-56", "L1", 0.3, 40),
        ("CIFAR-10", "ResNet-56", "L1", 0.5, 40),
        ("CIFAR-10", "ResNet-56", "L1", 0.7, 40),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.3, 40),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.5, 40),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.7, 40),
        
        # CIFAR-100
        ("CIFAR-100", "ResNet-56", "GRA", 0.5, 40),
        ("CIFAR-100", "ResNet-56", "L1", 0.5, 40),
        ("CIFAR-100", "ResNet-56", "FPGM", 0.5, 40),
        
        # ResNet-20
        ("CIFAR-10", "ResNet-20", "GRA", 0.5, 40),
        ("CIFAR-10", "ResNet-20", "L1", 0.5, 40),
        
        # VGG-16
        ("CIFAR-10", "VGG-16", "GRA", 0.5, 40),
        ("CIFAR-10", "VGG-16", "L1", 0.5, 40),
    ]
    
    for dataset, arch, method, ratio, epochs in configs:
        exp_id = f"{dataset}_{arch}_{method}_{ratio}".replace("-", "").replace(" ", "_").lower()
        experiments.append({
            "id": exp_id,
            "dataset": dataset,
            "arch": arch,
            "method": method,
            "ratio": ratio,
            "epochs": epochs
        })
    
    return experiments

def run_experiment(exp):
    """运行单个实验"""
    exp_id = exp["id"]
    start_time = time.time()
    
    try:
        log_event("START", f"开始: {exp_id}")
        
        cmd = [
            PYTHON_EXE,
            PRUNING_SCRIPT,
            "--arch", exp["arch"],
            "--dataset", exp["dataset"],
            "--method", exp["method"],
            "--ratio", str(exp["ratio"]),
            "--epochs", str(exp["epochs"])
        ]
        
        print(f"  命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2小时超时
            cwd=str(PROJECT_ROOT)
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log_event("SUCCESS", f"完成: {exp_id} ({elapsed/60:.1f}min)")
            return True
        else:
            log_error(exp_id, result.stderr if result.stderr else f"Exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        log_error(exp_id, "Timeout after 2 hours")
        return False
    except Exception as e:
        log_error(exp_id, str(e))
        return False

def main():
    start_time = datetime.now()
    log_event("INIT", f"通宵实验启动 @ {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    experiments = get_experiments()
    log_event("INFO", f"共 {len(experiments)} 个实验待运行")
    
    success_count = 0
    fail_count = 0
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"实验 {i+1}/{len(experiments)}")
        print(f"{'='*60}")
        
        if run_experiment(exp):
            success_count += 1
        else:
            fail_count += 1
        
        # 进度报告
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        log_event("PROGRESS", f"进度: {i+1}/{len(experiments)}, 成功: {success_count}, 失败: {fail_count}, 耗时: {elapsed:.2f}h")
    
    # 最终报告
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 3600
    
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_hours": round(elapsed, 2),
        "total": len(experiments),
        "success": success_count,
        "failed": fail_count,
        "success_rate": f"{100*success_count/len(experiments):.1f}%"
    }
    
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    log_event("COMPLETE", f"全部完成! 成功率: {summary['success_rate']}")
    
    print("\n" + "="*60)
    print("通宵实验完成!")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"耗时: {elapsed:.2f} 小时")
    print("="*60)

if __name__ == "__main__":
    main()
