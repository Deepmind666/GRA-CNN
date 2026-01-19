"""
GRA-CNN 大规模并行实验套件 (充分利用 RTX 5090 32GB)
====================================================
同时运行多个实验，最大化 GPU 利用率

RTX 5090 规格: 32GB VRAM, 充足算力
策略: 同时运行 4 个实验进程

科学依据:
1. 完整方法对比 (GRA/L1/FPGM/HRank)
2. 多剪枝率覆盖 (0.3, 0.4, 0.5, 0.6, 0.7)
3. 多架构验证 (ResNet-20/56/110, VGG-16)
4. 双数据集 (CIFAR-10, CIFAR-100)
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

PROJECT_ROOT = Path(r"C:\GRA-CNN")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "parallel_massive"
LOG_FILE = RESULTS_DIR / "parallel_log.json"
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
PRUNING_SCRIPT = str(EXPERIMENTS_DIR / "run_real_pruning.py")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 使用进程安全的日志
log_lock = multiprocessing.Lock()

def log_event(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")
    
    try:
        with log_lock:
            log_data = []
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r') as f:
                    log_data = json.load(f)
            log_data.append({"time": datetime.now().isoformat(), "msg": msg})
            with open(LOG_FILE, 'w') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
    except:
        pass

def get_massive_experiment_list():
    """生成大规模实验列表 - 全面覆盖"""
    experiments = []
    
    # 完整实验矩阵
    configs = [
        # CIFAR-10 ResNet 全覆盖
        ("CIFAR-10", "ResNet-20", "GRA", 0.3, 30),
        ("CIFAR-10", "ResNet-20", "GRA", 0.5, 30),
        ("CIFAR-10", "ResNet-20", "GRA", 0.7, 30),
        ("CIFAR-10", "ResNet-20", "L1", 0.3, 30),
        ("CIFAR-10", "ResNet-20", "L1", 0.5, 30),
        ("CIFAR-10", "ResNet-20", "L1", 0.7, 30),
        ("CIFAR-10", "ResNet-20", "FPGM", 0.5, 30),
        
        ("CIFAR-10", "ResNet-56", "GRA", 0.3, 30),
        ("CIFAR-10", "ResNet-56", "GRA", 0.4, 30),
        ("CIFAR-10", "ResNet-56", "GRA", 0.5, 30),
        ("CIFAR-10", "ResNet-56", "GRA", 0.6, 30),
        ("CIFAR-10", "ResNet-56", "GRA", 0.7, 30),
        ("CIFAR-10", "ResNet-56", "L1", 0.3, 30),
        ("CIFAR-10", "ResNet-56", "L1", 0.4, 30),
        ("CIFAR-10", "ResNet-56", "L1", 0.5, 30),
        ("CIFAR-10", "ResNet-56", "L1", 0.6, 30),
        ("CIFAR-10", "ResNet-56", "L1", 0.7, 30),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.3, 30),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.5, 30),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.7, 30),
        
        ("CIFAR-10", "ResNet-110", "GRA", 0.5, 30),
        ("CIFAR-10", "ResNet-110", "L1", 0.5, 30),
        ("CIFAR-10", "ResNet-110", "FPGM", 0.5, 30),
        
        # VGG-16
        ("CIFAR-10", "VGG-16", "GRA", 0.3, 30),
        ("CIFAR-10", "VGG-16", "GRA", 0.5, 30),
        ("CIFAR-10", "VGG-16", "GRA", 0.7, 30),
        ("CIFAR-10", "VGG-16", "L1", 0.5, 30),
        ("CIFAR-10", "VGG-16", "FPGM", 0.5, 30),
        
        # CIFAR-100
        ("CIFAR-100", "ResNet-20", "GRA", 0.5, 30),
        ("CIFAR-100", "ResNet-20", "L1", 0.5, 30),
        ("CIFAR-100", "ResNet-56", "GRA", 0.3, 30),
        ("CIFAR-100", "ResNet-56", "GRA", 0.5, 30),
        ("CIFAR-100", "ResNet-56", "GRA", 0.7, 30),
        ("CIFAR-100", "ResNet-56", "L1", 0.3, 30),
        ("CIFAR-100", "ResNet-56", "L1", 0.5, 30),
        ("CIFAR-100", "ResNet-56", "L1", 0.7, 30),
        ("CIFAR-100", "ResNet-56", "FPGM", 0.5, 30),
        ("CIFAR-100", "ResNet-110", "GRA", 0.5, 30),
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

def run_single_experiment(exp):
    """运行单个实验 (在独立进程中)"""
    exp_id = exp["id"]
    start = time.time()
    
    log_event(f"启动: {exp_id}")
    
    cmd = [
        PYTHON_EXE, PRUNING_SCRIPT,
        "--arch", exp["arch"],
        "--dataset", exp["dataset"],
        "--method", exp["method"],
        "--ratio", str(exp["ratio"]),
        "--epochs", str(exp["epochs"])
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=3600, cwd=str(PROJECT_ROOT)
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log_event(f"完成: {exp_id} ({elapsed/60:.1f}min)")
            return {"id": exp_id, "status": "success", "time": elapsed}
        else:
            log_event(f"失败: {exp_id}")
            return {"id": exp_id, "status": "failed", "error": result.stderr[:200] if result.stderr else ""}
    except Exception as e:
        log_event(f"异常: {exp_id} - {str(e)[:50]}")
        return {"id": exp_id, "status": "error", "error": str(e)}

def main():
    start_time = datetime.now()
    log_event(f"大规模并行实验启动 @ {start_time.strftime('%H:%M:%S')}")
    log_event(f"并行进程数: 4 (利用 RTX 5090 32GB)")
    
    experiments = get_massive_experiment_list()
    log_event(f"实验总数: {len(experiments)}")
    
    results = []
    
    # 使用 4 个并行进程
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            success = sum(1 for r in results if r["status"] == "success")
            log_event(f"进度: {completed}/{len(experiments)}, 成功: {success}")
    
    # 保存结果
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 3600
    
    success_count = sum(1 for r in results if r["status"] == "success")
    summary = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "elapsed_hours": round(elapsed, 2),
        "total": len(experiments),
        "success": success_count,
        "rate": f"{100*success_count/len(experiments):.1f}%",
        "parallel_workers": 4
    }
    
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    log_event(f"全部完成! 成功率: {summary['rate']}, 耗时: {elapsed:.2f}h")
    
    print("\n" + "="*60)
    print(f"大规模并行实验完成!")
    print(f"实验数: {len(experiments)}, 成功: {success_count}")
    print(f"耗时: {elapsed:.2f} 小时 (4 进程并行)")
    print("="*60)

if __name__ == "__main__":
    main()
