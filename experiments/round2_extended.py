"""
GRA-CNN 第二轮大规模实验 (充分利用剩余GPU时间)
==============================================
第一轮: 38实验/34成功/4.8小时
第二轮: 更多配置 + 更多epochs + CIFAR-100深度覆盖
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(r"C:\GRA-CNN")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "round2_extended"
LOG_FILE = RESULTS_DIR / "log.json"
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
PRUNING_SCRIPT = str(EXPERIMENTS_DIR / "run_real_pruning.py")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    try:
        data = []
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                data = json.load(f)
        data.append({"time": datetime.now().isoformat(), "msg": msg})
        with open(LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass

def get_round2_experiments():
    """第二轮实验 - 更深层次覆盖"""
    experiments = []
    
    configs = [
        # 高epochs训练 (验证长期收敛)
        ("CIFAR-10", "ResNet-56", "GRA", 0.5, 60),
        ("CIFAR-10", "ResNet-56", "L1", 0.5, 60),
        ("CIFAR-10", "ResNet-56", "FPGM", 0.5, 60),
        
        # 极端剪枝率测试 (0.8)
        ("CIFAR-10", "ResNet-56", "GRA", 0.8, 40),
        ("CIFAR-10", "ResNet-56", "L1", 0.8, 40),
        
        # CIFAR-100 完整覆盖
        ("CIFAR-100", "ResNet-20", "GRA", 0.3, 40),
        ("CIFAR-100", "ResNet-20", "GRA", 0.7, 40),
        ("CIFAR-100", "ResNet-20", "L1", 0.3, 40),
        ("CIFAR-100", "ResNet-20", "L1", 0.7, 40),
        ("CIFAR-100", "ResNet-56", "GRA", 0.4, 40),
        ("CIFAR-100", "ResNet-56", "GRA", 0.6, 40),
        ("CIFAR-100", "ResNet-56", "L1", 0.4, 40),
        ("CIFAR-100", "ResNet-56", "L1", 0.6, 40),
        ("CIFAR-100", "ResNet-110", "GRA", 0.3, 40),
        ("CIFAR-100", "ResNet-110", "GRA", 0.7, 40),
        ("CIFAR-100", "ResNet-110", "L1", 0.5, 40),
        
        # VGG-16 完整覆盖
        ("CIFAR-10", "VGG-16", "GRA", 0.4, 40),
        ("CIFAR-10", "VGG-16", "GRA", 0.6, 40),
        ("CIFAR-10", "VGG-16", "L1", 0.3, 40),
        ("CIFAR-10", "VGG-16", "L1", 0.7, 40),
        ("CIFAR-10", "VGG-16", "FPGM", 0.3, 40),
        ("CIFAR-10", "VGG-16", "FPGM", 0.7, 40),
        
        # HRank 方法补充
        ("CIFAR-10", "ResNet-56", "HRank", 0.3, 40),
        ("CIFAR-10", "ResNet-56", "HRank", 0.5, 40),
        ("CIFAR-10", "ResNet-56", "HRank", 0.7, 40),
        
        # 更多 ResNet-110 (深网验证)
        ("CIFAR-10", "ResNet-110", "GRA", 0.3, 40),
        ("CIFAR-10", "ResNet-110", "GRA", 0.7, 40),
        ("CIFAR-10", "ResNet-110", "L1", 0.3, 40),
        ("CIFAR-10", "ResNet-110", "L1", 0.7, 40),
    ]
    
    for dataset, arch, method, ratio, epochs in configs:
        exp_id = f"r2_{dataset}_{arch}_{method}_{ratio}".replace("-", "").lower()
        experiments.append({
            "id": exp_id,
            "dataset": dataset,
            "arch": arch,
            "method": method,
            "ratio": ratio,
            "epochs": epochs
        })
    
    return experiments

def run_exp(exp):
    exp_id = exp["id"]
    start = time.time()
    log(f"开始: {exp_id}")
    
    cmd = [
        PYTHON_EXE, PRUNING_SCRIPT,
        "--arch", exp["arch"],
        "--dataset", exp["dataset"],
        "--method", exp["method"],
        "--ratio", str(exp["ratio"]),
        "--epochs", str(exp["epochs"])
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log(f"成功: {exp_id} ({elapsed/60:.1f}min)")
            return {"id": exp_id, "status": "success", "time": elapsed}
        else:
            log(f"失败: {exp_id}")
            return {"id": exp_id, "status": "failed"}
    except Exception as e:
        log(f"异常: {exp_id}")
        return {"id": exp_id, "status": "error"}

def main():
    start = datetime.now()
    log(f"第二轮实验启动 @ {start.strftime('%H:%M:%S')}")
    
    experiments = get_round2_experiments()
    log(f"实验总数: {len(experiments)}, 并行数: 4")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_exp, e): e for e in experiments}
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            success = sum(1 for r in results if r["status"] == "success")
            log(f"进度: {i+1}/{len(experiments)}, 成功: {success}")
    
    end = datetime.now()
    elapsed = (end - start).total_seconds() / 3600
    success = sum(1 for r in results if r["status"] == "success")
    
    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "hours": round(elapsed, 2),
        "total": len(experiments),
        "success": success,
        "rate": f"{100*success/len(experiments):.1f}%"
    }
    
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"完成! 成功率: {summary['rate']}, 耗时: {elapsed:.2f}h")
    print(f"\n{'='*50}\n第二轮完成: {success}/{len(experiments)} 成功\n{'='*50}")

if __name__ == "__main__":
    main()
