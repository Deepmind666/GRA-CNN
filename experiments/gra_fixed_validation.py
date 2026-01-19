"""
GRA 修复后完整验证实验
======================
修复内容：
1. Min-Max 标准化（而非 Z-score）
2. 正确类别 logit 作为参考序列（而非 max logit）
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(r"C:\GRA-CNN")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "gra_fixed_validation"
LOG_FILE = RESULTS_DIR / "log.json"
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
SCRIPT = str(PROJECT_ROOT / "experiments" / "run_real_pruning.py")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    try:
        data = []
        if LOG_FILE.exists():
            data = json.load(open(LOG_FILE))
        data.append({"time": datetime.now().isoformat(), "msg": msg})
        json.dump(data, open(LOG_FILE, 'w'), indent=2)
    except: pass

def get_experiments():
    """关键对比实验"""
    return [
        # GRA vs L1 vs FPGM 完整对比 (40 epochs)
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
        
        # 额外架构
        ("CIFAR-10", "ResNet-20", "GRA", 0.5, 40),
        ("CIFAR-10", "ResNet-20", "L1", 0.5, 40),
        ("CIFAR-10", "VGG-16", "GRA", 0.5, 40),
        ("CIFAR-10", "VGG-16", "L1", 0.5, 40),
    ]

def run_exp(config):
    dataset, arch, method, ratio, epochs = config
    exp_id = f"{dataset}_{arch}_{method}_{ratio}".replace("-", "").lower()
    
    log(f"开始: {exp_id}")
    cmd = [PYTHON_EXE, SCRIPT, "--arch", arch, "--dataset", dataset, 
           "--method", method, "--ratio", str(ratio), "--epochs", str(epochs)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            log(f"成功: {exp_id}")
            return {"id": exp_id, "status": "success"}
        else:
            log(f"失败: {exp_id}")
            return {"id": exp_id, "status": "failed"}
    except Exception as e:
        log(f"异常: {exp_id}")
        return {"id": exp_id, "status": "error"}

def main():
    start = datetime.now()
    log(f"GRA修复验证实验启动 @ {start.strftime('%H:%M:%S')}")
    
    experiments = get_experiments()
    log(f"实验数: {len(experiments)}, 并行: 4")
    
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_exp, e): e for e in experiments}
        for future in as_completed(futures):
            results.append(future.result())
            success = sum(1 for r in results if r["status"] == "success")
            log(f"进度: {len(results)}/{len(experiments)}, 成功: {success}")
    
    elapsed = (datetime.now() - start).total_seconds() / 3600
    success = sum(1 for r in results if r["status"] == "success")
    
    summary = {"experiments": len(experiments), "success": success, 
               "rate": f"{100*success/len(experiments):.1f}%", "hours": round(elapsed, 2)}
    json.dump(summary, open(RESULTS_DIR / "summary.json", 'w'), indent=2)
    
    log(f"完成! 成功率: {summary['rate']}, 耗时: {elapsed:.2f}h")
    print(f"\n{'='*50}\nGRA修复验证完成: {success}/{len(experiments)}\n{'='*50}")

if __name__ == "__main__":
    main()
