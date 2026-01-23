"""GRA-Gradient 多配置验证实验"""
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path

PROJECT_ROOT = Path(r"C:\GRA-CNN")
PYTHON = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
SCRIPT = str(PROJECT_ROOT / "experiments" / "run_real_pruning.py")

experiments = [
    # 核心配置
    ("ResNet-20", "CIFAR-10", 0.5),
    ("ResNet-56", "CIFAR-10", 0.3),
    ("ResNet-56", "CIFAR-10", 0.7),
    ("VGG-16", "CIFAR-10", 0.5),
    ("ResNet-56", "CIFAR-100", 0.5),
]

def run_exp(config):
    arch, dataset, ratio = config
    exp_id = f"{arch}_{dataset}_{ratio}".replace("-", "")
    print(f"开始: {exp_id}")
    
    cmd = [PYTHON, SCRIPT, "--arch", arch, "--dataset", dataset, 
           "--method", "GRA", "--ratio", str(ratio), "--epochs", "40"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            print(f"成功: {exp_id}")
            return {"id": exp_id, "status": "success"}
        else:
            print(f"失败: {exp_id}")
            return {"id": exp_id, "status": "failed"}
    except Exception as e:
        print(f"异常: {exp_id}")
        return {"id": exp_id, "status": "error"}

if __name__ == "__main__":
    print(f"GRA-Gradient 多配置验证 @ {datetime.now()}")
    print(f"实验数: {len(experiments)}, 并行: 2")
    
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_exp, e): e for e in experiments}
        for future in as_completed(futures):
            results.append(future.result())
    
    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n完成! 成功: {success}/{len(experiments)}")
