"""
GRA-CNN 10小时通宵全覆盖实验套件
================================
目标: 补充所有图表缺失的数据点
横轴: 0.3, 0.4, 0.5, 0.6, 0.7 (5个剪枝率全覆盖)

实验矩阵:
- 架构: ResNet-20, ResNet-56, ResNet-110, VGG-16
- 数据集: CIFAR-10, CIFAR-100
- 方法: GRA, L1, FPGM
- 剪枝率: 0.3, 0.4, 0.5, 0.6, 0.7

RTX 5090: 32GB VRAM, 可并行4个实验
预计时间: 每个实验约15-20分钟
总实验数: ~120个 (4架构 × 2数据集 × 3方法 × 5剪枝率)
预计耗时: 120 × 15分钟 / 4并行 ≈ 7.5小时
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(r"C:\GRA-CNN")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "overnight_full_coverage"
LOG_FILE = RESULTS_DIR / "log.json"
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"
SCRIPT = str(PROJECT_ROOT / "experiments" / "run_real_pruning.py")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

experiment_log = []

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")
    experiment_log.append({"time": datetime.now().isoformat(), "msg": msg})
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(experiment_log, f, indent=2, ensure_ascii=False)
    except:
        pass

def get_full_experiment_matrix():
    """生成完整实验矩阵 - 5个剪枝率全覆盖"""
    experiments = []
    
    # 配置矩阵
    architectures = ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']
    datasets = ['CIFAR-10', 'CIFAR-100']
    methods = ['GRA', 'L1', 'FPGM']
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]  # 5个剪枝率全覆盖!
    
    for arch in architectures:
        for dataset in datasets:
            for method in methods:
                for ratio in ratios:
                    exp_id = f"{arch}_{dataset}_{method}_{ratio}".replace("-", "").replace(" ", "_").lower()
                    experiments.append({
                        "id": exp_id,
                        "arch": arch,
                        "dataset": dataset,
                        "method": method,
                        "ratio": ratio,
                        "epochs": 40  # 标准微调
                    })
    
    return experiments

def run_experiment(exp):
    """运行单个实验"""
    exp_id = exp["id"]
    start = time.time()
    
    log(f"开始: {exp_id}")
    
    cmd = [
        PYTHON_EXE, SCRIPT,
        "--arch", exp["arch"],
        "--dataset", exp["dataset"],
        "--method", exp["method"],
        "--ratio", str(exp["ratio"]),
        "--epochs", str(exp["epochs"])
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=5400,  # 90分钟超时
            cwd=str(PROJECT_ROOT)
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            log(f"成功: {exp_id} ({elapsed/60:.1f}min)")
            return {"id": exp_id, "status": "success", "time": elapsed}
        else:
            log(f"失败: {exp_id}")
            return {"id": exp_id, "status": "failed", "error": result.stderr[:200] if result.stderr else ""}
    except subprocess.TimeoutExpired:
        log(f"超时: {exp_id}")
        return {"id": exp_id, "status": "timeout"}
    except Exception as e:
        log(f"异常: {exp_id} - {str(e)[:50]}")
        return {"id": exp_id, "status": "error", "error": str(e)}

def main():
    start_time = datetime.now()
    log("="*60)
    log(f"10小时通宵全覆盖实验启动 @ {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*60)
    log("目标: 补充所有图表缺失的数据点 (5个剪枝率全覆盖)")
    log("GPU: RTX 5090 (32GB VRAM)")
    log("并行数: 4")
    
    experiments = get_full_experiment_matrix()
    log(f"实验总数: {len(experiments)}")
    log(f"预计耗时: {len(experiments) * 15 / 4 / 60:.1f} 小时")
    
    results = []
    
    # 4个并行进程
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_experiment, e): e for e in experiments}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            success = sum(1 for r in results if r["status"] == "success")
            failed = sum(1 for r in results if r["status"] != "success")
            
            elapsed = (datetime.now() - start_time).total_seconds() / 3600
            remaining = (len(experiments) - len(results)) * (elapsed / len(results)) if len(results) > 0 else 0
            
            log(f"进度: {len(results)}/{len(experiments)} | 成功: {success} | 失败: {failed} | 已用: {elapsed:.1f}h | 剩余: {remaining:.1f}h")
    
    # 最终摘要
    end_time = datetime.now()
    elapsed_hours = (end_time - start_time).total_seconds() / 3600
    success_count = sum(1 for r in results if r["status"] == "success")
    
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_hours": round(elapsed_hours, 2),
        "total_experiments": len(experiments),
        "successful": success_count,
        "failed": len(experiments) - success_count,
        "success_rate": f"{100*success_count/len(experiments):.1f}%",
        "parallel_workers": 4
    }
    
    with open(RESULTS_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    with open(RESULTS_DIR / "all_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log("="*60)
    log(f"通宵实验完成!")
    log(f"成功: {success_count}/{len(experiments)} ({summary['success_rate']})")
    log(f"耗时: {elapsed_hours:.2f} 小时")
    log("="*60)
    
    print("\n" + "="*60)
    print("🌙 通宵实验完成!")
    print(f"✅ 成功: {success_count}/{len(experiments)}")
    print(f"⏱️ 耗时: {elapsed_hours:.2f} 小时")
    print("="*60)

if __name__ == "__main__":
    main()
