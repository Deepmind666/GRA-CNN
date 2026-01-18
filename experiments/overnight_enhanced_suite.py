"""
GRA-CNN 10小时通宵实验套件 (RTX 5090 全功率)
=============================================
目标: 极大丰富 Figure 3/4/5 的实验数据支撑

实验矩阵:
1. Figure 3 增强: 多架构×多剪枝率的 FLOPs/吞吐量/精度 完整测量
2. Figure 4 增强: 不同方法的完整训练曲线 (epoch-by-epoch)
3. Figure 5 增强: 更细粒度的 ρ 参数扫描 + 跨数据集验证

执行策略: 并行 + 队列 + 异常记录
"""

import os
import sys
import json
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
import concurrent.futures

# 配置
PROJECT_ROOT = Path(r"C:\GRA-CNN")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "overnight_enhanced"
LOG_FILE = RESULTS_DIR / "overnight_log.json"
ERROR_FILE = RESULTS_DIR / "overnight_errors.json"
PYTHON_EXE = r"C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe"

# 创建结果目录
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 全局日志
experiment_log = []
error_log = []

def log_event(event_type, message, data=None):
    """记录事件到日志"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
        "data": data
    }
    experiment_log.append(entry)
    print(f"[{event_type}] {message}")
    
    # 实时保存日志
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=2, ensure_ascii=False)

def log_error(experiment_id, error_msg, traceback_str=None):
    """记录错误"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "error": error_msg,
        "traceback": traceback_str
    }
    error_log.append(entry)
    print(f"[ERROR] {experiment_id}: {error_msg}")
    
    with open(ERROR_FILE, 'w', encoding='utf-8') as f:
        json.dump(error_log, f, indent=2, ensure_ascii=False)

# ============================================================
# 实验定义
# ============================================================

def get_experiment_matrix():
    """定义完整实验矩阵"""
    
    experiments = []
    
    # === Figure 3 增强: FLOPs/吞吐量/精度 ===
    # 更多架构 × 更多剪枝率
    fig3_configs = [
        # (dataset, arch, method, ratios)
        ("CIFAR-10", "ResNet-20", ["GRA", "L1", "FPGM", "HRank"], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ("CIFAR-10", "ResNet-56", ["GRA", "L1", "FPGM", "HRank"], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ("CIFAR-10", "ResNet-110", ["GRA", "L1", "FPGM"], [0.3, 0.5, 0.7]),
        ("CIFAR-10", "VGG-16", ["GRA", "L1", "FPGM"], [0.3, 0.5, 0.7]),
        ("CIFAR-100", "ResNet-56", ["GRA", "L1", "FPGM", "HRank"], [0.3, 0.5, 0.7]),
        ("CIFAR-100", "ResNet-110", ["GRA", "L1", "FPGM"], [0.5, 0.7]),
    ]
    
    for dataset, arch, methods, ratios in fig3_configs:
        for method in methods:
            for ratio in ratios:
                exp_id = f"fig3_{dataset}_{arch}_{method}_{ratio}".replace("-", "").lower()
                experiments.append({
                    "id": exp_id,
                    "type": "fig3_flops",
                    "dataset": dataset,
                    "arch": arch,
                    "method": method,
                    "ratio": ratio,
                    "epochs": 40,
                    "priority": 1
                })
    
    # === Figure 4 增强: 收敛曲线 ===
    # 关键配置的完整训练曲线
    fig4_configs = [
        ("CIFAR-10", "ResNet-56", ["GRA", "L1", "FPGM", "HRank"], [0.5, 0.7]),
        ("CIFAR-100", "ResNet-56", ["GRA", "L1", "FPGM"], [0.5]),
    ]
    
    for dataset, arch, methods, ratios in fig4_configs:
        for method in methods:
            for ratio in ratios:
                exp_id = f"fig4_{dataset}_{arch}_{method}_{ratio}".replace("-", "").lower()
                experiments.append({
                    "id": exp_id,
                    "type": "fig4_convergence",
                    "dataset": dataset,
                    "arch": arch,
                    "method": method,
                    "ratio": ratio,
                    "epochs": 60,  # 更长训练以观察收敛
                    "priority": 2
                })
    
    # === Figure 5 增强: ρ 参数扫描 ===
    # 更细粒度的 ρ 值
    rho_values = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    fig5_configs = [
        ("CIFAR-10", "ResNet-56", 0.5),
        ("CIFAR-100", "ResNet-56", 0.5),
    ]
    
    for dataset, arch, ratio in fig5_configs:
        for rho in rho_values:
            exp_id = f"fig5_{dataset}_{arch}_rho{rho}".replace("-", "").replace(".", "").lower()
            experiments.append({
                "id": exp_id,
                "type": "fig5_rho",
                "dataset": dataset,
                "arch": arch,
                "method": "GRA",
                "ratio": ratio,
                "rho": rho,
                "epochs": 30,
                "priority": 3
            })
    
    return experiments

def run_single_experiment(exp):
    """执行单个实验"""
    exp_id = exp["id"]
    start_time = time.time()
    
    try:
        log_event("START", f"开始实验 {exp_id}", exp)
        
        # 构建命令
        cmd = [
            PYTHON_EXE,
            str(EXPERIMENTS_DIR / "run_real_pruning.py"),
            "--arch", exp["arch"],
            "--dataset", exp["dataset"],
            "--method", exp["method"],
            "--ratio", str(exp["ratio"]),
            "--epochs", str(exp["epochs"]),
            "--output-dir", str(RESULTS_DIR / exp_id)
        ]
        
        # 添加 ρ 参数 (如果是 Figure 5 实验)
        if "rho" in exp:
            cmd.extend(["--rho", str(exp["rho"])])
        
        # 执行
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            log_event("SUCCESS", f"实验 {exp_id} 完成 ({elapsed:.1f}s)", {
                "elapsed": elapsed,
                "stdout_tail": result.stdout[-500:] if result.stdout else ""
            })
            return {"id": exp_id, "status": "success", "elapsed": elapsed}
        else:
            log_error(exp_id, f"Exit code {result.returncode}", result.stderr[-1000:] if result.stderr else "")
            return {"id": exp_id, "status": "failed", "error": result.stderr[-500:] if result.stderr else "Unknown"}
            
    except subprocess.TimeoutExpired:
        log_error(exp_id, "Timeout after 1 hour")
        return {"id": exp_id, "status": "timeout"}
    except Exception as e:
        log_error(exp_id, str(e), traceback.format_exc())
        return {"id": exp_id, "status": "error", "error": str(e)}

def run_experiment_batch(experiments, max_workers=2):
    """并行运行实验批次"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {executor.submit(run_single_experiment, exp): exp for exp in experiments}
        
        for future in concurrent.futures.as_completed(future_to_exp):
            exp = future_to_exp[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                log_error(exp["id"], f"Future exception: {e}")
                results.append({"id": exp["id"], "status": "exception", "error": str(e)})
    
    return results

def generate_enhanced_figures():
    """生成增强版图表"""
    log_event("FIGURES", "开始生成增强版图表...")
    
    # 调用各个图表生成脚本
    scripts = [
        str(PROJECT_ROOT / "APIN_Submission" / "vis" / "generate_fig3_enhanced.py"),
        str(PROJECT_ROOT / "APIN_Submission" / "vis" / "generate_fig4_convergence.py"),
        str(PROJECT_ROOT / "APIN_Submission" / "vis" / "rho_sensitivity_analysis.py"),
    ]
    
    for script in scripts:
        if Path(script).exists():
            try:
                subprocess.run([PYTHON_EXE, script], timeout=300)
                log_event("FIGURE", f"图表生成完成: {Path(script).name}")
            except Exception as e:
                log_error(f"figure_{Path(script).name}", str(e))

def main():
    """主入口"""
    start_time = datetime.now()
    log_event("INIT", f"通宵实验套件启动 @ {start_time.isoformat()}")
    log_event("CONFIG", f"预计运行时间: 10小时 (RTX 5090 全功率)")
    
    # 获取实验矩阵
    experiments = get_experiment_matrix()
    log_event("MATRIX", f"实验矩阵包含 {len(experiments)} 个实验")
    
    # 按优先级分组
    priority_groups = {}
    for exp in experiments:
        p = exp.get("priority", 99)
        if p not in priority_groups:
            priority_groups[p] = []
        priority_groups[p].append(exp)
    
    # 依次执行各优先级组
    all_results = []
    for priority in sorted(priority_groups.keys()):
        group = priority_groups[priority]
        log_event("BATCH", f"开始优先级 {priority} 批次 ({len(group)} 个实验)")
        
        results = run_experiment_batch(group, max_workers=2)
        all_results.extend(results)
        
        # 批次统计
        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        log_event("BATCH_DONE", f"优先级 {priority} 完成: {success} 成功, {failed} 失败")
    
    # 生成增强图表
    generate_enhanced_figures()
    
    # 最终报告
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds() / 3600
    
    total_success = sum(1 for r in all_results if r["status"] == "success")
    total_failed = len(all_results) - total_success
    
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_hours": round(elapsed, 2),
        "total_experiments": len(all_results),
        "successful": total_success,
        "failed": total_failed,
        "success_rate": f"{100*total_success/len(all_results):.1f}%"
    }
    
    log_event("COMPLETE", "通宵实验完成!", summary)
    
    # 保存最终摘要
    with open(RESULTS_DIR / "overnight_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("通宵实验完成!")
    print(f"总耗时: {elapsed:.2f} 小时")
    print(f"成功率: {summary['success_rate']}")
    print(f"日志: {LOG_FILE}")
    print(f"错误: {ERROR_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
