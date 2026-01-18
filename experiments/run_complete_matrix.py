"""
完整实验矩阵运行器
==================
目标: 补充所有缺失的实验，确保每个配置都有完整数据

实验矩阵:
- 架构: ResNet-20, ResNet-56, ResNet-110, VGG-16
- 数据集: CIFAR-10, CIFAR-100
- 方法: GRA, L1, FPGM, HRank
- 剪枝率: 0.3, 0.4, 0.5, 0.6, 0.7

总计: 4 × 2 × 4 × 5 = 160个实验配置
"""

import os
import sys
import pandas as pd
import subprocess
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, r'C:\GRA-CNN')

# ============================================================================
# 实验配置
# ============================================================================

ARCHITECTURES = ['resnet20', 'resnet56', 'resnet110', 'vgg16']
DATASETS = ['cifar10', 'cifar100']
METHODS = ['gra', 'l1', 'fpgm', 'hrank']
RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]

PYTHON_EXE = r'C:\Users\admin\anaconda3\envs\pyt-sm12-build\python.exe'
RUN_SCRIPT = r'C:\GRA-CNN\run_all.py'
EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'
LOG_FILE = r'C:\GRA-CNN\experiments\experiment_queue_log.txt'

# ============================================================================
# 检查已有实验
# ============================================================================

def load_existing_data():
    """加载已有实验数据"""
    csv_path = os.path.join(EXPERIMENTS_DIR, 'complete_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def check_experiment_exists(arch, dataset, method, ratio, existing_df):
    """检查实验是否已存在"""
    if existing_df.empty:
        return False
    
    # 标准化名称
    arch_map = {'resnet20': 'ResNet-20', 'resnet56': 'ResNet-56', 
                'resnet110': 'ResNet-110', 'vgg16': 'VGG-16'}
    dataset_map = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}
    method_map = {'gra': 'GRA', 'l1': 'L1', 'fpgm': 'FPGM', 'hrank': 'HRank'}
    
    arch_std = arch_map.get(arch, arch)
    dataset_std = dataset_map.get(dataset, dataset)
    method_std = method_map.get(method, method)
    
    match = existing_df[
        (existing_df['architecture'] == arch_std) &
        (existing_df['dataset'] == dataset_std) &
        (existing_df['method'] == method_std) &
        (existing_df['ratio'] == ratio)
    ]
    
    return len(match) > 0

# ============================================================================
# 生成实验队列
# ============================================================================

def generate_experiment_queue():
    """生成需要运行的实验队列"""
    existing_df = load_existing_data()
    
    queue = []
    existing_count = 0
    
    for arch in ARCHITECTURES:
        for dataset in DATASETS:
            for method in METHODS:
                for ratio in RATIOS:
                    if check_experiment_exists(arch, dataset, method, ratio, existing_df):
                        existing_count += 1
                    else:
                        queue.append({
                            'arch': arch,
                            'dataset': dataset,
                            'method': method,
                            'ratio': ratio
                        })
    
    print(f"已有实验: {existing_count}")
    print(f"需要运行: {len(queue)}")
    
    return queue

# ============================================================================
# 运行单个实验
# ============================================================================

def run_experiment(config, timeout_minutes=30):
    """运行单个实验"""
    arch = config['arch']
    dataset = config['dataset']
    method = config['method']
    ratio = config['ratio']
    
    cmd = [
        PYTHON_EXE, RUN_SCRIPT,
        '--arch', arch,
        '--dataset', dataset,
        '--method', method,
        '--ratio', str(ratio),
        '--rho', '0.5',
        '--epochs', '60',  # 使用60个epoch以节省时间
    ]
    
    print(f"\n{'='*60}")
    print(f"运行: {arch}/{dataset}/{method}/r={ratio}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=r'C:\GRA-CNN',
            timeout=timeout_minutes * 60,
            capture_output=True,
            text=True
        )
        
        elapsed = (time.time() - start_time) / 60
        
        if result.returncode == 0:
            print(f"✅ 完成 (耗时: {elapsed:.1f}分钟)")
            return True, elapsed
        else:
            print(f"❌ 失败: {result.stderr[:200]}")
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 超时 (>{timeout_minutes}分钟)")
        return False, timeout_minutes
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False, 0

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("="*70)
    print("完整实验矩阵运行器")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 生成队列
    queue = generate_experiment_queue()
    
    if len(queue) == 0:
        print("所有实验已完成！")
        return
    
    print(f"\n实验队列 ({len(queue)}个):")
    for i, exp in enumerate(queue[:20]):
        print(f"  {i+1}. {exp['arch']}/{exp['dataset']}/{exp['method']}/r={exp['ratio']}")
    if len(queue) > 20:
        print(f"  ... 还有 {len(queue)-20} 个")
    
    # 估计时间
    estimated_hours = len(queue) * 15 / 60  # 假设每个实验15分钟
    print(f"\n预计运行时间: {estimated_hours:.1f}小时")
    
    # 开始运行
    success_count = 0
    fail_count = 0
    
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n\n{'='*60}\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"队列大小: {len(queue)}\n")
        f.write(f"{'='*60}\n")
    
    for i, config in enumerate(queue):
        print(f"\n[{i+1}/{len(queue)}] ", end='')
        
        success, elapsed = run_experiment(config)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # 记录日志
        with open(LOG_FILE, 'a') as f:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{config['arch']}/{config['dataset']}/{config['method']}/r={config['ratio']}: {status} ({elapsed:.1f}min)\n")
        
        # 每10个实验打印统计
        if (i + 1) % 10 == 0:
            print(f"\n--- 进度: {i+1}/{len(queue)}, 成功: {success_count}, 失败: {fail_count} ---")
    
    # 最终统计
    print("\n" + "="*70)
    print(f"运行完成!")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == '__main__':
    main()
