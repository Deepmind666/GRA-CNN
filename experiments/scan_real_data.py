"""
真实数据采集器 - 只采集真实实验结果
=====================================
扫描所有实验子目录，提取真实的训练日志和最终精度
"""

import os
import pandas as pd
import glob
import re

EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'
OUTPUT_FILE = r'C:\GRA-CNN\experiments\real_data_audit.csv'

def parse_experiment_dir(dirname):
    """从目录名解析实验配置"""
    # 格式: cifar10_resnet20_gra_0.5 或 resnet20_gra_r0.5_rho0.5
    patterns = [
        r'(cifar\d+)_(resnet\d+|vgg\d+)_(gra|l1|fpgm|hrank)_(\d+\.?\d*)',
        r'(resnet\d+|vgg\d+)_(gra|l1|fpgm|hrank)_r(\d+\.?\d*)_rho(\d+\.?\d*)',
        r'(resnet\d+|vgg\d+)_(gra|l1)_r(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, dirname, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 4 and 'cifar' in groups[0].lower():
                return {
                    'dataset': groups[0],
                    'architecture': groups[1],
                    'method': groups[2],
                    'ratio': float(groups[3]),
                    'rho': 0.5
                }
            elif len(groups) == 4:
                return {
                    'dataset': 'cifar10',  # 默认
                    'architecture': groups[0],
                    'method': groups[1],
                    'ratio': float(groups[2]),
                    'rho': float(groups[3])
                }
            elif len(groups) == 3:
                return {
                    'dataset': 'cifar10',
                    'architecture': groups[0],
                    'method': groups[1],
                    'ratio': float(groups[2]),
                    'rho': 0.5
                }
    return None

def extract_accuracy_from_csv(csv_path):
    """从训练日志CSV提取最终精度"""
    try:
        df = pd.read_csv(csv_path)
        # 尝试不同的列名
        for col in ['test_acc', 'acc', 'accuracy', 'val_acc', 'top1']:
            if col in df.columns:
                return df[col].iloc[-1]  # 最后一个epoch
        # 如果没有找到，尝试数值列
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            return df[numeric_cols[-1]].iloc[-1]
    except Exception as e:
        print(f"  [警告] 无法解析 {csv_path}: {e}")
    return None

def scan_experiments():
    """扫描所有实验目录"""
    results = []
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(EXPERIMENTS_DIR) 
               if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    
    print(f"发现 {len(subdirs)} 个子目录")
    print("="*60)
    
    for dirname in sorted(subdirs):
        dirpath = os.path.join(EXPERIMENTS_DIR, dirname)
        
        # 解析目录名
        config = parse_experiment_dir(dirname)
        if config is None:
            continue
        
        # 查找CSV文件
        csv_files = glob.glob(os.path.join(dirpath, '*.csv'))
        
        for csv_file in csv_files:
            acc = extract_accuracy_from_csv(csv_file)
            if acc is not None:
                result = config.copy()
                result['accuracy'] = acc
                result['source_file'] = os.path.basename(csv_file)
                result['source_dir'] = dirname
                results.append(result)
                print(f"✓ {dirname}: {acc:.2f}%")
    
    return results

def main():
    print("="*60)
    print("真实数据采集器 - 开始扫描")
    print("="*60)
    
    results = scan_experiments()
    
    print("\n" + "="*60)
    print(f"共采集到 {len(results)} 条真实实验记录")
    print("="*60)
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n已保存到: {OUTPUT_FILE}")
        
        # 统计摘要
        print("\n数据摘要:")
        print(f"  架构: {df['architecture'].unique()}")
        print(f"  数据集: {df['dataset'].unique()}")
        print(f"  方法: {df['method'].unique()}")
        print(f"  剪枝率: {sorted(df['ratio'].unique())}")
        
        # 按配置分组显示
        print("\n按配置分组:")
        for (arch, dataset, method), group in df.groupby(['architecture', 'dataset', 'method']):
            ratios = sorted(group['ratio'].unique())
            accs = [group[group['ratio']==r]['accuracy'].values[0] for r in ratios if r in group['ratio'].values]
            print(f"  {arch}/{dataset}/{method}: ratios={ratios}")
            for r, a in zip(ratios, accs):
                print(f"    ratio={r}: acc={a:.2f}%")

if __name__ == '__main__':
    main()
