"""
全面数据采集器 - 提取所有111个CSV文件的实验结果
================================================
"""

import os
import pandas as pd
import glob
import re

EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'

def extract_config_from_path(path):
    """从路径和文件内容推断实验配置"""
    dirname = os.path.dirname(path).split(os.sep)[-1]
    filename = os.path.basename(path)
    
    config = {
        'architecture': None,
        'dataset': None,
        'method': None,
        'ratio': None,
        'source': path
    }
    
    # 从路径推断
    path_lower = path.lower()
    
    # 架构
    if 'resnet110' in path_lower or 'resnet-110' in path_lower:
        config['architecture'] = 'ResNet-110'
    elif 'resnet56' in path_lower or 'resnet-56' in path_lower or 'r56' in path_lower:
        config['architecture'] = 'ResNet-56'
    elif 'resnet20' in path_lower or 'resnet-20' in path_lower or 'r20' in path_lower:
        config['architecture'] = 'ResNet-20'
    elif 'resnet18' in path_lower or 'r18' in path_lower:
        config['architecture'] = 'ResNet-18'
    elif 'vgg' in path_lower:
        config['architecture'] = 'VGG-16'
    
    # 数据集
    if 'cifar100' in path_lower or 'cifar-100' in path_lower:
        config['dataset'] = 'CIFAR-100'
    elif 'cifar10' in path_lower or 'cifar-10' in path_lower:
        config['dataset'] = 'CIFAR-10'
    elif 'tiny' in path_lower or 'imagenet' in path_lower:
        config['dataset'] = 'Tiny-ImageNet'
    
    # 方法
    if '_gra_' in path_lower or '_gra.' in path_lower or '/gra_' in path_lower:
        config['method'] = 'GRA'
    elif '_l1_' in path_lower or '_l1.' in path_lower or '/l1_' in path_lower:
        config['method'] = 'L1'
    elif '_fpgm_' in path_lower or '_fpgm.' in path_lower:
        config['method'] = 'FPGM'
    elif '_hrank_' in path_lower or '_hrank.' in path_lower:
        config['method'] = 'HRank'
    elif 'baseline' in path_lower:
        config['method'] = 'Baseline'
    
    # 剪枝率
    ratio_match = re.search(r'[_/]0\.(\d)', path_lower)
    if ratio_match:
        config['ratio'] = float(f"0.{ratio_match.group(1)}")
    elif 'baseline' in path_lower:
        config['ratio'] = 0.0
    
    return config

def extract_accuracy(csv_path):
    """从CSV提取精度"""
    try:
        df = pd.read_csv(csv_path)
        
        # 尝试不同的列名
        acc_cols = ['test_acc', 'acc', 'accuracy', 'val_acc', 'top1', 'pruned_acc', 'best_acc']
        for col in acc_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    return vals.iloc[-1]  # 最后一个值
        
        # 如果没有找到，检查是否有数值列
        for col in df.columns:
            if 'acc' in col.lower():
                vals = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(vals) > 0:
                    return vals.iloc[-1]
        
    except Exception as e:
        pass
    return None

def main():
    print("="*70)
    print("全面数据采集 - 扫描所有111个CSV文件")
    print("="*70)
    
    all_csvs = glob.glob(os.path.join(EXPERIMENTS_DIR, '**', '*.csv'), recursive=True)
    print(f"发现 {len(all_csvs)} 个CSV文件")
    
    results = []
    
    for csv_path in sorted(all_csvs):
        config = extract_config_from_path(csv_path)
        acc = extract_accuracy(csv_path)
        
        if acc is not None and config['architecture'] is not None:
            config['accuracy'] = acc
            results.append(config)
            print(f"✓ {config['architecture']}/{config['dataset']}/{config['method']}/r={config['ratio']}: {acc:.2f}%")
    
    print(f"\n总共提取 {len(results)} 条有效记录")
    
    # 保存
    df = pd.DataFrame(results)
    output_path = os.path.join(EXPERIMENTS_DIR, 'complete_data.csv')
    df.to_csv(output_path, index=False)
    print(f"保存到: {output_path}")
    
    # 统计
    print("\n" + "="*70)
    print("数据统计:")
    print("="*70)
    
    for arch in df['architecture'].unique():
        arch_df = df[df['architecture'] == arch]
        for dataset in arch_df['dataset'].unique():
            ds_df = arch_df[arch_df['dataset'] == dataset]
            methods = ds_df['method'].unique()
            print(f"\n{arch} on {dataset}:")
            for method in methods:
                m_df = ds_df[ds_df['method'] == method]
                ratios = sorted(m_df['ratio'].dropna().unique())
                print(f"  {method}: {len(m_df)} records, ratios={ratios}")

if __name__ == '__main__':
    main()
