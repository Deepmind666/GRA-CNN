"""
最终论文图表生成器 - 使用完整实验数据
====================================
按照参考图表风格:
- 12面板 (4列×3行)
- 4种方法对比: GRA-CNN (红), L1 (蓝), FPGM (青), HRank (青)
- 5个剪枝率: 0.3, 0.4, 0.5, 0.6, 0.7
- 基线参考线
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

# ============================================================================
# 加载所有实验数据
# ============================================================================

def load_all_data():
    """加载并合并所有实验数据"""
    all_dfs = []
    
    files = [
        r'C:\GRA-CNN\experiments\comprehensive_10hr_results.csv',
        r'C:\GRA-CNN\experiments\supplementary_results.csv',
        r'C:\GRA-CNN\experiments\complete_data.csv',
    ]
    
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                all_dfs.append(df)
                print(f'Loaded {f}: {len(df)} records')
            except:
                pass
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # 标准化列名
    if 'accuracy' in combined.columns:
        combined['acc'] = combined['accuracy']
    elif 'final_acc' in combined.columns:
        combined['acc'] = combined['final_acc']
    elif 'pruned_acc' in combined.columns:
        combined['acc'] = combined['pruned_acc']
    
    # 标准化架构名
    combined['arch_norm'] = combined['architecture'].str.replace('-', '').str.lower()
    combined['dataset_norm'] = combined['dataset'].str.replace('-', '').str.lower()
    combined['method_norm'] = combined['method'].str.upper()
    
    print(f'Total: {len(combined)} records')
    return combined

# ============================================================================
# 配色方案 (按照参考图)
# ============================================================================

COLORS = {
    'GRA': '#D62728',      # 红色 (主角)
    'L1': '#1F77B4',       # 蓝色
    'FPGM': '#17BECF',     # 青色
    'HRANK': '#17BECF',    # 青色
}

LINESTYLES = {
    'GRA': '-',
    'L1': '--',
    'FPGM': '-.',
    'HRANK': ':',
}

MARKERS = {
    'GRA': 'o',
    'L1': 's',
    'FPGM': '^',
    'HRANK': 'D',
}

LABELS = {
    'GRA': 'GRA-CNN',
    'L1': 'L1-Norm',
    'FPGM': 'FPGM',
    'HRANK': 'HRank',
}

# ============================================================================
# 设置专业样式
# ============================================================================

def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'figure.facecolor': 'white',
    })

OUTPUT = r'C:\GRA-CNN\APIN_Submission'

# ============================================================================
# 数据提取函数
# ============================================================================

def get_data(df, arch, dataset, method):
    """提取指定配置的数据"""
    arch_key = arch.replace('-', '').lower()
    dataset_key = dataset.replace('-', '').lower()
    method_key = method.upper()
    
    subset = df[(df['arch_norm'] == arch_key) & 
                (df['dataset_norm'] == dataset_key) & 
                (df['method_norm'] == method_key)]
    
    if len(subset) == 0:
        return np.array([]), np.array([])
    
    # 去重并按ratio排序
    subset = subset.drop_duplicates(subset=['ratio']).sort_values('ratio')
    
    if 'acc' in subset.columns:
        return subset['ratio'].values, subset['acc'].values
    return np.array([]), np.array([])

# ============================================================================
# 创建12面板主结果图
# ============================================================================

def create_12panel_figure(df):
    """创建12面板主结果图"""
    set_style()
    
    # 12面板配置 (4列 × 3行)
    panels = [
        # Row 1: CIFAR-10
        ('ResNet-20', 'CIFAR-10', 'a'),
        ('ResNet-32', 'CIFAR-10', 'b'),
        ('ResNet-44', 'CIFAR-10', 'c'),
        ('ResNet-56', 'CIFAR-10', 'd'),
        # Row 2: 混合
        ('ResNet-110', 'CIFAR-10', 'e'),
        ('VGG-16', 'CIFAR-10', 'f'),
        ('ResNet-20', 'CIFAR-100', 'g'),
        ('ResNet-32', 'CIFAR-100', 'h'),
        # Row 3: CIFAR-100
        ('ResNet-44', 'CIFAR-100', 'i'),
        ('ResNet-56', 'CIFAR-100', 'j'),
        ('ResNet-110', 'CIFAR-100', 'k'),
        ('VGG-16', 'CIFAR-100', 'l'),
    ]
    
    methods = ['GRA', 'L1', 'FPGM', 'HRANK']
    
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3)
    
    for idx, (arch, dataset, letter) in enumerate(panels):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        has_data = False
        for method in methods:
            ratios, accs = get_data(df, arch, dataset, method)
            if len(ratios) > 0:
                ax.plot(ratios, accs,
                       marker=MARKERS[method],
                       color=COLORS[method],
                       linestyle=LINESTYLES[method],
                       label=LABELS[method] if idx == 0 else '',
                       linewidth=1.8,
                       markersize=5,
                       markeredgecolor='white',
                       markeredgewidth=0.6)
                has_data = True
        
        ax.set_title(f'({letter}) {arch} on {dataset}', fontweight='bold', fontsize=9)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=8)
        if row == 2:
            ax.set_xlabel('Pruning Ratio', fontsize=8)
        
        ax.set_xlim(0.25, 0.75)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray', alpha=0.7)
    
    # 图例
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('Figure 2: Pruning Performance Across Architectures and Datasets',
                fontweight='bold', fontsize=11, y=0.98)
    
    plt.savefig(f'{OUTPUT}/fig2_12panel_final.pdf', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{OUTPUT}/fig2_12panel_final.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Created: fig2_12panel_final.pdf/png')

# ============================================================================
# 创建消融实验图
# ============================================================================

def create_ablation_figure(df):
    """创建ρ参数消融实验图"""
    set_style()
    
    # 假设有rho列
    if 'rho' not in df.columns:
        print('No rho data for ablation figure')
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    configs = [
        ('ResNet-20', 'CIFAR-10'),
        ('ResNet-56', 'CIFAR-10'),
        ('ResNet-20', 'CIFAR-100'),
        ('ResNet-56', 'CIFAR-100'),
    ]
    
    rho_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for idx, (arch, dataset) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        
        arch_key = arch.replace('-', '').lower()
        dataset_key = dataset.replace('-', '').lower()
        
        subset = df[(df['arch_norm'] == arch_key) & 
                    (df['dataset_norm'] == dataset_key) &
                    (df['method_norm'] == 'GRA')]
        
        if 'rho' in subset.columns and len(subset) > 0:
            for ratio in [0.3, 0.5, 0.7]:
                ratio_data = subset[subset['ratio'] == ratio]
                if len(ratio_data) > 0:
                    rhos = ratio_data['rho'].values
                    accs = ratio_data['acc'].values if 'acc' in ratio_data else []
                    if len(accs) > 0:
                        ax.plot(rhos, accs, marker='o', label=f'r={ratio}', linewidth=1.5)
        
        ax.set_title(f'({chr(97+idx)}) {arch} on {dataset}', fontweight='bold')
        ax.set_xlabel('ρ value')
        if idx % 2 == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    fig.suptitle('Figure: Ablation Study of ρ Parameter', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(f'{OUTPUT}/fig_ablation_rho_final.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig_ablation_rho_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig_ablation_rho_final.pdf/png')

# ============================================================================
# 主程序
# ============================================================================

def main():
    print('='*60)
    print('最终论文图表生成器')
    print('='*60)
    
    df = load_all_data()
    
    if len(df) == 0:
        print('No data found!')
        return
    
    create_12panel_figure(df)
    create_ablation_figure(df)
    
    print()
    print('='*60)
    print('所有图表生成完成!')
    print('='*60)

if __name__ == '__main__':
    main()
