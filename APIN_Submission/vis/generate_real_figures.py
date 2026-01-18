"""
专业论文图表生成器 - 仅使用真实数据
===================================
特点:
1. 只使用real_data_audit.csv中的真实实验数据
2. 缺失数据留空，不造假
3. Nature/Science级专业样式
4. 彩色、清晰、美观
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

# ============================================================================
# 加载真实数据
# ============================================================================

print("加载真实实验数据...")
df = pd.read_csv(r'C:\GRA-CNN\experiments\real_data_audit.csv')
print(f"总记录数: {len(df)}")

# 打印数据摘要
print("\n数据摘要:")
for (arch, dataset), group in df.groupby(['architecture', 'dataset']):
    methods = group['method'].unique()
    print(f"  {arch}/{dataset}: methods={list(methods)}")

# ============================================================================
# 专业样式设置
# ============================================================================

# Nature风格颜色
COLORS = {
    'gra': '#E64B35',      # 红色
    'l1': '#4DBBD5',       # 蓝色
    'fpgm': '#00A087',     # 绿色
    'hrank': '#3C5488',    # 深蓝
}

MARKERS = {'gra': 'o', 'l1': 's', 'fpgm': '^', 'hrank': 'D'}
LINESTYLES = {'gra': '-', 'l1': '--', 'fpgm': '-.', 'hrank': ':'}
METHOD_NAMES = {'gra': 'GRA-CNN', 'l1': 'L1-Norm', 'fpgm': 'FPGM', 'hrank': 'HRank'}

def set_professional_style():
    """设置专业论文样式"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })

OUTPUT_DIR = r'C:\GRA-CNN\APIN_Submission'

# ============================================================================
# 数据提取函数
# ============================================================================

def get_real_data(arch, dataset, method):
    """提取指定配置的真实数据"""
    subset = df[
        (df['architecture'].str.lower() == arch.lower()) &
        (df['dataset'].str.lower() == dataset.lower()) &
        (df['method'].str.lower() == method.lower())
    ].copy()
    subset = subset.sort_values('ratio')
    if len(subset) > 0:
        return subset['ratio'].values, subset['accuracy'].values
    return None, None

def get_available_configs():
    """获取所有可用的配置组合"""
    configs = []
    for (arch, dataset), group in df.groupby(['architecture', 'dataset']):
        methods = list(group['method'].unique())
        configs.append({
            'architecture': arch,
            'dataset': dataset,
            'methods': methods
        })
    return configs

# ============================================================================
# 图表1: 主结果图 - 每个配置一个面板，多方法对比
# ============================================================================

def create_main_results_figure():
    """创建主结果图 - 使用真实数据"""
    set_professional_style()
    
    configs = get_available_configs()
    n_panels = len(configs)
    
    if n_panels == 0:
        print("错误: 没有可用数据!")
        return
    
    # 计算网格尺寸
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4*n_cols, 3.5*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.35, wspace=0.28)
    
    panel_idx = 0
    for config in configs:
        arch = config['architecture']
        dataset = config['dataset']
        methods = config['methods']
        
        row = panel_idx // n_cols
        col = panel_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        has_data = False
        for method in methods:
            ratios, accs = get_real_data(arch, dataset, method)
            if ratios is not None and len(ratios) > 0:
                color = COLORS.get(method.lower(), 'gray')
                marker = MARKERS.get(method.lower(), 'o')
                ls = LINESTYLES.get(method.lower(), '-')
                label = METHOD_NAMES.get(method.lower(), method.upper())
                
                ax.plot(ratios, accs, marker=marker, color=color, linestyle=ls,
                       label=label, linewidth=2.0, markersize=7,
                       markeredgecolor='white', markeredgewidth=1.0)
                has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f'({chr(97+panel_idx)}) {arch} on {dataset}', fontweight='bold')
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        if row == n_rows - 1:
            ax.set_xlabel('Pruning Ratio')
        
        if has_data and panel_idx == 0:
            ax.legend(loc='lower left', fontsize=7)
        
        panel_idx += 1
    
    fig.suptitle('Pruning Performance: Real Experimental Results', 
                fontweight='bold', fontsize=12, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_main_results.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_main_results.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已创建: fig_real_main_results.pdf/png ({n_panels}个面板)")


# ============================================================================
# 图表2: ResNet对比图 - 同一架构不同方法
# ============================================================================

def create_resnet_comparison():
    """创建ResNet系列对比图"""
    set_professional_style()
    
    # 找到所有ResNet配置
    resnet_configs = df[df['architecture'].str.contains('resnet', case=False)]
    
    if len(resnet_configs) == 0:
        print("没有ResNet数据")
        return
    
    archs = sorted(resnet_configs['architecture'].unique())
    
    n_panels = len(archs)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.30, wspace=0.25)
    
    for idx, arch in enumerate(archs):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        arch_data = resnet_configs[resnet_configs['architecture'] == arch]
        methods = arch_data['method'].unique()
        
        for method in methods:
            ratios, accs = get_real_data(arch, 'cifar10', method)
            if ratios is None:
                ratios, accs = get_real_data(arch, 'cifar100', method)
            
            if ratios is not None and len(ratios) > 0:
                color = COLORS.get(method.lower(), 'gray')
                marker = MARKERS.get(method.lower(), 'o')
                ls = LINESTYLES.get(method.lower(), '-')
                label = METHOD_NAMES.get(method.lower(), method.upper())
                
                ax.plot(ratios, accs, marker=marker, color=color, linestyle=ls,
                       label=label, linewidth=2.2, markersize=8,
                       markeredgecolor='white', markeredgewidth=1.2)
        
        ax.set_title(f'({chr(97+idx)}) {arch}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Pruning Ratio')
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=8)
    
    fig.suptitle('ResNet Family: Pruning Method Comparison', 
                fontweight='bold', fontsize=13, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_resnet_comparison.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_resnet_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已创建: fig_real_resnet_comparison.pdf/png")


# ============================================================================
# 图表3: GRA-CNN优势图 (GRA vs L1 对比)
# ============================================================================

def create_gra_advantage_figure():
    """创建GRA-CNN相对于L1的优势对比图"""
    set_professional_style()
    
    # 找到同时有GRA和L1数据的配置
    comparisons = []
    
    for (arch, dataset), group in df.groupby(['architecture', 'dataset']):
        gra_data = group[group['method'].str.lower() == 'gra']
        l1_data = group[group['method'].str.lower() == 'l1']
        
        if len(gra_data) > 0 and len(l1_data) > 0:
            # 找到共同的ratio
            gra_ratios = set(gra_data['ratio'].values)
            l1_ratios = set(l1_data['ratio'].values)
            common_ratios = sorted(gra_ratios & l1_ratios)
            
            for r in common_ratios:
                gra_acc = gra_data[gra_data['ratio'] == r]['accuracy'].values[0]
                l1_acc = l1_data[l1_data['ratio'] == r]['accuracy'].values[0]
                comparisons.append({
                    'arch': arch,
                    'dataset': dataset,
                    'ratio': r,
                    'gra_acc': gra_acc,
                    'l1_acc': l1_acc,
                    'improvement': gra_acc - l1_acc
                })
    
    if len(comparisons) == 0:
        print("没有GRA vs L1对比数据")
        return
    
    comp_df = pd.DataFrame(comparisons)
    print(f"\nGRA vs L1 对比数据:")
    print(comp_df)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图: 散点图 GRA vs L1
    ax1.scatter(comp_df['l1_acc'], comp_df['gra_acc'], 
               c=COLORS['gra'], s=100, alpha=0.7, edgecolors='white', linewidths=1.5)
    
    # 对角线
    min_acc = min(comp_df['l1_acc'].min(), comp_df['gra_acc'].min()) - 1
    max_acc = max(comp_df['l1_acc'].max(), comp_df['gra_acc'].max()) + 1
    ax1.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('L1-Norm Accuracy (%)', fontweight='medium')
    ax1.set_ylabel('GRA-CNN Accuracy (%)', fontweight='medium')
    ax1.set_title('(a) GRA-CNN vs L1-Norm Performance', fontweight='bold')
    
    # 右图: 改进柱状图
    labels = [f"{row['arch']}\n{row['ratio']}" for _, row in comp_df.iterrows()]
    colors = [COLORS['gra'] if imp > 0 else 'gray' for imp in comp_df['improvement']]
    
    ax2.bar(range(len(comp_df)), comp_df['improvement'], color=colors, edgecolor='white', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=comp_df['improvement'].mean(), color=COLORS['gra'], linestyle='--', 
               linewidth=1.5, label=f"Mean: {comp_df['improvement'].mean():.2f}%")
    
    ax2.set_xticks(range(len(comp_df)))
    ax2.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Accuracy Improvement (%)', fontweight='medium')
    ax2.set_title('(b) GRA-CNN Improvement over L1-Norm', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_gra_advantage.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_real_gra_advantage.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已创建: fig_real_gra_advantage.pdf/png")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "="*60)
    print("专业论文图表生成器 - 仅使用真实数据")
    print("="*60)
    
    create_main_results_figure()
    create_resnet_comparison()
    create_gra_advantage_figure()
    
    print("\n" + "="*60)
    print("所有图表已基于真实实验数据生成!")
    print("="*60)

if __name__ == '__main__':
    main()
