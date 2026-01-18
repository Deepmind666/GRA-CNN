"""
最终论文图表生成器 - 使用82条真实实验数据
==========================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

# 加载真实数据
df = pd.read_csv(r'C:\GRA-CNN\experiments\complete_data.csv')
print(f'Loaded {len(df)} real experiment records')

# 专业配色 (colorblind-friendly)
COLORS = {
    'GRA': '#D55E00',      # 橙红 (主角)
    'L1': '#0072B2',       # 蓝色
    'FPGM': '#009E73',     # 绿色
    'HRank': '#CC79A7',    # 粉色
}

MARKERS = {'GRA': 'o', 'L1': 's', 'FPGM': '^', 'HRank': 'D'}
LINESTYLES = {'GRA': '-', 'L1': '--', 'FPGM': '-.', 'HRank': ':'}

# 设置专业样式
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

def get_data(arch, dataset, method):
    subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & (df['method']==method)]
    subset = subset.sort_values('ratio')
    return subset['ratio'].values, subset['accuracy'].values

# ============================================================================
# Figure 2: CIFAR-10 主结果 (4面板)
# ============================================================================

def create_cifar10_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    
    configs = [
        ('ResNet-20', 'CIFAR-10', 'a'),
        ('ResNet-56', 'CIFAR-10', 'b'),
        ('ResNet-110', 'CIFAR-10', 'c'),
        ('VGG-16', 'CIFAR-10', 'd'),
    ]
    
    for idx, (arch, dataset, letter) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        
        for method in ['GRA', 'L1', 'FPGM', 'HRank']:
            ratios, accs = get_data(arch, dataset, method)
            if len(ratios) > 0:
                ax.plot(ratios, accs, marker=MARKERS[method], color=COLORS[method],
                       linestyle=LINESTYLES[method], label=method if idx==0 else '',
                       linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
        
        ax.set_title(f'({letter}) {arch}', fontweight='bold')
        if idx >= 2:
            ax.set_xlabel('Pruning Ratio')
        if idx % 2 == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_xlim(0.25, 0.85)
    
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('CIFAR-10 Pruning Results', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    plt.savefig(f'{OUTPUT}/fig2_final.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig2_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig2_final.pdf/png')

# ============================================================================
# Figure 3: CIFAR-100 结果
# ============================================================================

def create_cifar100_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    configs = [
        ('ResNet-20', 'CIFAR-100', 'a'),
        ('ResNet-56', 'CIFAR-100', 'b'),
    ]
    
    for idx, (arch, dataset, letter) in enumerate(configs):
        ax = axes[idx]
        
        for method in ['GRA', 'L1', 'FPGM', 'HRank']:
            ratios, accs = get_data(arch, dataset, method)
            if len(ratios) > 0:
                ax.plot(ratios, accs, marker=MARKERS[method], color=COLORS[method],
                       linestyle=LINESTYLES[method], label=method if idx==0 else '',
                       linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
        
        ax.set_title(f'({letter}) {arch}', fontweight='bold')
        ax.set_xlabel('Pruning Ratio')
        if idx == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_xlim(0.25, 0.75)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('CIFAR-100 Pruning Results', fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    plt.savefig(f'{OUTPUT}/fig3_final.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig3_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig3_final.pdf/png')

# ============================================================================
# Figure: Tiny-ImageNet 结果
# ============================================================================

def create_tiny_figure():
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    for method in ['GRA', 'L1']:
        ratios, accs = get_data('ResNet-18', 'Tiny-ImageNet', method)
        if len(ratios) > 0:
            ax.plot(ratios, accs, marker=MARKERS[method], color=COLORS[method],
                   linestyle=LINESTYLES[method], label=method,
                   linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
    
    ax.set_title('ResNet-18 on Tiny-ImageNet', fontweight='bold')
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig_tiny_final.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig_tiny_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig_tiny_final.pdf/png')

# ============================================================================
# Figure: GRA vs L1 优势对比
# ============================================================================

def create_comparison_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # 收集GRA vs L1对比数据
    comparisons = []
    for _, gra_row in df[df['method']=='GRA'].iterrows():
        l1_match = df[(df['architecture']==gra_row['architecture']) &
                     (df['dataset']==gra_row['dataset']) &
                     (df['method']=='L1') &
                     (df['ratio']==gra_row['ratio'])]
        if len(l1_match) > 0:
            comparisons.append({
                'gra': gra_row['accuracy'],
                'l1': l1_match['accuracy'].values[0],
                'arch': gra_row['architecture'],
                'dataset': gra_row['dataset'],
                'ratio': gra_row['ratio']
            })
    
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        
        # 左图: 散点图
        ax1.scatter(comp_df['l1'], comp_df['gra'], c=COLORS['GRA'], s=50, 
                   alpha=0.7, edgecolors='white', linewidths=0.8)
        lims = [min(comp_df['l1'].min(), comp_df['gra'].min())-2,
                max(comp_df['l1'].max(), comp_df['gra'].max())+2]
        ax1.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('L1-Norm Accuracy (%)')
        ax1.set_ylabel('GRA-CNN Accuracy (%)')
        ax1.set_title('(a) Performance Comparison', fontweight='bold')
        
        mean_imp = (comp_df['gra'] - comp_df['l1']).mean()
        ax1.annotate(f'Mean Δ = {mean_imp:+.2f}%', xy=(0.05, 0.92),
                    xycoords='axes fraction', fontsize=8, fontweight='bold', color=COLORS['GRA'])
        
        # 右图: 柱状图
        comp_df['improvement'] = comp_df['gra'] - comp_df['l1']
        comp_df = comp_df.sort_values('improvement', ascending=False)
        colors = [COLORS['GRA'] if v > 0 else '#888888' for v in comp_df['improvement']]
        
        bars = ax2.bar(range(len(comp_df)), comp_df['improvement'], color=colors, 
                      edgecolor='white', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axhline(mean_imp, color=COLORS['GRA'], linestyle='--', linewidth=1,
                   label=f'Mean: {mean_imp:+.2f}%')
        ax2.set_xlabel('Experiment Configuration')
        ax2.set_ylabel('Accuracy Improvement (%)')
        ax2.set_title('(b) GRA Improvement over L1', fontweight='bold')
        ax2.legend(fontsize=7)
        ax2.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig_comparison_final.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig_comparison_final.pdf/png')

# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print('='*60)
    print('Final Publication Figure Generator')
    print('='*60)
    
    create_cifar10_figure()
    create_cifar100_figure()
    create_tiny_figure()
    create_comparison_figure()
    
    print()
    print('All final figures generated!')
