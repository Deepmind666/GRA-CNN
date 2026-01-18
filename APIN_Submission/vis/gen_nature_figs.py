"""
Nature/Science风格图表生成器
===========================
使用120条真实实验数据
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

# 加载数据
df = pd.read_csv(r'C:\GRA-CNN\experiments\all_experiment_results.csv')
print(f'Loaded {len(df)} experiments')

# Nature配色
COLORS = {'gra': '#DC3220', 'l1': '#0072B2', 'fpgm': '#009E73', 'hrank': '#CC79A7'}
MARKERS = {'gra': 'o', 'l1': 's', 'fpgm': '^', 'hrank': 'D'}
LABELS = {'gra': 'GRA-CNN', 'l1': 'L1-Norm', 'fpgm': 'FPGM', 'hrank': 'HRank'}

# 设置样式
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

OUTPUT_DIR = r'C:\GRA-CNN\APIN_Submission'

def create_12panel_figure():
    """创建12面板主结果图"""
    panels = [
        ('resnet20', 'cifar10'), ('resnet56', 'cifar10'), 
        ('resnet110', 'cifar10'), ('vgg16', 'cifar10'),
        ('resnet20', 'cifar100'), ('resnet56', 'cifar100'), 
        ('resnet110', 'cifar100'), ('vgg16', 'cifar100'),
        ('resnet20', 'cifar10'), ('resnet56', 'cifar10'), 
        ('resnet20', 'cifar100'), ('resnet56', 'cifar100'),
    ]
    
    fig = plt.figure(figsize=(7.5, 6))
    gs = gridspec.GridSpec(3, 4, hspace=0.38, wspace=0.32)
    
    for idx, (arch, dataset) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        
        for method in ['gra', 'l1', 'fpgm', 'hrank']:
            subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & (df['method']==method)]
            if len(subset) > 0:
                subset = subset.sort_values('ratio')
                label = LABELS[method] if idx == 0 else ''
                ax.plot(subset['ratio'], subset['pruned_acc'], 
                       marker=MARKERS[method], color=COLORS[method],
                       label=label, linewidth=1.5, markersize=4, 
                       markeredgecolor='white', markeredgewidth=0.5)
        
        # 标题
        title_arch = arch.replace('resnet', 'ResNet-').replace('vgg', 'VGG-')
        ax.set_title(f'({chr(97+idx)}) {title_arch}', fontsize=8, fontweight='bold')
        
        if idx % 4 == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=7)
        if idx >= 8:
            ax.set_xlabel('Pruning Ratio', fontsize=7)
        
        ax.set_xlim(0.25, 0.75)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    
    # 图例
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=7, 
               bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle('Pruning Performance: GRA-CNN vs Baseline Methods', 
                fontsize=10, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_12panel.pdf'), 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_12panel.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    print('Created: fig_nature_12panel.pdf/png')
    plt.close()

def create_bar_comparison():
    """创建柱状对比图"""
    configs = [
        ('resnet20', 'cifar10', 'R20/C10'),
        ('resnet56', 'cifar10', 'R56/C10'),
        ('vgg16', 'cifar10', 'VGG/C10'),
        ('resnet20', 'cifar100', 'R20/C100'),
        ('resnet56', 'cifar100', 'R56/C100'),
        ('vgg16', 'cifar100', 'VGG/C100'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.5))
    
    for idx, (arch, dataset, title) in enumerate(configs):
        ax = axes[idx // 3, idx % 3]
        
        ratios = [0.3, 0.5, 0.7]
        x = np.arange(len(ratios))
        width = 0.2
        
        for i, method in enumerate(['gra', 'l1', 'fpgm', 'hrank']):
            accs = []
            for ratio in ratios:
                subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                           (df['method']==method) & (df['ratio']==ratio)]
                accs.append(subset['pruned_acc'].values[0] if len(subset) > 0 else 0)
            
            label = LABELS[method] if idx == 0 else ''
            ax.bar(x + i*width - 0.3, accs, width, label=label, 
                  color=COLORS[method], edgecolor='white', linewidth=0.3)
        
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['30%', '50%', '70%'])
        
        if idx % 3 == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=7)
        if idx >= 3:
            ax.set_xlabel('Pruning Ratio', fontsize=7)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=6, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_bars.pdf'), 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_bars.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    print('Created: fig_nature_bars.pdf/png')
    plt.close()

def create_advantage_figure():
    """创建GRA优势分析图"""
    fig = plt.figure(figsize=(7.5, 3))
    
    # 左图: GRA vs L1 散点图
    ax1 = fig.add_subplot(121)
    
    comparisons = []
    for _, row in df[df['method'] == 'gra'].iterrows():
        l1_match = df[(df['architecture'] == row['architecture']) &
                     (df['dataset'] == row['dataset']) &
                     (df['method'] == 'l1') &
                     (df['ratio'] == row['ratio'])]
        if len(l1_match) > 0:
            comparisons.append({
                'gra': row['pruned_acc'],
                'l1': l1_match['pruned_acc'].values[0]
            })
    
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        ax1.scatter(comp_df['l1'], comp_df['gra'], c=COLORS['gra'], s=40, 
                   alpha=0.7, edgecolors='white', linewidths=0.5)
        
        lims = [min(comp_df['l1'].min(), comp_df['gra'].min()) - 2,
                max(comp_df['l1'].max(), comp_df['gra'].max()) + 2]
        ax1.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        
        mean_imp = (comp_df['gra'] - comp_df['l1']).mean()
        ax1.annotate(f'Mean improvement: {mean_imp:+.2f}%', xy=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=6, fontweight='bold',
                    color=COLORS['gra'])
    
    ax1.set_xlabel('L1-Norm Accuracy (%)', fontsize=7)
    ax1.set_ylabel('GRA-CNN Accuracy (%)', fontsize=7)
    ax1.set_title('(a) GRA-CNN vs L1-Norm', fontsize=8, fontweight='bold')
    
    # 右图: 改进柱状图
    ax2 = fig.add_subplot(122)
    
    improvements = []
    for arch in ['resnet20', 'resnet56', 'vgg16']:
        for dataset in ['cifar10', 'cifar100']:
            for ratio in [0.3, 0.5, 0.7]:
                gra = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                        (df['method']=='gra') & (df['ratio']==ratio)]
                l1 = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                       (df['method']=='l1') & (df['ratio']==ratio)]
                if len(gra) > 0 and len(l1) > 0:
                    imp = gra['pruned_acc'].values[0] - l1['pruned_acc'].values[0]
                    improvements.append({
                        'config': f'{arch[:3]}/{dataset[-2:]}/{int(ratio*100)}%',
                        'imp': imp
                    })
    
    if improvements:
        imp_df = pd.DataFrame(improvements)
        colors_bar = [COLORS['gra'] if v > 0 else '#999999' for v in imp_df['imp']]
        ax2.bar(range(len(imp_df)), imp_df['imp'], color=colors_bar, 
               edgecolor='white', linewidth=0.3)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axhline(imp_df['imp'].mean(), color=COLORS['gra'], linestyle='--',
                   linewidth=1, label=f"Mean: {imp_df['imp'].mean():.2f}%")
        ax2.set_xticks(range(len(imp_df)))
        ax2.set_xticklabels(imp_df['config'], fontsize=5, rotation=45, ha='right')
        ax2.legend(fontsize=5)
    
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=7)
    ax2.set_title('(b) GRA Improvement over L1', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_advantage.pdf'),
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_advantage.png'),
               dpi=300, bbox_inches='tight', facecolor='white')
    print('Created: fig_nature_advantage.pdf/png')
    plt.close()

if __name__ == '__main__':
    print('='*60)
    print('Nature/Science Style Figure Generator')
    print('='*60)
    
    create_12panel_figure()
    create_bar_comparison()
    create_advantage_figure()
    
    print('\nAll figures generated successfully!')
