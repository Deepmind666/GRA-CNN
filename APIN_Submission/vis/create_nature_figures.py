"""
顶级期刊图表生成器 - Nature/Science风格
=========================================
特点:
1. 精致的配色方案 (Nature调色板)
2. 精确的字体和间距控制
3. 专业的标注和图例
4. 高分辨率输出 (600 DPI)
5. 使用真实实验数据
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

# ============================================================================
# 加载真实实验数据
# ============================================================================

df = pd.read_csv(r'C:\GRA-CNN\experiments\all_experiment_results.csv')
print(f"加载 {len(df)} 条真实实验数据")

# ============================================================================
# Nature期刊配色方案
# ============================================================================

NATURE_COLORS = {
    'gra': '#DC3220',      # 深红色 (主角方法)
    'l1': '#0072B2',       # 深蓝色
    'fpgm': '#009E73',     # 绿色  
    'hrank': '#CC79A7',    # 粉紫色
    'baseline': '#999999', # 灰色
}

MARKERS = {'gra': 'o', 'l1': 's', 'fpgm': '^', 'hrank': 'D'}
LINESTYLES = {'gra': '-', 'l1': '--', 'fpgm': '-.', 'hrank': ':'}
METHOD_LABELS = {'gra': 'GRA-CNN', 'l1': 'L1-Norm', 'fpgm': 'FPGM', 'hrank': 'HRank'}

def set_nature_style():
    """设置Nature期刊专业样式"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8,
        'axes.labelweight': 'medium',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        
        # 图形质量
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # 轴线设置
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # 网格设置
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '-',
        'grid.linewidth': 0.3,
        
        # 线条设置
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.6,
        'lines.markeredgecolor': 'white',
        
        # 图例设置
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'none',
        'legend.borderpad': 0.3,
    })

OUTPUT_DIR = r'C:\GRA-CNN\APIN_Submission'

# ============================================================================
# 数据提取
# ============================================================================

def get_data(arch, dataset, method):
    """提取指定配置的真实数据"""
    subset = df[
        (df['architecture'].str.lower() == arch.lower()) &
        (df['dataset'].str.lower() == dataset.lower()) &
        (df['method'].str.lower() == method.lower())
    ].copy()
    subset = subset.sort_values('ratio')
    if len(subset) > 0:
        return subset['ratio'].values, subset['pruned_acc'].values
    return np.array([]), np.array([])

def get_baseline(arch, dataset):
    """获取基线精度"""
    subset = df[
        (df['architecture'].str.lower() == arch.lower()) &
        (df['dataset'].str.lower() == dataset.lower())
    ]
    if len(subset) > 0 and 'baseline_acc' in subset.columns:
        return subset['baseline_acc'].values[0]
    return None

# ============================================================================
# 图表1: 主结果 - 12面板折线图网格
# ============================================================================

def create_main_results_figure():
    """创建Nature风格的12面板主结果图"""
    set_nature_style()
    
    # 定义12个面板配置
    panels = [
        # Row 1: CIFAR-10
        ('resnet20', 'cifar10', 'ResNet-20', 'CIFAR-10'),
        ('resnet56', 'cifar10', 'ResNet-56', 'CIFAR-10'),
        ('resnet110', 'cifar10', 'ResNet-110', 'CIFAR-10'),
        ('vgg16', 'cifar10', 'VGG-16', 'CIFAR-10'),
        # Row 2: CIFAR-100
        ('resnet20', 'cifar100', 'ResNet-20', 'CIFAR-100'),
        ('resnet56', 'cifar100', 'ResNet-56', 'CIFAR-100'),
        ('resnet110', 'cifar100', 'ResNet-110', 'CIFAR-100'),
        ('vgg16', 'cifar100', 'VGG-16', 'CIFAR-100'),
        # Row 3: 不同视角 (按方法汇总)
        ('resnet20', 'cifar10', 'ResNet-20', 'CIFAR-10 (Detail)'),
        ('resnet56', 'cifar10', 'ResNet-56', 'CIFAR-10 (Detail)'),
        ('resnet20', 'cifar100', 'ResNet-20', 'CIFAR-100 (Detail)'),
        ('resnet56', 'cifar100', 'ResNet-56', 'CIFAR-100 (Detail)'),
    ]
    
    methods = ['gra', 'l1', 'fpgm', 'hrank']
    
    fig = plt.figure(figsize=(7.2, 6))  # Nature双栏宽度
    gs = gridspec.GridSpec(3, 4, hspace=0.38, wspace=0.32,
                          left=0.08, right=0.98, top=0.92, bottom=0.08)
    
    for idx, (arch, dataset, arch_label, ds_label) in enumerate(panels):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        has_data = False
        for method in methods:
            ratios, accs = get_data(arch, dataset, method)
            if len(ratios) > 0:
                ax.plot(ratios, accs,
                       marker=MARKERS[method],
                       color=NATURE_COLORS[method],
                       linestyle=LINESTYLES[method],
                       label=METHOD_LABELS[method],
                       linewidth=1.5,
                       markersize=4,
                       markeredgecolor='white',
                       markeredgewidth=0.6,
                       zorder=3 if method == 'gra' else 2)
                has_data = True
        
        # 标题
        panel_letter = chr(97 + idx)
        ax.set_title(f'({panel_letter}) {arch_label}', fontsize=8, fontweight='bold', pad=3)
        
        # 添加数据集标签
        if row < 2:
            if col == 0:
                ax.annotate(ds_label, xy=(-0.35, 0.5), xycoords='axes fraction',
                           rotation=90, fontsize=7, fontweight='bold', va='center')
        
        # 轴标签
        if col == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=7)
        if row == 2:
            ax.set_xlabel('Pruning Ratio', fontsize=7)
        
        # X轴设置
        ax.set_xlim(0.25, 0.75)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        
        # Y轴自动范围
        if has_data:
            all_accs = []
            for method in methods:
                _, accs = get_data(arch, dataset, method)
                if len(accs) > 0:
                    all_accs.extend(accs)
            if all_accs:
                ymin = min(all_accs) - 2
                ymax = max(all_accs) + 2
                ax.set_ylim(ymin, ymax)
        
        # 图例 (只在第一个面板)
        if idx == 0:
            ax.legend(loc='lower left', fontsize=5.5, ncol=2,
                     handlelength=1.5, handletextpad=0.4, columnspacing=0.8)
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=8, color='gray', alpha=0.7)
    
    # 主标题
    fig.suptitle('Pruning Performance Comparison Across Architectures and Datasets',
                fontsize=10, fontweight='bold', y=0.98)
    
    # 保存
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_12panel.pdf'), 
               dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_12panel.png'), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✅ 已创建: fig_nature_12panel.pdf/png")

# ============================================================================
# 图表2: 方法对比柱状图
# ============================================================================

def create_method_comparison_figure():
    """创建方法对比柱状图"""
    set_nature_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    
    configs = [
        ('resnet20', 'cifar10', 'ResNet-20/C10'),
        ('resnet56', 'cifar10', 'ResNet-56/C10'),
        ('vgg16', 'cifar10', 'VGG-16/C10'),
        ('resnet20', 'cifar100', 'ResNet-20/C100'),
        ('resnet56', 'cifar100', 'ResNet-56/C100'),
        ('vgg16', 'cifar100', 'VGG-16/C100'),
    ]
    
    methods = ['gra', 'l1', 'fpgm', 'hrank']
    ratios = [0.3, 0.5, 0.7]
    
    for idx, (arch, dataset, title) in enumerate(configs):
        ax = axes[idx // 3, idx % 3]
        
        x = np.arange(len(ratios))
        width = 0.2
        
        for i, method in enumerate(methods):
            accs = []
            for ratio in ratios:
                subset = df[
                    (df['architecture'].str.lower() == arch) &
                    (df['dataset'].str.lower() == dataset) &
                    (df['method'].str.lower() == method) &
                    (df['ratio'] == ratio)
                ]
                if len(subset) > 0:
                    accs.append(subset['pruned_acc'].values[0])
                else:
                    accs.append(0)
            
            bars = ax.bar(x + i*width - 0.3, accs, width,
                         label=METHOD_LABELS[method] if idx == 0 else '',
                         color=NATURE_COLORS[method],
                         edgecolor='white', linewidth=0.3)
        
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(r*100)}%' for r in ratios])
        
        if idx % 3 == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=7)
        if idx >= 3:
            ax.set_xlabel('Pruning Ratio', fontsize=7)
    
    # 共享图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
              fontsize=6, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_bars.pdf'),
               dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_bars.png'),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✅ 已创建: fig_nature_bars.pdf/png")

# ============================================================================
# 图表3: GRA优势分析
# ============================================================================

def create_gra_advantage_figure():
    """创建GRA相对于其他方法的优势分析图"""
    set_nature_style()
    
    fig = plt.figure(figsize=(7.2, 3))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)
    
    # 左图: 散点图 (GRA vs L1)
    ax1 = fig.add_subplot(gs[0, 0])
    
    comparisons = []
    for _, row in df.iterrows():
        if row['method'].lower() == 'gra':
            l1_match = df[
                (df['architecture'] == row['architecture']) &
                (df['dataset'] == row['dataset']) &
                (df['method'].str.lower() == 'l1') &
                (df['ratio'] == row['ratio'])
            ]
            if len(l1_match) > 0:
                comparisons.append({
                    'gra_acc': row['pruned_acc'],
                    'l1_acc': l1_match['pruned_acc'].values[0],
                    'arch': row['architecture'],
                    'ratio': row['ratio']
                })
    
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        
        scatter = ax1.scatter(comp_df['l1_acc'], comp_df['gra_acc'],
                             c=NATURE_COLORS['gra'], s=40, alpha=0.7,
                             edgecolors='white', linewidths=0.5)
        
        # 对角线
        lim_min = min(comp_df['l1_acc'].min(), comp_df['gra_acc'].min()) - 1
        lim_max = max(comp_df['l1_acc'].max(), comp_df['gra_acc'].max()) + 1
        ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.8, alpha=0.5)
        
        ax1.set_xlabel('L1-Norm Accuracy (%)', fontsize=7)
        ax1.set_ylabel('GRA-CNN Accuracy (%)', fontsize=7)
        ax1.set_title('(a) GRA-CNN vs L1-Norm', fontsize=8, fontweight='bold')
        
        # 添加均值改进标注
        mean_imp = (comp_df['gra_acc'] - comp_df['l1_acc']).mean()
        ax1.annotate(f'Mean Δ = {mean_imp:+.2f}%',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=6, fontweight='bold', color=NATURE_COLORS['gra'])
    
    # 右图: 改进柱状图
    ax2 = fig.add_subplot(gs[0, 1])
    
    improvements = []
    for arch in df['architecture'].unique():
        for dataset in df['dataset'].unique():
            for ratio in [0.3, 0.5, 0.7]:
                gra_data = df[
                    (df['architecture'] == arch) &
                    (df['dataset'] == dataset) &
                    (df['method'].str.lower() == 'gra') &
                    (df['ratio'] == ratio)
                ]
                l1_data = df[
                    (df['architecture'] == arch) &
                    (df['dataset'] == dataset) &
                    (df['method'].str.lower() == 'l1') &
                    (df['ratio'] == ratio)
                ]
                if len(gra_data) > 0 and len(l1_data) > 0:
                    imp = gra_data['pruned_acc'].values[0] - l1_data['pruned_acc'].values[0]
                    improvements.append({
                        'config': f'{arch[:3]}/{dataset[-2:]}/{int(ratio*100)}%',
                        'improvement': imp
                    })
    
    if improvements:
        imp_df = pd.DataFrame(improvements)
        colors = [NATURE_COLORS['gra'] if v > 0 else '#999999' for v in imp_df['improvement']]
        
        bars = ax2.bar(range(len(imp_df)), imp_df['improvement'],
                      color=colors, edgecolor='white', linewidth=0.3)
        
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axhline(imp_df['improvement'].mean(), color=NATURE_COLORS['gra'],
                   linestyle='--', linewidth=1, label=f"Mean: {imp_df['improvement'].mean():.2f}%")
        
        ax2.set_xticks(range(len(imp_df)))
        ax2.set_xticklabels(imp_df['config'], fontsize=5, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy Improvement (%)', fontsize=7)
        ax2.set_title('(b) GRA-CNN Improvement over L1', fontsize=8, fontweight='bold')
        ax2.legend(fontsize=5, loc='upper right')
    
    plt.tight_layout()
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_advantage.pdf'),
               dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_nature_advantage.png'),
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✅ 已创建: fig_nature_advantage.pdf/png")

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("="*60)
    print("顶级期刊图表生成器 - Nature/Science风格")
    print("="*60)
    print(f"数据: {len(df)} 条实验记录")
    print()
    
    create_main_results_figure()
    create_method_comparison_figure()
    create_gra_advantage_figure()
    
    print("\n" + "="*60)
    print("所有顶级期刊风格图表生成完成!")
    print("="*60)

if __name__ == '__main__':
    main()
