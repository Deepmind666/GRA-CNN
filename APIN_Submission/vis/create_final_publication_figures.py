"""
高质量12面板论文图表生成器
============================
基于5957条完整实验数据
严格遵循参考图表风格
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = r'C:\GRA-CNN\APIN_Submission'

# 颜色方案 (参考图表风格)
COLORS = {
    'GRA': '#D62728',      # 红色 - 主角，最亮眼
    'L1': '#1F77B4',       # 蓝色
    'FPGM': '#2CA02C',     # 绿色
    'HRANK': '#9467BD',    # 紫色
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
# 数据加载
# ============================================================================

def load_all_data():
    files = [
        r'C:\GRA-CNN\experiments\final_12panel_results.csv',
        r'C:\GRA-CNN\experiments\comprehensive_10hr_results.csv',
        r'C:\GRA-CNN\experiments\supplementary_results.csv',
        r'C:\GRA-CNN\experiments\complete_data.csv',
    ]
    
    all_dfs = []
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                all_dfs.append(df)
            except:
                pass
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # 标准化
    combined['arch'] = combined['architecture'].str.lower().str.replace('-', '')
    combined['ds'] = combined['dataset'].str.lower().str.replace('-', '')
    combined['meth'] = combined['method'].str.upper()
    
    # 精度列
    for col in ['accuracy', 'final_acc', 'acc', 'pruned_acc']:
        if col in combined.columns:
            combined['acc'] = pd.to_numeric(combined[col], errors='coerce')
            break
    
    return combined

def get_curve_data(df, arch, dataset, method):
    """提取指定配置的曲线数据，返回按ratio排序的(ratios, accs)"""
    arch_key = arch.lower().replace('-', '')
    ds_key = dataset.lower().replace('-', '')
    meth_key = method.upper()
    
    subset = df[(df['arch'] == arch_key) & 
                (df['ds'] == ds_key) & 
                (df['meth'] == meth_key) &
                (df['ratio'] > 0) &  # 排除baseline
                (df['ratio'] <= 0.7)]
    
    if len(subset) == 0:
        return np.array([]), np.array([])
    
    # 按ratio分组取平均
    grouped = subset.groupby('ratio')['acc'].mean().reset_index()
    grouped = grouped.sort_values('ratio')
    
    return grouped['ratio'].values, grouped['acc'].values

# ============================================================================
# 样式设置
# ============================================================================

def set_publication_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })

# ============================================================================
# 12面板主图
# ============================================================================

def create_12panel_figure(df):
    set_publication_style()
    
    # 面板配置 (按参考图顺序)
    panels = [
        # Row 1: CIFAR-10 小模型
        ('ResNet-20', 'CIFAR-10', 'a'),
        ('ResNet-32', 'CIFAR-10', 'b'),
        ('ResNet-44', 'CIFAR-10', 'c'),
        ('ResNet-56', 'CIFAR-10', 'd'),
        # Row 2: CIFAR-10 大模型 + CIFAR-100 小模型
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
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.25, 
                          left=0.06, right=0.98, top=0.93, bottom=0.08)
    
    for idx, (arch, dataset, letter) in enumerate(panels):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        has_any_data = False
        y_min, y_max = 100, 0
        
        for method in methods:
            ratios, accs = get_curve_data(df, arch, dataset, method)
            
            if len(ratios) >= 2:  # 至少2个点才画线
                ax.plot(ratios, accs,
                       marker=MARKERS[method],
                       color=COLORS[method],
                       linestyle=LINESTYLES[method],
                       label=LABELS[method] if idx == 0 else '',
                       linewidth=2.0,
                       markersize=6,
                       markeredgecolor='white',
                       markeredgewidth=0.8,
                       zorder=10 if method == 'GRA' else 5)
                has_any_data = True
                y_min = min(y_min, accs.min())
                y_max = max(y_max, accs.max())
        
        # 标题
        ax.set_title(f'({letter}) {arch} on {dataset}', fontweight='bold', fontsize=9, pad=4)
        
        # 坐标轴
        ax.set_xlim(0.25, 0.75)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
        
        if has_any_data:
            margin = (y_max - y_min) * 0.15
            ax.set_ylim(y_min - margin, y_max + margin)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=9)
        if row == 2:
            ax.set_xlabel('Pruning Ratio', fontsize=9)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        if not has_any_data:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray', alpha=0.6)
    
    # 图例
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True)
    
    # 总标题
    fig.suptitle('Figure 2: Pruning Performance Across Architectures and Datasets',
                fontweight='bold', fontsize=12, y=0.98)
    
    # 保存
    plt.savefig(f'{OUTPUT_DIR}/fig2_final_12panel.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig2_final_12panel.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig2_final_12panel.pdf/png')

# ============================================================================
# GRA优势分析图
# ============================================================================

def create_advantage_figure(df):
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 收集GRA vs L1对比数据
    comparisons = []
    
    for arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'vgg16']:
        for ds in ['cifar10', 'cifar100']:
            for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
                gra_data = df[(df['arch']==arch) & (df['ds']==ds) & 
                             (df['meth']=='GRA') & (df['ratio']==ratio)]
                l1_data = df[(df['arch']==arch) & (df['ds']==ds) & 
                            (df['meth']=='L1') & (df['ratio']==ratio)]
                
                if len(gra_data) > 0 and len(l1_data) > 0:
                    gra_acc = gra_data['acc'].mean()
                    l1_acc = l1_data['acc'].mean()
                    comparisons.append({
                        'arch': arch, 'ds': ds, 'ratio': ratio,
                        'gra': gra_acc, 'l1': l1_acc,
                        'advantage': gra_acc - l1_acc
                    })
    
    if comparisons:
        comp_df = pd.DataFrame(comparisons)
        
        # 散点图
        ax1 = axes[0]
        ax1.scatter(comp_df['l1'], comp_df['gra'], alpha=0.6, s=30, c='#D62728')
        
        # 对角线
        lims = [min(comp_df['l1'].min(), comp_df['gra'].min()) - 1,
                max(comp_df['l1'].max(), comp_df['gra'].max()) + 1]
        ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        ax1.fill_between(lims, lims, [lims[1]]*2, alpha=0.1, color='green')
        
        ax1.set_xlabel('L1-Norm Accuracy (%)')
        ax1.set_ylabel('GRA-CNN Accuracy (%)')
        ax1.set_title('(a) GRA-CNN vs L1-Norm', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 优势直方图
        ax2 = axes[1]
        advantages = comp_df['advantage'].values
        ax2.hist(advantages, bins=20, color='#D62728', alpha=0.7, edgecolor='white')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.axvline(x=advantages.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {advantages.mean():.2f}%')
        ax2.set_xlabel('Accuracy Improvement (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('(b) GRA-CNN Advantage Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        win_rate = (advantages > 0).sum() / len(advantages) * 100
        ax2.text(0.95, 0.95, f'Win Rate: {win_rate:.1f}%', transform=ax2.transAxes,
                ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_gra_advantage.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig_gra_advantage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: fig_gra_advantage.pdf/png')

# ============================================================================
# 主程序
# ============================================================================

def main():
    print('='*70)
    print('高质量论文图表生成器')
    print('='*70)
    
    df = load_all_data()
    print(f'加载数据: {len(df)} 条记录')
    
    if len(df) == 0:
        print('无数据!')
        return
    
    create_12panel_figure(df)
    create_advantage_figure(df)
    
    print()
    print('='*70)
    print('所有图表生成完成!')
    print('='*70)

if __name__ == '__main__':
    main()
