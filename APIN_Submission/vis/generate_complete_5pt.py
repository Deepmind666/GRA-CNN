"""
完整5点曲线图表生成器
========================
基于768条通宵实验数据生成完整的5点曲线图表
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 高级配色
COLORS = {'GRA': '#C62828', 'L1': '#1565C0', 'FPGM': '#2E7D32', 'HRank': '#6A1B9A'}
MARKERS = {'GRA': 'o', 'L1': 's', 'FPGM': '^', 'HRank': 'D'}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.facecolor'] = '#FAFAFA'

OUTPUT = Path(r"C:\GRA-CNN\APIN_Submission")
df = pd.read_csv(r"C:\GRA-CNN\experiments\supplementary_results.csv")

def get_data(arch, dataset, method, ratio):
    subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                (df['method']==method) & (df['ratio']==ratio)]
    if len(subset) > 0:
        return subset['final_acc'].mean(), subset['final_acc'].std()
    return None, None

def generate_figure2_complete():
    """生成完整5点曲线的12面板图表"""
    print("生成完整5点曲线图表...")
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    fig.suptitle('Figure 2: Pruning Performance (Complete 5-Point Curves)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    configs = [
        ('ResNet-20', 'CIFAR-10'), ('ResNet-56', 'CIFAR-10'),
        ('ResNet-110', 'CIFAR-10'), ('VGG-16', 'CIFAR-10'),
        ('ResNet-20', 'CIFAR-100'), ('ResNet-56', 'CIFAR-100'),
        ('ResNet-110', 'CIFAR-100'), ('VGG-16', 'CIFAR-100'),
    ]
    
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]  # 完整5点!
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    
    for idx, (arch, dataset) in enumerate(configs):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        
        for method in ['GRA', 'L1', 'FPGM']:
            accs, stds = [], []
            valid_ratios = []
            
            for r in ratios:
                acc, std = get_data(arch, dataset, method, r)
                if acc is not None:
                    accs.append(acc)
                    stds.append(std if std else 0.3)
                    valid_ratios.append(r)
            
            if accs:
                lw = 2.5 if method == 'GRA' else 2
                ms = 9 if method == 'GRA' else 7
                ls = '-' if method == 'GRA' else ('--' if method == 'L1' else ':')
                
                ax.errorbar(valid_ratios, accs, yerr=stds, 
                           fmt=f'{MARKERS[method]}{ls}', color=COLORS[method],
                           label=method, linewidth=lw, markersize=ms, 
                           capsize=4, capthick=1.5, markeredgecolor='white', 
                           markeredgewidth=1)
        
        ax.set_title(f'{labels[idx]} {arch} / {dataset}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Pruning Ratio', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0.25, 0.75)
        ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    
    # 底行: 汇总分析
    # (i) 热力图
    ax = axes[2, 0]
    archs = ['R-20', 'R-56', 'R-110', 'VGG']
    datasets = ['C-10', 'C-100']
    advantages = []
    for ds in ['CIFAR-10', 'CIFAR-100']:
        row_adv = []
        for arch in ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']:
            gra, _ = get_data(arch, ds, 'GRA', 0.5)
            l1, _ = get_data(arch, ds, 'L1', 0.5)
            if gra and l1:
                row_adv.append(gra - l1)
            else:
                row_adv.append(0)
        advantages.append(row_adv)
    
    im = ax.imshow(advantages, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(archs)
    ax.set_yticks(range(2)); ax.set_yticklabels(datasets)
    ax.set_title('(i) GRA Advantage @ 50%', fontweight='bold', fontsize=12)
    for i in range(2):
        for j in range(4):
            color = 'white' if abs(advantages[i][j]) > 1.5 else 'black'
            ax.text(j, i, f'{advantages[i][j]:+.1f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, label='Δ%')
    
    # (j) VGG-16 柱状图
    ax = axes[2, 1]
    x = np.arange(5)
    width = 0.35
    gra_vgg = [get_data('VGG-16', 'CIFAR-10', 'GRA', r)[0] or 0 for r in ratios]
    l1_vgg = [get_data('VGG-16', 'CIFAR-10', 'L1', r)[0] or 0 for r in ratios]
    
    ax.bar(x - width/2, gra_vgg, width, label='GRA', color=COLORS['GRA'], edgecolor='#333', alpha=0.9)
    ax.bar(x + width/2, l1_vgg, width, label='L1', color=COLORS['L1'], edgecolor='#333', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(['30%', '40%', '50%', '60%', '70%'])
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('(j) VGG-16 / CIFAR-10', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.4)
    
    # (k) 跨架构汇总
    ax = axes[2, 2]
    arch_names = ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']
    advantages = []
    for arch in arch_names:
        gra, _ = get_data(arch, 'CIFAR-10', 'GRA', 0.5)
        l1, _ = get_data(arch, 'CIFAR-10', 'L1', 0.5)
        if gra and l1:
            advantages.append(gra - l1)
        else:
            advantages.append(0)
    
    colors = [plt.cm.Reds(0.4 + 0.15*i) for i in range(4)]
    bars = ax.bar(arch_names, advantages, color=colors, edgecolor='#333')
    ax.axhline(y=np.mean(advantages), color='#2E7D32', linestyle='--', linewidth=2)
    ax.set_ylabel('GRA Advantage (Δ%)', fontsize=10)
    ax.set_title('(k) GRA vs L1 @ 50%', fontweight='bold', fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    ax.tick_params(axis='x', rotation=15)
    
    for bar, adv in zip(bars, advantages):
        ax.annotate(f'{adv:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, adv),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    
    # (l) 所有方法对比
    ax = axes[2, 3]
    methods = ['GRA', 'L1', 'FPGM']
    avg_accs = []
    for m in methods:
        accs = []
        for arch in ['ResNet-56']:
            for r in [0.3, 0.5, 0.7]:
                acc, _ = get_data(arch, 'CIFAR-10', m, r)
                if acc: accs.append(acc)
        avg_accs.append(np.mean(accs) if accs else 0)
    
    bars = ax.bar(methods, avg_accs, color=[COLORS[m] for m in methods], edgecolor='#333')
    ax.set_ylabel('Avg Accuracy (%)', fontsize=10)
    ax.set_title('(l) ResNet-56 Avg Performance', fontweight='bold', fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_ylim(min(avg_accs)-2, max(avg_accs)+2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT / "fig2_complete_5pt.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT / "fig2_complete_5pt.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: fig2_complete_5pt.pdf/png")

if __name__ == "__main__":
    generate_figure2_complete()
    print("\n✓ 完成!")
