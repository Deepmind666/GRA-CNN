"""
GRA-CNN 论文图表生成器 (基于修复后真实数据)
==========================================
基于2026-01-19通宵实验的641+条真实数据
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 学术图表设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path(r"C:\GRA-CNN\APIN_Submission")

# 加载真实数据
df = pd.read_csv(r"C:\GRA-CNN\experiments\supplementary_results.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 使用最新修复后的数据 + 之前的完整数据
def get_best_result(arch, dataset, method, ratio):
    """获取最佳结果（优先使用最新数据）"""
    subset = df[(df['architecture']==arch) & (df['dataset']==dataset) & 
                (df['method']==method) & (df['ratio']==ratio)]
    if len(subset) > 0:
        # 优先使用最新数据
        recent = subset[subset['timestamp'] > '2026-01-19 16:00:00']
        if len(recent) > 0:
            return recent['final_acc'].mean(), recent['final_acc'].std()
        return subset['final_acc'].mean(), subset['final_acc'].std()
    return None, None

def generate_figure3_improved():
    """Figure 3: Accuracy vs Pruning Ratio (改进版)"""
    print("生成 Figure 3 (Accuracy vs Ratio)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'GRA': '#E74C3C', 'L1': '#3498DB', 'FPGM': '#27AE60', 'HRank': '#9B59B6'}
    markers = {'GRA': 'o', 'L1': 's', 'FPGM': '^', 'HRank': 'D'}
    
    # (a) ResNet-56 / CIFAR-10
    ax = axes[0]
    ratios = [0.3, 0.5, 0.7]
    
    for method in ['GRA', 'L1', 'FPGM']:
        accs, stds = [], []
        for r in ratios:
            acc, std = get_best_result('ResNet-56', 'CIFAR-10', method, r)
            accs.append(acc if acc else 0)
            stds.append(std if std else 0)
        
        if any(accs):
            ax.errorbar(ratios, accs, yerr=stds, fmt=f'{markers[method]}-', 
                       color=colors[method], label=method, linewidth=2, 
                       markersize=8, capsize=4, capthick=1.5)
    
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('(a) ResNet-56 / CIFAR-10', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(86, 95)
    
    # (b) ResNet-56 / CIFAR-100
    ax = axes[1]
    for method in ['GRA', 'L1', 'FPGM']:
        accs, stds = [], []
        for r in ratios:
            acc, std = get_best_result('ResNet-56', 'CIFAR-100', method, r)
            accs.append(acc if acc else 0)
            stds.append(std if std else 0)
        
        if any(accs):
            ax.errorbar(ratios, accs, yerr=stds, fmt=f'{markers[method]}-', 
                       color=colors[method], label=method, linewidth=2, 
                       markersize=8, capsize=4, capthick=1.5)
    
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('(b) ResNet-56 / CIFAR-100', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.25, 0.75)
    
    # (c) GRA Advantage Summary
    ax = axes[2]
    configs = ['ResNet-20\nCIFAR-10', 'ResNet-56\nCIFAR-10', 'VGG-16\nCIFAR-10', 'ResNet-56\nCIFAR-100']
    advantages = []
    
    for arch, dataset in [('ResNet-20', 'CIFAR-10'), ('ResNet-56', 'CIFAR-10'), 
                          ('VGG-16', 'CIFAR-10'), ('ResNet-56', 'CIFAR-100')]:
        gra, _ = get_best_result(arch, dataset, 'GRA', 0.5)
        l1, _ = get_best_result(arch, dataset, 'L1', 0.5)
        if gra and l1:
            advantages.append(gra - l1)
        else:
            advantages.append(0)
    
    bars = ax.bar(configs, advantages, color=['#E74C3C' if a > 0 else '#3498DB' for a in advantages],
                  edgecolor='black', alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('GRA Advantage over L1 (Δ%)', fontsize=12)
    ax.set_title('(c) GRA vs L1 @ 50% Pruning', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, adv in zip(bars, advantages):
        if adv != 0:
            ax.annotate(f'{adv:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, adv),
                       xytext=(0, 5 if adv > 0 else -15), textcoords="offset points",
                       ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_final.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig3_final.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: fig3_final.pdf/png")

def generate_figure4_improved():
    """Figure 4: Cross-Architecture Comparison"""
    print("生成 Figure 4 (跨架构对比)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'GRA': '#E74C3C', 'L1': '#3498DB', 'FPGM': '#27AE60'}
    
    # (a) 不同架构在 50% 剪枝率下的表现
    ax = axes[0]
    archs = ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']
    x = np.arange(len(archs))
    width = 0.25
    
    for i, method in enumerate(['GRA', 'L1', 'FPGM']):
        accs = []
        for arch in archs:
            acc, _ = get_best_result(arch, 'CIFAR-10', method, 0.5)
            accs.append(acc if acc else 0)
        
        if any(accs):
            ax.bar(x + i*width, accs, width, label=method, color=colors[method], 
                   edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(archs)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('(a) Cross-Architecture @ 50% Pruning', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(85, 95)
    
    # (b) 不同剪枝率下 GRA 的保持率
    ax = axes[1]
    ratios = [0.3, 0.5, 0.7]
    
    # 计算精度保持率 (相对于基线)
    baseline = 93.5  # ResNet-56 baseline
    for method in ['GRA', 'L1', 'FPGM']:
        retention = []
        for r in ratios:
            acc, _ = get_best_result('ResNet-56', 'CIFAR-10', method, r)
            if acc:
                retention.append(acc / baseline * 100)
            else:
                retention.append(0)
        
        if any(retention):
            ax.plot(ratios, retention, f'o-', color=colors[method], label=method,
                   linewidth=2, markersize=8)
    
    ax.set_xlabel('Pruning Ratio', fontsize=12)
    ax.set_ylabel('Accuracy Retention (%)', fontsize=12)
    ax.set_title('(b) Accuracy Retention (ResNet-56)', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Baseline')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_final.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig4_final.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存: fig4_final.pdf/png")

def generate_summary_table():
    """生成论文表格数据"""
    print("\n=== 论文表格数据 ===\n")
    
    print("Table: ResNet-56 / CIFAR-10 Pruning Results")
    print("-" * 60)
    print(f"{'Ratio':<10} {'GRA':<15} {'L1':<15} {'FPGM':<15}")
    print("-" * 60)
    
    for ratio in [0.3, 0.5, 0.7]:
        row = f"{int(ratio*100)}%{'':<7}"
        for method in ['GRA', 'L1', 'FPGM']:
            acc, std = get_best_result('ResNet-56', 'CIFAR-10', method, ratio)
            if acc:
                row += f"{acc:.2f}±{std:.2f}{'':>5}"
            else:
                row += f"N/A{'':>11}"
        print(row)
    
    print("-" * 60)

if __name__ == "__main__":
    print("="*60)
    print("GRA-CNN 论文图表生成 (基于修复后真实数据)")
    print("="*60)
    
    generate_figure3_improved()
    generate_figure4_improved()
    generate_summary_table()
    
    print("\n✓ 所有图表生成完成!")
