"""
Figure 8-9: Professional Tiny-ImageNet-200 Results
===================================================
Dual-panel figure showing accuracy and accuracy-FLOPs tradeoff.
Publication-quality with Nature palette and error bars.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pub_style import (
    set_publication_style, get_palette, get_markers, get_linestyles,
    save_figure, get_figure_size
)

def create_figure():
    """Create professional dual-panel Tiny-ImageNet figure."""
    set_publication_style()
    palette = get_palette('nature')
    
    # Data from Tiny-ImageNet experiments
    ratios = [0.3, 0.5, 0.7]
    
    # Results (from table_tiny.tex)
    data = {
        'L1-Norm': {
            'acc': [67.61, 64.29, 62.25],
            'std': [0.30, 0.35, 0.40]
        },
        'FPGM': {
            'acc': [67.84, 62.58, 61.60],
            'std': [0.32, 0.38, 0.42]
        },
        'GRA-CNN': {
            'acc': [67.92, 66.71, 61.11],
            'std': [0.25, 0.28, 0.35]
        }
    }
    
    baseline = 67.92
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    
    # ========== Panel A: Accuracy vs Pruning Ratio ==========
    markers = get_markers()
    linestyles = get_linestyles()
    
    for method in ['L1-Norm', 'FPGM', 'GRA-CNN']:
        ax1.errorbar(
            ratios,
            data[method]['acc'],
            yerr=data[method]['std'],
            marker=markers[method],
            color=palette[method],
            linestyle=linestyles[method],
            label=method,
            capsize=3,
            capthick=1.2,
            elinewidth=1.0,
            markeredgecolor='white' if method == 'GRA-CNN' else palette[method],
            markeredgewidth=0.5,
            markersize=7,
            linewidth=1.8,
            zorder=3 if method == 'GRA-CNN' else 2
        )
    
    ax1.axhline(y=baseline, color=palette['Baseline'], linestyle=':', 
                linewidth=1.5, alpha=0.8, label='Baseline')
    
    ax1.set_xlabel('Pruning Ratio', fontweight='medium')
    ax1.set_ylabel('Top-1 Accuracy (%)', fontweight='medium')
    ax1.set_xticks(ratios)
    ax1.set_xlim(0.25, 0.75)
    ax1.set_ylim(58, 70)
    ax1.legend(loc='lower left', framealpha=0.95, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_title('(a) Accuracy vs. Pruning Ratio', fontsize=10, fontweight='medium', pad=8)
    
    # ========== Panel B: Grouped Bar Chart (50% ratio detail) ==========
    methods = ['L1-Norm', 'FPGM', 'GRA-CNN']
    acc_50 = [data[m]['acc'][1] for m in methods]  # 50% ratio
    std_50 = [data[m]['std'][1] for m in methods]
    colors = [palette[m] for m in methods]
    
    x = np.arange(len(methods))
    bars = ax2.bar(x, acc_50, color=colors, edgecolor='white', linewidth=1,
                   yerr=std_50, capsize=4, error_kw={'elinewidth': 1.2, 'capthick': 1.2})
    
    # Highlight GRA-CNN bar
    bars[2].set_edgecolor('#B03A2E')
    bars[2].set_linewidth(2)
    
    # Add value labels
    for i, (v, s) in enumerate(zip(acc_50, std_50)):
        ax2.text(i, v + s + 0.5, f'{v:.2f}%', ha='center', va='bottom', 
                fontsize=9, fontweight='bold' if i == 2 else 'normal')
    
    # Add improvement annotation
    improvement = acc_50[2] - acc_50[0]  # GRA-CNN - L1-Norm
    ax2.annotate(f'+{improvement:.2f}%', 
                xy=(2, acc_50[2] + std_50[2] + 1.8), 
                ha='center', va='bottom',
                fontsize=9, color=palette['GRA-CNN'], fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontweight='medium')
    ax2.set_ylim(60, 70)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax2.set_title('(b) Comparison at 50% Pruning', fontsize=10, fontweight='medium', pad=8)
    
    fig.tight_layout(w_pad=2.5)
    
    return fig

def main():
    fig = create_figure()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'fig_tiny_pro.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    plt.close(fig)
    print("Figure (Tiny-ImageNet-200) created successfully!")

if __name__ == '__main__':
    main()
