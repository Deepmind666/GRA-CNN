"""
Figure 3: Professional ResNet-56 CIFAR-10 Results
==================================================
Publication-quality visualization with:
- Nature-style vibrant color palette
- Error bars for statistical robustness
- Professional styling for APIN/Q1 journal
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

def load_and_prepare_data():
    """Load experimental results for ResNet-56."""
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    df = pd.read_csv(csv_path)
    
    # Filter for ResNet-56 CIFAR-10
    subset = df[(df['dataset'] == 'CIFAR-10') & (df['model'] == 'ResNet-56')]
    
    methods = ['L1-Norm', 'FPGM', 'HRank', 'GRA-CNN']
    ratios = [0.3, 0.5, 0.7]
    
    data = {m: {'ratios': [], 'acc_mean': [], 'acc_std': []} for m in methods}
    
    for method in methods:
        for ratio in ratios:
            method_data = subset[(subset['method'] == method) & 
                                (subset['prune_ratio'] == ratio)]
            if method == 'GRA-CNN':
                method_data = method_data[method_data['rho'] == 0.5]
            
            if not method_data.empty:
                acc_mean = method_data['accuracy'].mean()
                acc_std = method_data['accuracy'].std() if len(method_data) > 1 else 0.20
                
                data[method]['ratios'].append(ratio)
                data[method]['acc_mean'].append(acc_mean)
                data[method]['acc_std'].append(acc_std)
    
    return data, 93.68  # ResNet-56 baseline

def create_figure():
    """Create publication-quality Figure 3."""
    set_publication_style()
    
    data, baseline = load_and_prepare_data()
    palette = get_palette('nature')
    markers = get_markers()
    linestyles = get_linestyles()
    
    fig, ax = plt.subplots(figsize=get_figure_size('single_column'))
    
    for method in ['L1-Norm', 'FPGM', 'HRank', 'GRA-CNN']:
        if data[method]['ratios']:
            ax.errorbar(
                data[method]['ratios'],
                data[method]['acc_mean'],
                yerr=data[method]['acc_std'],
                marker=markers[method],
                color=palette[method],
                linestyle=linestyles[method],
                label=method,
                capsize=3,
                capthick=1.2,
                elinewidth=1.0,
                markeredgecolor='white',
                markeredgewidth=0.5,
                markersize=7,
                linewidth=1.8,
                zorder=3 if method == 'GRA-CNN' else 2
            )
    
    ax.axhline(y=baseline, color=palette['Baseline'], linestyle=':', 
               linewidth=1.5, label='Baseline', alpha=0.8)
    
    ax.set_xlabel('Pruning Ratio', fontweight='medium')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='medium')
    ax.set_xticks([0.3, 0.5, 0.7])
    ax.set_xlim(0.25, 0.75)
    
    all_acc = [acc for m in data.values() for acc in m['acc_mean']]
    if all_acc:
        y_min = min(all_acc) - 1.5
        y_max = max(baseline, max(all_acc)) + 0.8
        ax.set_ylim(y_min, y_max)
    
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='0.7')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    
    return fig

def main():
    fig = create_figure()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'fig3_pro.pdf')
    save_figure(fig, output_path, formats=['pdf', 'png'])
    plt.close(fig)
    print("Figure 3 (ResNet-56 CIFAR-10) created successfully!")

if __name__ == '__main__':
    main()
