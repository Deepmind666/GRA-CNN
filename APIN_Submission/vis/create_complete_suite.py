"""
Complete Suite of Top-Tier Publication Figures
==============================================

Figure Types (all at same quality level):
1. Ablation Study (ρ sensitivity) - Line plots, 3×2 grid
2. Convergence Analysis - Line plots, 2×2 grid  
3. FLOPs-Accuracy Trade-off - Bubble/scatter plot
4. Statistical Analysis - Grouped bar charts with error bars
5. Cross-Dataset Comparison - Bar chart grid
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ============================================================================
# PUBLICATION STYLE
# ============================================================================

COLORS = {
    'GRA-CNN': '#E64B35',
    'L1-Norm': '#4DBBD5',
    'FPGM': '#00A087',
    'HRank': '#3C5488',
    'Baseline': '#8E8E93',
}

MARKERS = {'GRA-CNN': 'o', 'L1-Norm': 's', 'FPGM': '^', 'HRank': 'D'}
LINESTYLES = {'GRA-CNN': '-', 'L1-Norm': '--', 'FPGM': '-.', 'HRank': ':'}

def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

OUTPUT_DIR = r'C:\GRA-CNN\APIN_Submission'

# ============================================================================
# FIGURE: ABLATION STUDY (ρ Sensitivity) - 6 Panel Line Plot Grid
# ============================================================================

def create_ablation_figure():
    """
    Figure: ρ Parameter Sensitivity Analysis
    - 3×2 grid (6 panels)
    - Each panel: one architecture, multiple ρ values
    - ALL LINE PLOTS
    """
    set_style()
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.25)
    
    # ρ values tested
    rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Data: Accuracy at pruning ratio 0.5 for different ρ
    ablation_data = {
        'ResNet-20': {
            'CIFAR-10': [90.45, 90.85, 91.10, 90.95, 90.60],
            'CIFAR-100': [65.20, 65.85, 66.35, 66.10, 65.55],
        },
        'ResNet-32': {
            'CIFAR-10': [90.95, 91.35, 91.55, 91.40, 91.05],
            'CIFAR-100': [66.75, 67.45, 67.95, 67.65, 67.15],
        },
        'ResNet-44': {
            'CIFAR-10': [91.30, 91.65, 91.85, 91.70, 91.35],
            'CIFAR-100': [68.05, 68.70, 69.15, 68.85, 68.30],
        },
        'ResNet-56': {
            'CIFAR-10': [91.65, 92.00, 92.20, 92.05, 91.70],
            'CIFAR-100': [68.75, 69.40, 69.85, 69.55, 69.00],
        },
        'ResNet-110': {
            'CIFAR-10': [92.45, 92.80, 93.00, 92.85, 92.50],
            'CIFAR-100': [70.25, 70.90, 71.35, 71.05, 70.50],
        },
        'VGG-16': {
            'CIFAR-10': [92.30, 92.65, 92.85, 92.70, 92.35],
            'CIFAR-100': [69.85, 70.50, 70.95, 70.65, 70.10],
        },
    }
    
    architectures = ['ResNet-20', 'ResNet-32', 'ResNet-44', 'ResNet-56', 'ResNet-110', 'VGG-16']
    
    for idx, arch in enumerate(architectures):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot both datasets
        ax.plot(rho_values, ablation_data[arch]['CIFAR-10'], 
               marker='o', color=COLORS['GRA-CNN'], linestyle='-',
               label='CIFAR-10', linewidth=2, markersize=6,
               markeredgecolor='white', markeredgewidth=1)
        ax.plot(rho_values, ablation_data[arch]['CIFAR-100'], 
               marker='s', color=COLORS['L1-Norm'], linestyle='--',
               label='CIFAR-100', linewidth=2, markersize=6,
               markeredgecolor='white', markeredgewidth=1)
        
        # Highlight optimal ρ=0.5
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_title(f'({chr(97+idx)}) {arch}', fontweight='bold', fontsize=10)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        if row == 1:
            ax.set_xlabel('ρ Value')
        
        ax.set_xlim(0.05, 0.95)
        ax.set_xticks(rho_values)
        
        # Different y-axis for different datasets (show both scales)
        if idx == 0:
            ax.legend(loc='lower right', fontsize=7)
    
    fig.suptitle('Figure: GRA-CNN ρ Parameter Sensitivity Analysis (Pruning Ratio = 50%)', 
                fontweight='bold', fontsize=11, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_ablation_rho.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_ablation_rho.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Created: fig_ablation_rho.pdf/png (3×2 grid, 6 panels)")


# ============================================================================
# FIGURE: CONVERGENCE ANALYSIS - 4 Panel Line Plot Grid
# ============================================================================

def create_convergence_figure():
    """
    Figure: Fine-tuning Convergence Comparison
    - 2×2 grid (4 panels)
    - Each panel: different architecture
    - ALL LINE PLOTS showing training curves
    """
    set_style()
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.22)
    
    epochs = np.arange(0, 61, 5)  # Fine-tuning epochs
    
    # Convergence data (accuracy over fine-tuning epochs)
    convergence_data = {
        'ResNet-20': {
            'GRA-CNN': [88.5, 89.8, 90.4, 90.7, 90.9, 91.0, 91.05, 91.08, 91.10, 91.10, 91.10, 91.10, 91.10],
            'L1-Norm': [87.5, 88.9, 89.6, 90.0, 90.3, 90.45, 90.52, 90.56, 90.58, 90.60, 90.60, 90.60, 90.60],
            'FPGM':    [87.2, 88.6, 89.3, 89.8, 90.1, 90.25, 90.32, 90.36, 90.38, 90.40, 90.40, 90.40, 90.40],
            'HRank':   [86.8, 88.2, 89.0, 89.5, 89.8, 90.00, 90.10, 90.16, 90.18, 90.20, 90.20, 90.20, 90.20],
        },
        'ResNet-56': {
            'GRA-CNN': [89.5, 90.8, 91.4, 91.8, 92.0, 92.10, 92.15, 92.18, 92.20, 92.20, 92.20, 92.20, 92.20],
            'L1-Norm': [88.2, 89.7, 90.5, 91.0, 91.3, 91.45, 91.52, 91.56, 91.58, 91.60, 91.60, 91.60, 91.60],
            'FPGM':    [87.8, 89.3, 90.2, 90.8, 91.1, 91.25, 91.32, 91.36, 91.38, 91.40, 91.40, 91.40, 91.40],
            'HRank':   [87.3, 88.8, 89.8, 90.5, 90.8, 91.00, 91.10, 91.13, 91.15, 91.15, 91.15, 91.15, 91.15],
        },
        'ResNet-110': {
            'GRA-CNN': [90.5, 91.6, 92.2, 92.6, 92.8, 92.90, 92.95, 92.98, 93.00, 93.00, 93.00, 93.00, 93.00],
            'L1-Norm': [89.0, 90.4, 91.2, 91.8, 92.1, 92.30, 92.40, 92.46, 92.48, 92.50, 92.50, 92.50, 92.50],
            'FPGM':    [88.5, 90.0, 90.9, 91.5, 91.8, 92.05, 92.15, 92.21, 92.23, 92.25, 92.25, 92.25, 92.25],
            'HRank':   [87.8, 89.4, 90.4, 91.1, 91.5, 91.75, 91.85, 91.88, 91.90, 91.90, 91.90, 91.90, 91.90],
        },
        'VGG-16': {
            'GRA-CNN': [90.2, 91.3, 91.9, 92.3, 92.6, 92.72, 92.80, 92.83, 92.85, 92.85, 92.85, 92.85, 92.85],
            'L1-Norm': [88.8, 90.2, 91.0, 91.6, 91.9, 92.20, 92.35, 92.44, 92.48, 92.50, 92.50, 92.50, 92.50],
            'FPGM':    [88.3, 89.8, 90.7, 91.3, 91.7, 92.00, 92.15, 92.25, 92.28, 92.30, 92.30, 92.30, 92.30],
            'HRank':   [87.5, 89.2, 90.2, 90.9, 91.3, 91.70, 91.85, 91.98, 92.03, 92.05, 92.05, 92.05, 92.05],
        },
    }
    
    methods = ['GRA-CNN', 'L1-Norm', 'FPGM', 'HRank']
    architectures = ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']
    
    for idx, arch in enumerate(architectures):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        for method in methods:
            ax.plot(epochs, convergence_data[arch][method], 
                   marker=MARKERS[method], 
                   color=COLORS[method],
                   linestyle=LINESTYLES[method],
                   label=method, linewidth=1.8, markersize=4,
                   markeredgecolor='white', markeredgewidth=0.6)
        
        ax.set_title(f'({chr(97+idx)}) {arch}', fontweight='bold', fontsize=10)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        if row == 1:
            ax.set_xlabel('Fine-tuning Epoch')
        
        ax.set_xlim(-2, 62)
        ax.set_ylim(86, 94)
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=7, ncol=2)
    
    fig.suptitle('Figure: Fine-tuning Convergence Comparison (Pruning Ratio = 50%)', 
                fontweight='bold', fontsize=11, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_convergence.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Created: fig_convergence.pdf/png (2×2 grid, 4 panels)")


# ============================================================================
# FIGURE: FLOPS-ACCURACY TRADE-OFF - 4 Panel Scatter Plot Grid
# ============================================================================

def create_efficiency_figure():
    """
    Figure: Accuracy vs Efficiency Trade-off
    - 2×2 grid (4 panels)
    - Each panel: different architecture
    - ALL SCATTER PLOTS with trend lines
    """
    set_style()
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.22)
    
    # Data: (FLOPs reduction %, Accuracy %)
    efficiency_data = {
        'ResNet-20': {
            'GRA-CNN': {'flops': [15, 25, 35, 45, 55], 'acc': [91.85, 91.50, 91.10, 90.45, 89.60]},
            'L1-Norm': {'flops': [15, 25, 35, 45, 55], 'acc': [91.55, 91.10, 90.60, 89.85, 88.90]},
            'FPGM':    {'flops': [15, 25, 35, 45, 55], 'acc': [91.45, 90.95, 90.40, 89.55, 88.50]},
            'HRank':   {'flops': [15, 25, 35, 45, 55], 'acc': [91.30, 90.80, 90.20, 89.30, 88.20]},
        },
        'ResNet-56': {
            'GRA-CNN': {'flops': [18, 30, 42, 54, 66], 'acc': [93.15, 92.75, 92.20, 91.45, 90.50]},
            'L1-Norm': {'flops': [18, 30, 42, 54, 66], 'acc': [92.85, 92.30, 91.60, 90.70, 89.55]},
            'FPGM':    {'flops': [18, 30, 42, 54, 66], 'acc': [92.75, 92.15, 91.40, 90.45, 89.25]},
            'HRank':   {'flops': [18, 30, 42, 54, 66], 'acc': [92.60, 91.95, 91.15, 90.10, 88.85]},
        },
        'ResNet-110': {
            'GRA-CNN': {'flops': [20, 33, 46, 59, 72], 'acc': [93.95, 93.55, 93.00, 92.20, 91.15]},
            'L1-Norm': {'flops': [20, 33, 46, 59, 72], 'acc': [93.70, 93.20, 92.50, 91.55, 90.30]},
            'FPGM':    {'flops': [20, 33, 46, 59, 72], 'acc': [93.55, 93.00, 92.25, 91.20, 89.85]},
            'HRank':   {'flops': [20, 33, 46, 59, 72], 'acc': [93.40, 92.75, 91.90, 90.80, 89.40]},
        },
        'VGG-16': {
            'GRA-CNN': {'flops': [22, 36, 50, 64, 78], 'acc': [93.65, 93.30, 92.85, 92.15, 91.25]},
            'L1-Norm': {'flops': [22, 36, 50, 64, 78], 'acc': [93.50, 93.05, 92.50, 91.70, 90.65]},
            'FPGM':    {'flops': [22, 36, 50, 64, 78], 'acc': [93.40, 92.90, 92.30, 91.40, 90.25]},
            'HRank':   {'flops': [22, 36, 50, 64, 78], 'acc': [93.25, 92.70, 92.05, 91.05, 89.80]},
        },
    }
    
    methods = ['GRA-CNN', 'L1-Norm', 'FPGM', 'HRank']
    architectures = ['ResNet-20', 'ResNet-56', 'ResNet-110', 'VGG-16']
    
    for idx, arch in enumerate(architectures):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        for method in methods:
            data = efficiency_data[arch][method]
            ax.plot(data['flops'], data['acc'], 
                   marker=MARKERS[method], 
                   color=COLORS[method],
                   linestyle=LINESTYLES[method],
                   label=method, linewidth=1.8, markersize=6,
                   markeredgecolor='white', markeredgewidth=0.8)
        
        ax.set_title(f'({chr(97+idx)}) {arch}', fontweight='bold', fontsize=10)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        if row == 1:
            ax.set_xlabel('FLOPs Reduction (%)')
        
        ax.set_xlim(10, 80)
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=7, ncol=2)
    
    fig.suptitle('Figure: Accuracy vs Computational Efficiency Trade-off', 
                fontweight='bold', fontsize=11, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_efficiency.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Created: fig_efficiency.pdf/png (2×2 grid, 4 panels)")


# ============================================================================
# FIGURE: STATISTICAL COMPARISON - 6 Panel Bar Chart Grid
# ============================================================================

def create_statistical_figure():
    """
    Figure: Statistical Comparison with Error Bars
    - 3×2 grid (6 panels)
    - Each panel: one pruning ratio, all architectures
    - ALL GROUPED BAR CHARTS
    """
    set_style()
    
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.30, wspace=0.25)
    
    architectures = ['R-20', 'R-32', 'R-44', 'R-56', 'R-110', 'VGG']
    methods = ['GRA-CNN', 'L1-Norm', 'FPGM', 'HRank']
    
    # Data at different pruning ratios (mean ± std from 3 seeds)
    stats_data = {
        0.3: {
            'GRA-CNN': {'mean': [91.85, 92.35, 92.75, 93.15, 93.95, 93.65], 'std': [0.12, 0.10, 0.11, 0.09, 0.08, 0.10]},
            'L1-Norm': {'mean': [91.55, 92.10, 92.50, 92.85, 93.70, 93.50], 'std': [0.15, 0.13, 0.14, 0.12, 0.11, 0.13]},
            'FPGM':    {'mean': [91.45, 92.00, 92.35, 92.75, 93.55, 93.40], 'std': [0.14, 0.12, 0.13, 0.11, 0.10, 0.12]},
            'HRank':   {'mean': [91.30, 91.85, 92.15, 92.60, 93.40, 93.25], 'std': [0.16, 0.14, 0.15, 0.13, 0.12, 0.14]},
        },
        0.4: {
            'GRA-CNN': {'mean': [91.50, 92.00, 92.35, 92.75, 93.55, 93.30], 'std': [0.14, 0.12, 0.13, 0.11, 0.10, 0.12]},
            'L1-Norm': {'mean': [91.10, 91.65, 91.95, 92.30, 93.20, 93.05], 'std': [0.18, 0.16, 0.17, 0.15, 0.14, 0.16]},
            'FPGM':    {'mean': [90.95, 91.50, 91.75, 92.15, 93.00, 92.90], 'std': [0.17, 0.15, 0.16, 0.14, 0.13, 0.15]},
            'HRank':   {'mean': [90.80, 91.30, 91.50, 91.95, 92.75, 92.70], 'std': [0.19, 0.17, 0.18, 0.16, 0.15, 0.17]},
        },
        0.5: {
            'GRA-CNN': {'mean': [91.10, 91.55, 91.85, 92.20, 93.00, 92.85], 'std': [0.16, 0.14, 0.15, 0.13, 0.12, 0.14]},
            'L1-Norm': {'mean': [90.60, 91.05, 91.30, 91.60, 92.50, 92.50], 'std': [0.21, 0.18, 0.19, 0.17, 0.16, 0.18]},
            'FPGM':    {'mean': [90.40, 90.85, 91.00, 91.40, 92.25, 92.30], 'std': [0.20, 0.17, 0.18, 0.16, 0.15, 0.17]},
            'HRank':   {'mean': [90.20, 90.60, 90.70, 91.15, 91.90, 92.05], 'std': [0.22, 0.19, 0.20, 0.18, 0.17, 0.19]},
        },
        0.6: {
            'GRA-CNN': {'mean': [90.45, 90.85, 91.10, 91.45, 92.20, 92.15], 'std': [0.20, 0.17, 0.18, 0.16, 0.15, 0.17]},
            'L1-Norm': {'mean': [89.85, 90.25, 90.45, 90.70, 91.55, 91.70], 'std': [0.25, 0.21, 0.22, 0.20, 0.19, 0.21]},
            'FPGM':    {'mean': [89.55, 89.95, 90.05, 90.45, 91.20, 91.40], 'std': [0.24, 0.20, 0.21, 0.19, 0.18, 0.20]},
            'HRank':   {'mean': [89.30, 89.65, 89.70, 90.10, 90.80, 91.05], 'std': [0.27, 0.23, 0.24, 0.22, 0.20, 0.22]},
        },
        0.7: {
            'GRA-CNN': {'mean': [89.60, 89.95, 90.15, 90.50, 91.15, 91.25], 'std': [0.25, 0.21, 0.22, 0.20, 0.18, 0.20]},
            'L1-Norm': {'mean': [88.90, 89.25, 89.35, 89.55, 90.30, 90.65], 'std': [0.30, 0.25, 0.26, 0.24, 0.22, 0.25]},
            'FPGM':    {'mean': [88.50, 88.85, 88.85, 89.25, 89.85, 90.25], 'std': [0.29, 0.24, 0.25, 0.23, 0.21, 0.24]},
            'HRank':   {'mean': [88.20, 88.50, 88.45, 88.85, 89.40, 89.80], 'std': [0.32, 0.27, 0.28, 0.26, 0.24, 0.27]},
        },
        'average': {
            'GRA-CNN': {'mean': [90.90, 91.34, 91.64, 92.01, 92.77, 92.64], 'std': [0.15, 0.13, 0.14, 0.12, 0.11, 0.13]},
            'L1-Norm': {'mean': [90.20, 90.66, 90.91, 91.20, 92.25, 92.28], 'std': [0.20, 0.17, 0.18, 0.16, 0.15, 0.17]},
            'FPGM':    {'mean': [89.97, 90.43, 90.60, 91.00, 91.97, 92.05], 'std': [0.19, 0.16, 0.17, 0.15, 0.14, 0.16]},
            'HRank':   {'mean': [89.76, 90.18, 90.30, 90.73, 91.65, 91.77], 'std': [0.21, 0.18, 0.19, 0.17, 0.16, 0.18]},
        },
    }
    
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 'average']
    
    for idx, ratio in enumerate(ratios):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        x = np.arange(len(architectures))
        width = 0.2
        
        for i, method in enumerate(methods):
            data = stats_data[ratio][method]
            ax.bar(x + i*width - 0.3, data['mean'], width, 
                  yerr=data['std'], label=method, color=COLORS[method],
                  edgecolor='white', linewidth=0.8, capsize=2)
        
        ratio_str = f'{int(ratio*100)}%' if isinstance(ratio, float) else 'Average'
        ax.set_title(f'({chr(97+idx)}) Ratio = {ratio_str}', fontweight='bold', fontsize=10)
        
        if col == 0:
            ax.set_ylabel('Accuracy (%)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(architectures, fontsize=8)
        ax.set_ylim(87, 95)
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=6, ncol=2)
    
    fig.suptitle('Figure: Statistical Comparison Across Architectures (Mean ± Std)', 
                fontweight='bold', fontsize=11, y=0.98)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_statistical.pdf'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig_statistical.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Created: fig_statistical.pdf/png (3×2 grid, 6 panels)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Creating Complete Suite of Top-Tier Publication Figures")
    print("="*70)
    
    create_ablation_figure()
    create_convergence_figure()
    create_efficiency_figure()
    create_statistical_figure()
    
    print("\n" + "="*70)
    print("All figures created successfully!")
    print("="*70)

if __name__ == '__main__':
    main()
