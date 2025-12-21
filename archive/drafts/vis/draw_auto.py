import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 12

CSV_PATH = 'vis/results_master.csv'

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found. Using simulated data for testing.")
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def plot_fig8_r110(df):
    # ResNet-110, CIFAR-10
    # Filter data
    if df is None: return
    
    subset = df[(df['dataset'] == 'cifar10') & (df['model'] == 'resnet110')]
    if subset.empty: return

    methods = ['l1', 'fpgm', 'gra']
    colors = {"l1": "#1f77b4", "fpgm": "#2ca02c", "gra": "#d62728"}
    markers = {"l1": "o", "fpgm": "s", "gra": "^"}
    labels = {"l1": "L1-Norm", "fpgm": "FPGM", "gra": "GRA-CNN"}

    # Subplot 1: Acc vs Ratio
    plt.figure(figsize=(6, 5))
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        plt.errorbar(data['prune_ratio'], data['mean_acc'], yerr=data['std_acc'],
                     label=labels[m], color=colors[m], marker=markers[m], markersize=7, linewidth=2.2, capsize=4)
    
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Accuracy vs Pruning Ratio (ResNet-110)")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig8_accuracy.pdf')
    plt.close()

    # Subplot 2: FLOPs Reduction vs Acc
    plt.figure(figsize=(6, 5))
    
    # Calculate Reduction: Baseline FLOPs ~ 255M for ResNet-110
    baseline_flops = 255.0 
    
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        
        # Calculate reduction percentage
        reduction = 100 * (1 - data['flops'] / baseline_flops)
        
        plt.plot(reduction, data['mean_acc'], label=labels[m], color=colors[m], marker=markers[m])
        
    # Add Baseline Point
    baseline_acc = 94.3
    plt.plot(0, baseline_acc, marker='*', color='black', markersize=12, label='Baseline', linestyle='None')

    plt.xlabel("FLOPs Reduction (%)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("FLOPs Reduction vs Accuracy (ResNet-110)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig8_tradeoff.pdf')
    plt.close()

def plot_fig9_vgg(df):
    if df is None: return
    subset = df[(df['dataset'] == 'cifar10') & (df['model'] == 'vgg16')]
    if subset.empty: return

    methods = ['l1', 'fpgm', 'gra']
    colors = {"l1": "#1f77b4", "fpgm": "#2ca02c", "gra": "#d62728"}
    markers = {"l1": "o", "fpgm": "s", "gra": "^"}
    labels = {"l1": "L1-Norm", "fpgm": "FPGM", "gra": "GRA-CNN"}

    plt.figure(figsize=(6, 5))
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        plt.errorbar(data['prune_ratio'], data['mean_acc'], yerr=data['std_acc'],
                     label=labels[m], color=colors[m], marker=markers[m], markersize=7, linewidth=2.2, capsize=4)
    
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("VGG-16 Accuracy vs Pruning Ratio")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig9_accuracy.pdf')
    plt.close()

    plt.figure(figsize=(6, 5))
    baseline_flops_vgg = 313.0 # Approx 313M for VGG16-CIFAR
    
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        
        reduction = 100 * (1 - data['flops'] / baseline_flops_vgg)
        plt.plot(reduction, data['mean_acc'], label=labels[m], color=colors[m], marker=markers[m])

    # Add Baseline
    baseline_acc_vgg = 93.96
    plt.plot(0, baseline_acc_vgg, marker='*', color='black', markersize=12, label='Baseline', linestyle='None')

    plt.xlabel("FLOPs Reduction (%)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("VGG-16 FLOPs Reduction vs Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig9_tradeoff.pdf')
    plt.close()

def plot_fig10_rho(df):
    if df is None: return
    # ResNet-20/56, method=gra, ratio=0.5, varying rho
    
    subset = df[(df['dataset'] == 'cifar10') & (df['method'] == 'gra') & (df['prune_ratio'] == 0.5)]
    
    # R20
    r20 = subset[subset['model'] == 'resnet20'].sort_values('rho')
    # R56
    r56 = subset[subset['model'] == 'resnet56'].sort_values('rho')
    
    if not r20.empty:
        plt.figure(figsize=(6, 5))
        plt.plot(r20['rho'], r20['mean_acc'], marker='o', label='ResNet-20')
        plt.xlabel(r'Hyperparameter $\rho$')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig('APIN_Submission/fig10_rho_20.pdf')
        plt.close()

    if not r56.empty:
        plt.figure(figsize=(6, 5))
        plt.plot(r56['rho'], r56['mean_acc'], marker='^', color='g', label='ResNet-56')
        plt.xlabel(r'Hyperparameter $\rho$')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig('APIN_Submission/fig10_rho_56.pdf')
        plt.close()

if __name__ == "__main__":
    df = load_data()
    plot_fig8_r110(df)
    plot_fig9_vgg(df)
    plot_fig10_rho(df)
    print("Auto-drawing finished.")
