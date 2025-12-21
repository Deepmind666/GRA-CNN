import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Style settings for Journal Quality
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid", {'grid.linestyle': '--', 'axes.grid': True})

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.9
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False # Set to True if latex is available, but safe False

CSV_PATH = 'vis/results_master.csv'

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found.")
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def plot_fig8_r110(df):
    if df is None: return
    subset = df[(df['dataset'] == 'cifar10') & (df['model'] == 'resnet110')]
    if subset.empty: return

    methods = ['l1', 'fpgm', 'gra']
    colors = {"l1": "#377eb8", "fpgm": "#4daf4a", "gra": "#e41a1c"} # Set 1 Colors
    markers = {"l1": "o", "fpgm": "s", "gra": "^"}
    labels = {"l1": "L1-Norm", "fpgm": "FPGM", "gra": "GRA-CNN (Ours)"}

    # Subplot 1: Acc vs Ratio
    plt.figure(figsize=(7, 6))
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        
        # Shade error region if std exists and is non-zero
        if 'std_acc' in data.columns:
            plt.fill_between(data['prune_ratio'], 
                             data['mean_acc'] - data['std_acc'], 
                             data['mean_acc'] + data['std_acc'], 
                             color=colors[m], alpha=0.15)
                             
        plt.plot(data['prune_ratio'], data['mean_acc'], 
                 label=labels[m], color=colors[m], marker=markers[m], markersize=9)

    plt.xlabel("Pruning Ratio", fontweight='bold')
    plt.ylabel("Top-1 Accuracy (%)", fontweight='bold')
    plt.title("ResNet-110 on CIFAR-10", fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig8_accuracy.pdf')
    plt.close()

    # Subplot 2: FLOPs Reduction vs Acc
    plt.figure(figsize=(7, 6))
    baseline_flops = 255.0 
    
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        
        reduction = 100 * (1 - data['flops'] / baseline_flops)
        plt.plot(reduction, data['mean_acc'], label=labels[m], color=colors[m], marker=markers[m], markersize=9)

    baseline_acc = 94.3
    plt.plot(0, baseline_acc, marker='*', color='black', markersize=14, label='Baseline', linestyle='None')

    plt.xlabel("FLOPs Reduction (%)", fontweight='bold')
    plt.ylabel("Top-1 Accuracy (%)", fontweight='bold')
    plt.title("ResNet-110 Efficiency Trade-off", fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig8_tradeoff.pdf')
    plt.close()

def plot_fig9_vgg(df):
    if df is None: return
    subset = df[(df['dataset'] == 'cifar10') & (df['model'] == 'vgg16')]
    if subset.empty: return

    methods = ['l1', 'fpgm', 'gra']
    colors = {"l1": "#377eb8", "fpgm": "#4daf4a", "gra": "#e41a1c"}
    markers = {"l1": "o", "fpgm": "s", "gra": "^"}
    labels = {"l1": "L1-Norm", "fpgm": "FPGM", "gra": "GRA-CNN (Ours)"}

    # Acc vs Ratio
    plt.figure(figsize=(7, 6))
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        
        if 'std_acc' in data.columns:
            plt.fill_between(data['prune_ratio'], 
                             data['mean_acc'] - data['std_acc'], 
                             data['mean_acc'] + data['std_acc'], 
                             color=colors[m], alpha=0.15)

        plt.plot(data['prune_ratio'], data['mean_acc'], 
                 label=labels[m], color=colors[m], marker=markers[m], markersize=9)

    plt.xlabel("Pruning Ratio", fontweight='bold')
    plt.ylabel("Top-1 Accuracy (%)", fontweight='bold')
    plt.title("VGG-16 on CIFAR-10", fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig9_accuracy.pdf')
    plt.close()

    # Tradeoff
    plt.figure(figsize=(7, 6))
    baseline_flops_vgg = 313.0
    
    for m in methods:
        data = subset[subset['method'] == m].sort_values('prune_ratio')
        if data.empty: continue
        
        reduction = 100 * (1 - data['flops'] / baseline_flops_vgg)
        plt.plot(reduction, data['mean_acc'], label=labels[m], color=colors[m], marker=markers[m], markersize=9)

    baseline_acc_vgg = 93.96
    plt.plot(0, baseline_acc_vgg, marker='*', color='black', markersize=14, label='Baseline', linestyle='None')

    plt.xlabel("FLOPs Reduction (%)", fontweight='bold')
    plt.ylabel("Top-1 Accuracy (%)", fontweight='bold')
    plt.title("VGG-16 Efficiency Trade-off", fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig9_tradeoff.pdf')
    plt.close()

def plot_fig10_rho(df):
    if df is None: return
    subset = df[(df['dataset'] == 'cifar10') & (df['method'] == 'gra') & (df['prune_ratio'] == 0.5)]
    
    r20 = subset[subset['model'] == 'resnet20'].sort_values('rho')
    r56 = subset[subset['model'] == 'resnet56'].sort_values('rho')
    
    if not r20.empty:
        plt.figure(figsize=(6, 5))
        plt.plot(r20['rho'], r20['mean_acc'], marker='o', color="#e41a1c", markersize=9, linewidth=2.5)
        plt.fill_between(r20['rho'], 84.5, r20['mean_acc'], color="#e41a1c", alpha=0.05)
        
        # Mark rho=0.5
        plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        
        plt.xlabel(r'Resolution Coefficient $\rho$', fontweight='bold')
        plt.ylabel('Top-1 Accuracy (%)', fontweight='bold')
        plt.title('ResNet-20 Sensitivity', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('APIN_Submission/fig10_rho_20.pdf')
        plt.close()

    if not r56.empty:
        plt.figure(figsize=(6, 5))
        plt.plot(r56['rho'], r56['mean_acc'], marker='^', color="#377eb8", markersize=9, linewidth=2.5)
        plt.fill_between(r56['rho'], 92.5, r56['mean_acc'], color="#377eb8", alpha=0.05)
        
        # Mark rho=0.5
        plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        plt.text(0.51, 91.5, r'$\rho=0.5$', fontsize=12, color='gray', rotation=90, verticalalignment='bottom')

        plt.xlabel(r'Resolution Coefficient $\rho$', fontweight='bold')
        plt.ylabel('Top-1 Accuracy (%)', fontweight='bold')
        plt.title('ResNet-56 Sensitivity', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('APIN_Submission/fig10_rho_56.pdf')
        plt.close()

def plot_cifar100_bar(df):
    if df is None: return
    subset = df[(df['dataset'] == 'cifar100') & (df['model'] == 'resnet56') & (df['prune_ratio'] == 0.5)]
    if subset.empty: return

    # Prepare data
    methods = ['l1', 'fpgm', 'gra']
    labels = ['L1-Norm', 'FPGM', 'GRA-CNN']
    accs = []
    errs = []
    
    for m in methods:
        row = subset[subset['method'] == m]
        if not row.empty:
            accs.append(row['mean_acc'].values[0])
            errs.append(row['std_acc'].values[0] if 'std_acc' in row.columns else 0)
        else:
            accs.append(0)
            errs.append(0)

    # Plot
    plt.figure(figsize=(7, 6))
    colors = ["#377eb8", "#4daf4a", "#e41a1c"]
    bars = plt.bar(labels, accs, yerr=errs, capsize=10, color=colors, alpha=0.8, width=0.6)
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylim(65, 73)
    plt.ylabel("Top-1 Accuracy (%)", fontweight='bold')
    plt.title("ResNet-56 on CIFAR-100 (Ratio=0.5)", fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig_cifar100_bar.pdf')
    plt.close()

if __name__ == "__main__":
    df = load_data()
    plot_fig8_r110(df)
    plot_fig9_vgg(df)
    plot_fig10_rho(df)
    plot_cifar100_bar(df)
    print("Professional auto-drawing finished.")
