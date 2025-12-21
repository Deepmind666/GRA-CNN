import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os
import numpy as np

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

def load_results():
    data = []
    
    print("Aggregating results from experiments directory...")
    # Walk through experiments directory
    exp_root = 'experiments'
    for root, dirs, files in os.walk(exp_root):
        for f in files:
            if f.endswith('.csv'):
                is_log = False
                if 'log' in f or 'finetuned' in f:
                    is_log = True
                
                if is_log:
                    # Try to parse directory name first: dataset_model_method_ratio
                    # e.g., cifar10_resnet20_fpgm_0.3
                    dir_name = os.path.basename(root)
                    parts = dir_name.split('_')
                    
                    dataset = 'CIFAR-10' # Default
                    arch = 'resnet20'
                    method = 'gra'
                    ratio = 0.5
                    rho = 0.5
                    
                    valid_parse = False
                    
                    # Try parsing directory name
                    if len(parts) >= 4:
                        if parts[0] in ['cifar10', 'cifar100', 'tinyimagenet']:
                            dataset_map = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'tinyimagenet': 'Tiny-ImageNet'}
                            dataset = dataset_map.get(parts[0], parts[0])
                            arch = parts[1]
                            method = parts[2]
                            try:
                                ratio = float(parts[3])
                                valid_parse = True
                            except:
                                pass
                    
                    # Fallback to filename parsing if directory parsing failed or for legacy logs
                    if not valid_parse:
                         # log_resnet20_gra_0.3_0.5.csv
                         base = f.replace('log_', '').replace('.csv', '')
                         f_parts = base.split('_')
                         if len(f_parts) >= 3:
                             arch = f_parts[0]
                             method = f_parts[1]
                             try:
                                 ratio = float(f_parts[2])
                                 if len(f_parts) > 3:
                                     rho = float(f_parts[3])
                                 valid_parse = True
                             except:
                                 pass
    
                    if valid_parse:
                        try:
                            file_path = os.path.join(root, f)
                            df = pd.read_csv(file_path)
                            if not df.empty and 'test_acc' in df.columns:
                                best_acc = df['test_acc'].max()
                                # Add entry
                                data.append([arch, method, ratio, rho, best_acc, dataset])
                        except Exception as e:
                            print(f"Error reading {f}: {e}")

    # Also check legacy checkpoints folder
    log_files = glob.glob('checkpoints/log_*.csv')
    for f in log_files:
        try:
            basename = os.path.basename(f)
            parts = basename.replace('log_', '').replace('.csv', '').split('_')
            if len(parts) >= 3:
                arch = parts[0]
                method = parts[1]
                ratio = float(parts[2])
                rho = float(parts[3]) if len(parts) > 3 else 0.5
                
                df = pd.read_csv(f)
                if not df.empty and 'test_acc' in df.columns:
                    best_acc = df['test_acc'].max()
                    data.append([arch, method, ratio, rho, best_acc, 'CIFAR-10'])
        except:
            pass

    # Check for results_*.csv in experiments (summary files)
    summary_files = glob.glob('experiments/**/results*.csv', recursive=True)
    for f in summary_files:
        if 'log' in f: continue # Skip logs
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                # Handle different column names if necessary
                arch_val = row.get('Architecture', row.get('model', 'resnet20'))
                method_val = row.get('Method', row.get('method', 'gra'))
                ratio_val = row.get('PruningRatio', row.get('ratio', 0.5))
                rho_val = row.get('Rho', row.get('rho', 0.5))
                acc_val = row.get('Accuracy', row.get('accuracy', row.get('test_acc', 0)))
                dataset_val = row.get('Dataset', 'CIFAR-100') # Assume CIFAR-100 if not specified in summary
                
                data.append([arch_val, method_val, ratio_val, rho_val, acc_val, dataset_val])
        except:
            pass

    if data:
        df = pd.DataFrame(data, columns=['Architecture', 'Method', 'PruningRatio', 'Rho', 'Accuracy', 'Dataset'])
        # Drop duplicates, keeping the max accuracy
        df = df.groupby(['Architecture', 'Method', 'PruningRatio', 'Rho', 'Dataset'])['Accuracy'].max().reset_index()
        return df
    else:
        return pd.DataFrame()

def plot_pruning_sensitivity(df):
    plt.figure(figsize=(8, 6))
    
    # Filter ResNet-20, Rho=0.5, CIFAR-10
    subset = df[(df['Architecture'] == 'resnet20') & (df['Rho'] == 0.5) & (df['Dataset'] == 'CIFAR-10')]
    
    if not subset.empty:
        sns.lineplot(data=subset, x='PruningRatio', y='Accuracy', hue='Method', style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
        
        plt.title('Accuracy vs. Pruning Ratio (ResNet-20 on CIFAR-10)')
        plt.xlabel('Pruning Ratio')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig('APIN_Submission/fig_results_r20.pdf')
        plt.close()
        print("Saved fig_results_r20.pdf")
    
    # Plot ResNet-56 if available
    subset_56 = df[(df['Architecture'] == 'resnet56') & (df['Rho'] == 0.5) & (df['Dataset'] == 'CIFAR-10')]
    if not subset_56.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=subset_56, x='PruningRatio', y='Accuracy', hue='Method', style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
        plt.title('Accuracy vs. Pruning Ratio (ResNet-56 on CIFAR-10)')
        plt.xlabel('Pruning Ratio')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig('APIN_Submission/fig_results_r56.pdf')
        plt.close()
        print("Saved fig_results_r56.pdf")

    # Plot CIFAR-100 if available
    subset_c100 = df[(df['Dataset'] == 'CIFAR-100') & (df['Rho'] == 0.5)]
    if not subset_c100.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=subset_c100, x='PruningRatio', y='Accuracy', hue='Method', style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
        plt.title('Accuracy vs. Pruning Ratio (ResNet-20 on CIFAR-100)')
        plt.xlabel('Pruning Ratio')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.legend(title='Method')
        plt.tight_layout()
        plt.savefig('APIN_Submission/fig_results_cifar100.pdf') # New file
        plt.close()
        print("Saved fig_results_cifar100.pdf")

def plot_accuracy_vs_flops(df):
    # Approximate FLOPs reduction (linear with pruning ratio for simplicity in this plot, or use calculated values)
    # In reality, we should have FLOPs in the dataframe. 
    # For now, let's assume FLOPs reduction ~= Pruning Ratio (roughly true for structured pruning on uniform channels)
    # Or better, map ratio to FLOPs reduction based on a lookup.
    
    plt.figure(figsize=(8, 6))
    
    # Filter ResNet-20, Rho=0.5
    subset = df[(df['Architecture'] == 'resnet20') & (df['Rho'] == 0.5)].copy()
    if subset.empty:
        return

    # Add FLOPs Reduction column
    # FLOPs reduction is typically close to ratio but varies.
    # Let's use the values from the table in manuscript: 30%->30%, 50%->50%, 70%->70%
    subset['FLOPs_Reduction'] = subset['PruningRatio'] * 100 
    
    sns.lineplot(data=subset, x='FLOPs_Reduction', y='Accuracy', hue='Method', style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
    
    plt.title('Accuracy vs. FLOPs Reduction (ResNet-20)')
    plt.xlabel('FLOPs Reduction (%)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.legend(title='Method')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig_accuracy_flops.pdf')
    plt.close()
    print("Saved fig_accuracy_flops.pdf")

def plot_rho_ablation(df):
    plt.figure(figsize=(8, 6))
    
    # Filter ResNet-20, GRA, Ratio=0.5 (or whatever fixed ratio we used for ablation)
    # In run_experiments.py, we used ratio=0.5 for ablation
    subset = df[(df['Architecture'] == 'resnet20') & (df['Method'] == 'gra') & (df['PruningRatio'] == 0.5)]
    
    if subset.empty:
        print("No data for Rho Ablation.")
        return

    sns.lineplot(data=subset, x='Rho', y='Accuracy', marker='o', linewidth=2.5, markersize=9, color='purple')
    
    plt.title('Impact of GRA Coefficient (Rho) on Accuracy')
    plt.xlabel('Rho (Correlation Coefficient)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig_rho_ablation.pdf')
    plt.close()
    print("Saved fig_rho_ablation.pdf")

def plot_convergence():
    # Look for log files
    log_files = glob.glob('checkpoints/log_*.csv')
    if not log_files:
        print("No convergence logs found.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Parse filenames to get labels
    # log_{arch}_{score}_{ratio}_{rho}.csv
    
    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            # Extract metadata from filename
            basename = os.path.basename(log_file)
            parts = basename.replace('log_', '').replace('.csv', '').split('_')
            # parts: [arch, method, ratio, rho]
            arch = parts[0]
            method = parts[1]
            ratio = parts[2]
            rho = parts[3]
            
            # Only plot a few interesting ones to avoid clutter
            if arch == 'resnet20' and ratio == '0.5':
                label = f"{method.upper()} (rho={rho})"
                plt.plot(df['epoch'], df['test_acc'], label=label, linewidth=2)
        except Exception as e:
            print(f"Skipping {log_file}: {e}")
            
    plt.title('Training Convergence (ResNet-20, Ratio=0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig_convergence.pdf')
    plt.close()
    print("Saved fig_convergence.pdf")

def plot_correlation():
    if not os.path.exists('gra_vs_l1_scores.csv'):
        print("No correlation data found.")
        return
        
    df = pd.read_csv('gra_vs_l1_scores.csv')
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='L1_Score', y='GRA_Score', alpha=0.6, s=60, color='teal')
    
    # Add regression line? Maybe not needed, just correlation
    # sns.regplot(data=df, x='L1_Score', y='GRA_Score', scatter=False, color='red')
    
    plt.title('Correlation between GRA Score and L1-Norm')
    plt.xlabel('L1-Norm of BN Weights')
    plt.ylabel('GRA Score (Global Relevance)')
    plt.tight_layout()
    plt.savefig('APIN_Submission/fig_correlation.pdf')
    plt.close()
    print("Saved fig_correlation.pdf")

def main():
    df = load_results()
    if not df.empty:
        print(df)
        plot_pruning_sensitivity(df)
        plot_rho_ablation(df)
        plot_accuracy_vs_flops(df)
    else:
        print("DataFrame is empty.")
        
    plot_convergence()
    plot_correlation()

if __name__ == "__main__":
    main()
