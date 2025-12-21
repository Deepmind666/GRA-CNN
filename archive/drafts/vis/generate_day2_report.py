import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

def get_last_acc(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        df = pd.read_csv(log_path)
        if df.empty: return None
        return df['test_acc'].max() # Use MAX accuracy
    except:
        return None

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    day2_dir = os.path.join(base_dir, 'experiments', 'day2')
    
    results = {}
    
    # 1. Load from results_multi.csv (L1, GRA)
    multi_path = os.path.join(base_dir, 'vis', 'results_multi.csv')
    if os.path.exists(multi_path):
        df = pd.read_csv(multi_path)
        # Filter for ResNet-56 CIFAR-10 Ratio 0.5
        subset = df[(df['model'] == 'resnet56') & (df['dataset'] == 'cifar10') & (df['prune_ratio'] == 0.5)]
        for _, row in subset.iterrows():
            results[row['method']] = row['mean_acc']
            
    # 2. Load from Day 2 logs (Cosine, Correlation, MI)
    metrics = ['cosine', 'correlation', 'mi']
    for m in metrics:
        log_path = os.path.join(day2_dir, f'comparative_{m}', 'training_log.csv')
        acc = get_last_acc(log_path)
        if acc is not None:
            results[m] = acc
        else:
            results[m] = 0.0 # Running or Failed
            
    print("=== Comparative Results (ResNet-56, Ratio=0.5) ===")
    print(results)
    
    # 3. Plot Bar Chart
    methods = ['l1', 'gra', 'cosine', 'correlation', 'mi']
    # Filter methods that exist
    methods = [m for m in methods if m in results and results[m] > 0]
    accs = [results[m] for m in methods]
    
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    # Map colors to methods if possible for consistency
    color_map = {'l1': '#1f77b4', 'gra': '#d62728', 'cosine': '#2ca02c', 'correlation': '#ff7f0e', 'mi': '#9467bd'}
    bar_colors = [color_map.get(m, 'gray') for m in methods]
    
    bars = plt.bar(methods, accs, color=bar_colors)
    plt.ylim(90, 94) # Zoom in
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Method Comparison (ResNet-56, 50% Pruning)')
    
    for bar, v in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}%", ha='center', va='bottom')
        
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(base_dir, 'vis', 'comparative_metrics_bar.pdf'))
    print(f"Saved bar chart to vis/comparative_metrics_bar.pdf")
    
    # 4. Generate LaTeX Table
    tex = []
    tex.append("\\begin{table}[h]")
    tex.append("\\centering")
    tex.append("\\caption{Comparison of different importance metrics on ResNet-56 (CIFAR-10, 50\\% Pruning).}")
    tex.append("\\label{tab:comparative_metrics}")
    tex.append("\\begin{tabular}{lc}")
    tex.append("\\toprule")
    tex.append("Metric & Top-1 Accuracy (\\%) \\\\")
    tex.append("\\midrule")
    
    # Sort by accuracy desc
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    name_map = {'l1': 'L1-Norm', 'gra': 'GRA (Ours)', 'cosine': 'Cosine Similarity', 'correlation': 'Pearson Correlation', 'mi': 'Mutual Information'}
    
    for m, acc in sorted_res:
        if acc > 0:
            name = name_map.get(m, m.upper())
            if m == 'gra':
                tex.append(f"\\textbf{{{name}}} & \\textbf{{{acc:.2f}}} \\\\")
            else:
                tex.append(f"{name} & {acc:.2f} \\\\")
            
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    
    with open(os.path.join(base_dir, 'vis', 'table_comparative.tex'), 'w') as f:
        f.write("\n".join(tex))
    print("Saved table to vis/table_comparative.tex")

if __name__ == "__main__":
    main()
