import pandas as pd
import matplotlib.pyplot as plt
import os

def draw_tiny_results():
    results_path = r'C:\GRA-CNN\experiments\tinyimagenet\all_results.csv'
    if not os.path.exists(results_path):
        print(f"Results not found at {results_path}")
        return

    df = pd.read_csv(results_path)
    
    # Filter methods
    methods = ['gra', 'l1', 'fpgm']
    labels = {'gra': 'GRA-CNN (Ours)', 'l1': 'L1-Norm', 'fpgm': 'FPGM'}
    colors = {'gra': '#d62728', 'l1': '#1f77b4', 'fpgm': '#2ca02c'}
    markers = {'gra': 'o', 'l1': 's', 'fpgm': '^'}

    plt.figure(figsize=(8, 6))
    
    # Add baseline point (ratio 0)
    baseline_acc = 67.92
    plt.plot(0, baseline_acc, 'k*', markersize=12, label='Baseline')

    for method in methods:
        data = df[df['method'] == method].sort_values('prune_ratio')
        ratios = [0] + list(data['prune_ratio'])
        accs = [baseline_acc] + list(data['final_acc'])
        
        plt.plot(ratios, accs, marker=markers[method], color=colors[method], 
                 label=labels[method], linewidth=2, markersize=8)

    plt.xlabel('Pruning Ratio', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Pruning Performance on Tiny-ImageNet (ResNet-18)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    save_path = r'C:\GRA-CNN\APIN_Submission\fig_tiny_results.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    draw_tiny_results()
