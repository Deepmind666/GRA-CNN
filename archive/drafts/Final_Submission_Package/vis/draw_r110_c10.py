import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    palette = get_palette()
    
    csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')
    df = pd.read_csv(csv_path)
    
    # Filter for ResNet-110 on CIFAR-10
    df = df[(df['dataset'] == 'CIFAR-10') & (df['model'] == 'ResNet-110') & (df['prune_ratio'] == 0.5)]
    
    methods = ['L1-Norm', 'GRA-CNN']
    accs = []
    
    for m in methods:
        row = df[df['method'] == m]
        if not row.empty:
            accs.append(row['accuracy'].values[-1])
        else:
            accs.append(0)
            
    # Baseline
    baseline = 93.5 # Approx for ResNet-110
    
    plt.figure(figsize=(6, 5))
    
    x = np.arange(len(methods))
    width = 0.5
    
    colors = [palette['L1-Norm'], palette['GRA-CNN']]
    
    bars = plt.bar(x, accs, width, color=colors)
    plt.axhline(y=baseline, color='black', linestyle='--', label='Baseline')
    
    # Add text labels
    for bar, v in zip(bars, accs):
        if v > 0:
            plt.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.2f}%", 
                     ha='center', va='bottom', fontsize=12)
        
    plt.xticks(x, methods)
    plt.ylabel('Top-1 Accuracy (%)')
    
    # Zoom
    if any(a > 0 for a in accs):
        min_val = min([a for a in accs if a > 0])
        plt.ylim(min_val - 2.0, baseline + 1.0)
    
    plt.legend()
    plt.grid(True, axis='y')
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig9.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == "__main__":
    main()
