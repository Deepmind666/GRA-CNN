import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    palette = get_palette()
    
    csv_path = os.path.join(os.path.dirname(__file__), 'results_vgg.csv')
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(6, 5))
    
    ratios = [0.5, 0.7]
    l1_accs = []
    gra_accs = []
    
    for r in ratios:
        row_l1 = df[(df['method'] == 'l1') & (df['ratio'] == r)]
        row_gra = df[(df['method'] == 'gra') & (df['ratio'] == r)]
        
        l1_accs.append(row_l1['acc'].values[0] if not row_l1.empty else 0)
        gra_accs.append(row_gra['acc'].values[0] if not row_gra.empty else 0)
        
    x = np.arange(len(ratios))
    width = 0.35
    
    plt.bar(x - width/2, l1_accs, width, label='L1-Norm', color=palette['L1-Norm'])
    plt.bar(x + width/2, gra_accs, width, label='GRA-CNN', color=palette['GRA-CNN'])
    
    baseline = 93.0 # VGG16 Baseline approx
    plt.axhline(y=baseline, color='black', linestyle='--', label=f'Baseline ({baseline}%)')
    
    plt.xticks(x, [f"Ratio {r}" for r in ratios])
    
    # Add value labels
    for i, v in enumerate(l1_accs):
        plt.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    for i, v in enumerate(gra_accs):
        plt.text(i + width/2, v + 0.1, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    
    # Zoom Y-axis
    valid_accs = [a for a in l1_accs + gra_accs if a > 0]
    if valid_accs:
        y_min = min(valid_accs)
        plt.ylim(y_min - 2.0, baseline + 1.0)
        
    plt.ylabel('Top-1 Accuracy (%)')
    
    plt.legend(loc='lower left')
    plt.grid(True, axis='y')
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig10.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == "__main__":
    main()
