import pandas as pd
import matplotlib.pyplot as plt
import os
from draw_utils import set_style, save_fig, get_palette

def main():
    set_style()
    palette = get_palette()
    
    # Paths to log files
    l1_path = os.path.join(os.path.dirname(__file__), '../experiments/resnet20_l1_r0.5/training_log.csv')
    gra_path = os.path.join(os.path.dirname(__file__), '../experiments/resnet20_gra_r0.5_rho0.5/training_log.csv')
    
    # Check if files exist
    if not os.path.exists(l1_path) or not os.path.exists(gra_path):
        print("Log files not found. Skipping convergence plot.")
        return

    try:
        df_l1 = pd.read_csv(l1_path)
        df_gra = pd.read_csv(gra_path)
    except Exception as e:
        print(f"Error reading logs: {e}")
        return
    
    plt.figure(figsize=(6, 5))
    
    # Identify columns
    # Common names: 'epoch', 'test_acc', 'acc', 'val_acc'
    epoch_col = 'epoch'
    acc_col = 'test_acc'
    
    # Fallback column detection
    if epoch_col not in df_l1.columns:
        epoch_col = df_l1.columns[0] # Assume first is epoch
    if acc_col not in df_l1.columns:
        # Find column with 'acc'
        acc_cols = [c for c in df_l1.columns if 'acc' in c.lower()]
        if acc_cols:
            acc_col = acc_cols[-1]
        else:
            acc_col = df_l1.columns[-1]

    # Plot L1
    plt.plot(df_l1[epoch_col], df_l1[acc_col], 
             label='L1-Norm', 
             color=palette['L1-Norm'], 
             linestyle='-',
             linewidth=2.2)
             
    # Plot GRA
    # Ensure columns match or detect for GRA too
    epoch_col_gra = 'epoch'
    acc_col_gra = 'test_acc'
    if epoch_col_gra not in df_gra.columns:
        epoch_col_gra = df_gra.columns[0]
    if acc_col_gra not in df_gra.columns:
         acc_cols = [c for c in df_gra.columns if 'acc' in c.lower()]
         if acc_cols:
            acc_col_gra = acc_cols[-1]
         else:
            acc_col_gra = df_gra.columns[-1]
            
    plt.plot(df_gra[epoch_col_gra], df_gra[acc_col_gra], 
             label='GRA-CNN', 
             color=palette['GRA-CNN'], 
             linestyle='-',
             linewidth=2.2)
            
    plt.xlabel('Epochs')
    plt.ylabel('Top-1 Accuracy (%)')
    
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), '../APIN_Submission/fig5.pdf')
    save_fig(plt.gcf(), output_path)

if __name__ == "__main__":
    main()
