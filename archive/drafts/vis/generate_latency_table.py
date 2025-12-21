import pandas as pd

def main():
    df = pd.read_csv('vis/latency_results.csv')
    
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Inference Latency on RTX 5090 (Batch Size=128).}")
    print("\\label{tab:latency}")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Dataset & Model & Pruned? & Latency (ms) & FPS (img/s) & Speedup \\\\")
    print("\\midrule")
    
    # Group by dataset/model
    grouped = df.groupby(['dataset', 'model'])
    
    for (d, m), group in grouped:
        # Get baseline
        base = group[group['ratio'] == 0.0].iloc[0]
        pruned = group[group['ratio'] > 0.0].iloc[0]
        
        d_str = "CIFAR-10" if d == "cifar10" else "Tiny-ImageNet"
        m_str = m.replace("resnet", "ResNet-")
        if "tiny" in m: m_str = "ResNet-18"
        
        # Baseline Row
        print(f"{d_str} & {m_str} & No & {base['latency_ms']:.2f} & {base['fps']:.0f} & 1.00x \\\\")
        
        # Pruned Row
        speedup = base['latency_ms'] / pruned['latency_ms']
        print(f" & & Yes (50\\%) & {pruned['latency_ms']:.2f} & {pruned['fps']:.0f} & {speedup:.2f}x \\\\")
        print("\\midrule")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    main()
