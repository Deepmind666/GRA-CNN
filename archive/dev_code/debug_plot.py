import pandas as pd
import os
import glob

def load_results():
    data = []
    exp_root = 'experiments'
    print(f"Searching in {os.path.abspath(exp_root)}")
    for root, dirs, files in os.walk(exp_root):
        print(f"Visiting: {root}")
        for f in files:
            if f.endswith('.csv'):
                print(f"  Found CSV: {f}")
            if f.endswith('.csv') and 'log' in f:
                dir_name = os.path.basename(root)
                parts = dir_name.split('_')
                
                if len(parts) >= 4:
                    if parts[0] in ['cifar10', 'cifar100', 'tinyimagenet']:
                        dataset = parts[0]
                        arch = parts[1]
                        method = parts[2]
                        ratio = parts[3]
                        
                        print(f"Found candidate: {dir_name} -> {dataset}, {arch}, {method}, {ratio}")
                        
                        try:
                            df = pd.read_csv(os.path.join(root, f))
                            if 'test_acc' in df.columns:
                                best = df['test_acc'].max()
                                print(f"  -> Best Acc: {best}")
                                data.append([method, best])
                            else:
                                print("  -> No test_acc column")
                        except Exception as e:
                            print(f"  -> Error: {e}")

    df = pd.DataFrame(data, columns=['Method', 'Accuracy'])
    print("\nSummary:")
    print(df.groupby('Method').max())

if __name__ == "__main__":
    load_results()
