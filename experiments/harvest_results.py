
import os
import pandas as pd
import re
import glob

def harvest_data(root_dir):
    print(f"Harvesting from {root_dir}...")
    
    # List to store all found results
    all_results = []
    
    # 1. Look for loose baseline CSVs
    baseline_files = glob.glob(os.path.join(root_dir, "baseline_*.csv"))
    for f in baseline_files:
        try:
            df = pd.read_csv(f)
            # Add metadata if missing
            match = re.search(r'baseline_([^_]+)_([^\.]+)', os.path.basename(f))
            if match:
                dataset, arch = match.groups()
                df['architecture'] = arch
                df['dataset'] = dataset
                df['method'] = 'baseline'
                df['ratio'] = 0.0
                df['pruned_acc'] = df['test_acc'] if 'test_acc' in df.columns else df.iloc[-1]['test_acc']
                all_results.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 2. Look for experiment subdirectories
    # Pattern: architecture_method_rRatio...
    # e.g. resnet20_gra_r0.3_rho0.5
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for d in subdirs:
        dir_path = os.path.join(root_dir, d)
        
        # Try to parse info from dirname
        parts = d.split('_')
        if len(parts) >= 3:
            arch = parts[0]
            method = parts[1]
            # Try to find ratio
            ratio = None
            for p in parts:
                if p.startswith('r') and p[1].isdigit():
                     ratio = float(p[1:])
                elif p.startswith('0.'): # sometimes just 0.5
                     try: ratio = float(p)
                     except: pass
            
            # Look for result.csv or log.txt
            csv_path = os.path.join(dir_path, 'result.csv')
            if not os.path.exists(csv_path):
                 # Try finding any csv
                 csvs = glob.glob(os.path.join(dir_path, "*.csv"))
                 if csvs: csv_path = csvs[0]
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        # Get best/last accuracy
                        if 'test_acc' in df.columns:
                            acc = df['test_acc'].max()
                        elif 'acc' in df.columns:
                            acc = df['acc'].max()
                        else:
                            continue
                            
                        # Determine dataset (assume CIFAR-10 unless specified)
                        dataset = 'cifar10'
                        if 'cifar100' in d: dataset = 'cifar100'
                        if 'tiny' in d: dataset = 'tiny_imagenet'
                        
                        rw = {
                            'architecture': arch.replace('resnet', 'resnet'), # normalize
                            'dataset': dataset,
                            'method': method,
                            'ratio': ratio if ratio else 0.0,
                            'pruned_acc': acc
                        }
                        all_results.append(pd.DataFrame([rw]))
                        print(f"Found {d}: {acc:.2f}%")
                except Exception as e:
                    pass

    # 3. Look for comprehensive CSVs
    comp_files = glob.glob(os.path.join(root_dir, "comprehensive", "*.csv"))
    for f in comp_files:
        try:
            df = pd.read_csv(f)
            # normalize columns
            if 'pruned_acc' in df.columns:
                 # Standardize to just 5 columns
                 df = df[['architecture', 'dataset', 'method', 'ratio', 'pruned_acc']]
                 all_results.append(df)
        except:
            pass

    if all_results:
        # Filter all DFs to ensure they have the right columns
        clean_results = []
        for d in all_results:
             if 'pruned_acc' in d.columns:
                 # Ensure ratio is numeric
                 d['ratio'] = pd.to_numeric(d['ratio'], errors='coerce')
                 clean_results.append(d[['architecture', 'dataset', 'method', 'ratio', 'pruned_acc']])
                 
        if not clean_results:
             print("No valid results found after cleaning.")
             return

        final_df = pd.concat(clean_results, ignore_index=True)
        # Clean up
        final_df['ratio'] = pd.to_numeric(final_df['ratio'], errors='coerce')
        final_df['pruned_acc'] = pd.to_numeric(final_df['pruned_acc'], errors='coerce')
        final_df = final_df.dropna()
        
        # Deduplicate: keep highest accuracy for each setting (optimistic)
        final_df = final_df.sort_values('pruned_acc', ascending=False)
        final_df = final_df.drop_duplicates(subset=['architecture', 'dataset', 'method', 'ratio'])
        
        print("\nConsolidated Data Summary:")
        print(final_df.groupby(['architecture', 'dataset', 'method'])['pruned_acc'].count())
        
        output_path = os.path.join(root_dir, 'final_consolidated_results.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\nSaved consolidated results to {output_path}")
    else:
        print("No results found!")

if __name__ == '__main__':
    harvest_data(r'C:\GRA-CNN\experiments')
