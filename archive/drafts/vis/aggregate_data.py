import os
import pandas as pd
import glob

def aggregate():
    data = []
    exp_root = 'experiments'
    
    print(f"Scanning {exp_root}...")
    
    for root, dirs, files in os.walk(exp_root):
        dir_name = os.path.basename(root)
        parts = dir_name.split('_')
        
        # Default values from directory name
        # e.g. cifar10_resnet20_gra_0.5
        # e.g. resnet20_gra_r0.5_rho0.2
        
        d_dataset = 'CIFAR-10'
        d_model = 'ResNet-20'
        d_method = 'GRA-CNN'
        d_ratio = 0.5
        d_rho = 0.5
        
        # Heuristic parsing of directory name
        if 'cifar100' in dir_name: d_dataset = 'CIFAR-100'
        elif 'tiny' in dir_name: d_dataset = 'Tiny-ImageNet'
        elif 'cifar10' in dir_name: d_dataset = 'CIFAR-10'
        
        if 'resnet56' in dir_name: d_model = 'ResNet-56'
        elif 'resnet110' in dir_name: d_model = 'ResNet-110'
        elif 'resnet18' in dir_name: d_model = 'ResNet-18'
        elif 'resnet20' in dir_name: d_model = 'ResNet-20'
        
        if 'l1' in dir_name: d_method = 'L1-Norm'
        elif 'fpgm' in dir_name: d_method = 'FPGM'
        elif 'hrank' in dir_name: d_method = 'HRank'
        elif 'gra' in dir_name: d_method = 'GRA-CNN'
        
        # Parse ratio/rho
        for p in parts:
            try:
                if p.replace('.', '').isdigit():
                    val = float(p)
                    if val < 1.0 and val > 0: d_ratio = val
            except: pass
            
            if p.startswith('rho'):
                try: d_rho = float(p[3:])
                except: pass
            if p.startswith('r') and p[1].isdigit(): # r0.5
                try: d_ratio = float(p[1:])
                except: pass

        # Check for results.csv
        found_res = False
        if 'results.csv' in files or 'results_comprehensive.csv' in files:
            fname = 'results.csv' if 'results.csv' in files else 'results_comprehensive.csv'
            try:
                df = pd.read_csv(os.path.join(root, fname))
                for _, row in df.iterrows():
                    entry = {}
                    # Use row data if available, else defaults
                    entry['dataset'] = row.get('dataset', row.get('Dataset', d_dataset))
                    entry['model'] = row.get('model', row.get('Architecture', d_model))
                    entry['method'] = row.get('method', row.get('Method', d_method))
                    
                    # Ratio
                    r = row.get('prune_ratio', row.get('PruningRatio', row.get('ratio', d_ratio)))
                    entry['prune_ratio'] = float(r)
                    
                    # Rho
                    rh = row.get('rho', row.get('Rho', d_rho))
                    entry['rho'] = float(rh)
                    
                    # Accuracy
                    acc = row.get('test_acc', row.get('Accuracy', row.get('acc', row.get('acc_top1', 0))))
                    entry['accuracy'] = float(acc)

                    # FLOPs/Params
                    if 'FLOPs' in row: entry['flops'] = float(row['FLOPs'])
                    if 'Params' in row: entry['params'] = float(row['Params'])
                    if 'FLOPs_Red' in row: entry['flops_red'] = float(row['FLOPs_Red'])
                    if 'Params_Red' in row: entry['params_red'] = float(row['Params_Red'])
                    
                    if 'accuracy' in entry and entry['accuracy'] > 0:
                        data.append(entry)
                        found_res = True
            except: pass
            
        # If no results.csv, check training_log.csv or log_*.csv
        if not found_res:
            log_files = [f for f in files if f.endswith('.csv') and ('log' in f or 'finetuned' in f)]
            for log_f in log_files:
                try:
                    df = pd.read_csv(os.path.join(root, log_f))
                    best_acc = 0
                    if 'test_acc' in df.columns:
                        best_acc = df['test_acc'].max()
                    elif 'TestAccuracy' in df.columns:
                        best_acc = df['TestAccuracy'].max()
                        
                    if best_acc > 0:
                        entry = {
                            'dataset': d_dataset,
                            'model': d_model,
                            'method': d_method,
                            'prune_ratio': d_ratio,
                            'rho': d_rho,
                            'accuracy': best_acc
                        }
                        data.append(entry)
                        break # Only need one best acc per folder
                except: pass

    # Special files
    special_files = [
        ('tiny_prune50_results.csv', 'Tiny-ImageNet', 'ResNet-18'),
        ('scalability_results.csv', 'CIFAR-100', 'ResNet-20')
    ]
    for fname, default_ds, default_model in special_files:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                for _, row in df.iterrows():
                    entry = {}
                    entry['dataset'] = default_ds
                    entry['model'] = default_model
                    if 'Method' in row: entry['method'] = row['Method']
                    if 'Accuracy' in row: entry['accuracy'] = float(row['Accuracy'])
                    if 'Pruned Ratio' in row: 
                        val = row['Pruned Ratio']
                        if isinstance(val, str) and '%' in val:
                            val = float(val.strip('%')) / 100.0
                        entry['prune_ratio'] = float(val)
                    else: entry['prune_ratio'] = 0.5
                    entry['rho'] = 0.5
                    data.append(entry)
            except: pass

    # Clean and Save
    df = pd.DataFrame(data)
    if not df.empty:
        # Normalize
        df['dataset'] = df['dataset'].astype(str).replace({
            'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'tinyimagenet': 'Tiny-ImageNet', 'tinyimagenet200': 'Tiny-ImageNet'
        })
        df['model'] = df['model'].astype(str).replace({
            'resnet20': 'ResNet-20', 'resnet56': 'ResNet-56', 'resnet18': 'ResNet-18', 'resnet18tiny': 'ResNet-18'
        })
        df['method'] = df['method'].astype(str).replace({
            'l1': 'L1-Norm', 'gra': 'GRA-CNN', 'fpgm': 'FPGM', 'hrank': 'HRank',
            'l1-norm': 'L1-Norm', 'gra-cnn': 'GRA-CNN'
        })
        
        # Drop duplicates, keep best
        df = df.sort_values('accuracy', ascending=False).drop_duplicates(
            subset=['dataset', 'model', 'method', 'prune_ratio', 'rho']
        )
        
        df.to_csv('vis/results.csv', index=False)
        print(f"Saved {len(df)} entries to vis/results.csv")
    else:
        print("No data found.")

if __name__ == '__main__':
    aggregate()
