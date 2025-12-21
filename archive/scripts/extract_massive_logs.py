import os
import re

def extract_best_acc(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # Look for "Result: Acc=XX.XX%"
        match = re.search(r"Result: Acc=([\d\.]+)%", content)
        if match:
            return float(match.group(1))
            
        # Fallback: look for "Best: XX.XX%" in the last few lines
        matches = re.findall(r"Best: ([\d\.]+)%", content)
        if matches:
            return float(matches[-1])
            
    except Exception:
        pass
    return None

def main():
    log_files = [f for f in os.listdir('.') if f.startswith('log_') and f.endswith('.txt')]
    
    results = []
    for log_f in log_files:
        # Parse filename: log_DATASET_MODEL_METHOD_RATIO.txt
        parts = log_f.replace('.txt', '').split('_')
        if len(parts) < 5: continue
        
        dataset = parts[1]
        model = parts[2]
        method = parts[3]
        ratio = parts[4]
        
        acc = extract_best_acc(log_f)
        if acc is not None:
            # Normalize names
            if dataset == 'cifar10': d_name = 'CIFAR-10'
            elif dataset == 'cifar100': d_name = 'CIFAR-100'
            else: d_name = dataset
            
            if model == 'resnet20': m_name = 'ResNet-20'
            elif model == 'resnet56': m_name = 'ResNet-56'
            elif model == 'resnet110': m_name = 'ResNet-110'
            else: m_name = model
            
            if method == 'l1': met_name = 'L1-Norm'
            elif method == 'gra': met_name = 'GRA-CNN'
            else: met_name = method
            
            results.append(f"{d_name},{m_name},{met_name},{ratio},0.5,{acc},0,0,0,0")
            
    with open('vis/extra_results_massive.csv', 'w') as f:
        for r in results:
            f.write(r + "\n")
            print(f"Extracted: {r}")

if __name__ == "__main__":
    main()
