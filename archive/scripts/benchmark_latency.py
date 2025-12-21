import time
import torch
import pandas as pd
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_cifar import resnet20, resnet56
from models.resnet18_tiny import ResNet18Tiny
from pruning.prune_model import prune_model
from pruning.l1_score import L1ChannelScorer
from datasets.cifar10 import get_cifar10_dataloader

def measure_latency(model, input_shape=(1, 3, 32, 32), device='cuda', iterations=200, warmup=50):
    model.eval()
    model.to(device)
    input_tensor = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            
    # Sync
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    avg_time = total_time / iterations # seconds
    return avg_time * 1000 # ms

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}...")
    
    # Setup scenarios
    # (dataset, model_name, prune_ratio)
    scenarios = [
        ('cifar10', 'resnet20', 0.0),
        ('cifar10', 'resnet20', 0.5),
        ('cifar10', 'resnet56', 0.0),
        ('cifar10', 'resnet56', 0.5),
        ('tinyimagenet', 'resnet18', 0.0),
        ('tinyimagenet', 'resnet18', 0.5),
    ]
    
    results = []
    
    # Dummy loader for pruning structure
    train_loader, _ = get_cifar10_dataloader(batch_size=1)
    
    for dataset, model_name, ratio in scenarios:
        print(f"Benchmarking {model_name} on {dataset} (Ratio={ratio})...")
        
        # 1. Create Model
        if model_name == 'resnet20':
            model = resnet20(num_classes=10)
            model_type = 'resnet20'
            input_res = 32
        elif model_name == 'resnet56':
            model = resnet56(num_classes=10)
            model_type = 'resnet56'
            input_res = 32
        elif model_name == 'resnet18':
            model = ResNet18Tiny(num_classes=200)
            model_type = 'resnet18_tiny'
            input_res = 64
            
        model.to(device)
        
        # 2. Prune if needed
        if ratio > 0:
            # Use L1 scorer just to get the structure
            scorer = L1ChannelScorer(model)
            model = prune_model(model, scorer, ratio, 'l1', train_loader, device, model_type=model_type)
            model.to(device)
            
        # 3. Measure
        batch_size = 128
        input_shape = (batch_size, 3, input_res, input_res)
        
        latency_batch = measure_latency(model, input_shape, device)
        latency_img = latency_batch / batch_size
        fps = 1000.0 / latency_batch * batch_size
        
        print(f"  -> Latency: {latency_batch:.2f} ms/batch")
        print(f"  -> Throughput: {fps:.0f} img/s")
        
        results.append({
            'dataset': dataset,
            'model': model_name,
            'ratio': ratio,
            'latency_ms': latency_batch,
            'fps': fps
        })
        
    # Save
    df = pd.DataFrame(results)
    df.to_csv('vis/latency_results.csv', index=False)
    print("Saved to vis/latency_results.csv")

if __name__ == "__main__":
    main()
