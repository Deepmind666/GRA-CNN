import os
import sys
import argparse
import torch
import pandas as pd
from datasets.tinyimagenet import get_tinyimagenet_dataloader
from models.resnet18_tiny import ResNet18Tiny
from pruning.prune_model import prune_model
from pruning.l1_score import L1ChannelScorer
from pruning.fpgm_score import FPGMChannelScorer
from pruning.gra_score import GrayRelationalChannelScorer
from train.train_supervised import train_model
from utils.flops_counter import get_model_complexity_info

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = r"C:\GRA-CNN\data\tiny-imagenet-200"
    
    train_loader, test_loader = get_tinyimagenet_dataloader(batch_size=64, num_workers=0, data_root=data_root)
    
    # Experiments
    # Methods: l1, fpgm, gra
    # Ratios: 0.3, 0.5
    configs = []
    for m in ['l1', 'fpgm', 'gra']:
        for r in [0.3, 0.5]:
            configs.append((m, r))
            
    results = []
    
    for method, ratio in configs:
        print(f"Running Tiny-ImageNet {method} {ratio}")
        
        model = ResNet18Tiny(num_classes=200).to(device)
        # Ideally load baseline checkpoint
        
        if method == 'l1': scorer = L1ChannelScorer(model)
        elif method == 'fpgm': scorer = FPGMChannelScorer(model)
        else: scorer = GrayRelationalChannelScorer(rho=0.5)
        
        pruned_model = prune_model(model, scorer, ratio, method, train_loader, device, model_type='resnet18_tiny')
        
        acc = train_model(pruned_model, train_loader, test_loader, epochs=30, lr=0.01, device=device)
        
        flops, params = get_model_complexity_info(pruned_model, (3, 64, 64))
        
        results.append({
            'method': method, 'ratio': ratio, 'acc': acc, 'flops': flops, 'params': params
        })
        
    df = pd.DataFrame(results)
    df.to_csv('vis/results_tiny_extended.csv', index=False)

if __name__ == "__main__":
    main()
