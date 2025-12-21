import torch
import sys
import os
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_cifar import resnet56
from datasets.cifar10 import get_cifar10_dataloader
from pruning.prune_model import prune_model
from pruning.l1_score import L1ChannelScorer
from pruning.gra_score import GrayRelationalChannelScorer
from pruning.fpgm_score import FPGMChannelScorer
from pruning.hrank_score import HRankChannelScorer
from utils.flops_counter import get_model_complexity_info

def get_scorer(method, model, rho=0.5):
    if method == 'l1': return L1ChannelScorer(model)
    elif method == 'fpgm': return FPGMChannelScorer(model)
    elif method == 'hrank': return HRankChannelScorer(model)
    elif method == 'gra': return GrayRelationalChannelScorer(rho=rho)
    return None

def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' # Force CPU due to RTX 5090 incompatibility
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading data...")
    train_loader, _ = get_cifar10_dataloader(batch_size=128, num_workers=0)
    
    methods = ['l1', 'fpgm', 'hrank', 'gra']
    ratios = [0.3, 0.5, 0.7]
    
    print("Method,Ratio,FLOPs,Params,FLOPs_Red,Params_Red")
    
    # Baseline
    model = resnet56(num_classes=10).to(device)
    flops_base, params_base = get_model_complexity_info(model, (3, 32, 32))
    print(f"Baseline,-,{flops_base},{params_base},0,0")
    
    ckpt_path = r"C:\GRA-CNN\experiments\baseline_cifar10_resnet56.pth"
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using random weights.")
    
    for m in methods:
        for r in ratios:
            # Reload model
            model = resnet56(num_classes=10).to(device)
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path))
                
            scorer = get_scorer(m, model)
            
            try:
                # Suppress print from prune_model
                # sys.stdout = open(os.devnull, 'w')
                pruned_model = prune_model(
                    model, 
                    scorer=scorer, 
                    prune_ratio=r, 
                    method=m, 
                    dataloader=train_loader, 
                    device=device,
                    model_type='resnet56'
                )
                # sys.stdout = sys.__stdout__
                
                flops, params = get_model_complexity_info(pruned_model, (3, 32, 32))
                flops_red = 100 * (1 - flops/flops_base)
                params_red = 100 * (1 - params/params_base)
                
                print(f"{m},{r},{flops},{params},{flops_red:.2f},{params_red:.2f}")
            except Exception as e:
                # sys.stdout = sys.__stdout__
                print(f"{m},{r},ERROR,{e}")

if __name__ == "__main__":
    main()
