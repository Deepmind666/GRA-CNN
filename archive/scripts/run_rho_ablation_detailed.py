import os
import sys
import argparse
import torch
import pandas as pd
from datasets.cifar10 import get_cifar10_dataloader
from models.resnet_cifar import resnet20, resnet56
from pruning.prune_model import prune_model
from pruning.gra_score import GrayRelationalChannelScorer
from train.train_supervised import train_model
from utils.flops_counter import get_model_complexity_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ablation: Try different rho values
    rhos = [0.3, 0.4, 0.6, 0.7] # 0.5 is already done
    
    train_loader, test_loader = get_cifar10_dataloader(batch_size=128, num_workers=0)
    
    if args.model == 'resnet20':
        model_fn = resnet20
        m_type = 'resnet20'
    elif args.model == 'resnet56':
        model_fn = resnet56
        m_type = 'resnet56'
        
    res_file = 'vis/results_rho_ablation_detailed.csv'
    if not os.path.exists(res_file):
        with open(res_file, 'w') as f:
            f.write("model,ratio,rho,acc\n")
            
    for rho in rhos:
        print(f"Running Rho Ablation: {args.model} Ratio={args.ratio} Rho={rho}")
        model = model_fn(num_classes=10).to(device)
        scorer = GrayRelationalChannelScorer(rho=rho)
        pruned = prune_model(model, scorer, args.ratio, 'gra', train_loader, device, rho=rho, model_type=m_type)
        
        acc = train_model(pruned, train_loader, test_loader, epochs=30, lr=0.01, device=device)
        
        with open(res_file, 'a') as f:
            f.write(f"{args.model},{args.ratio},{rho},{acc:.2f}\n")

if __name__ == "__main__":
    main()
