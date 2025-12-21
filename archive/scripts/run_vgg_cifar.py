import os
import sys
import argparse
import torch
import pandas as pd
from datasets.cifar10 import get_cifar10_dataloader
from models.vgg_cifar import vgg16
from pruning.prune_model import prune_model
from pruning.l1_score import L1ChannelScorer
from pruning.gra_score import GrayRelationalChannelScorer
from train.train_supervised import train_model
from utils.flops_counter import get_model_complexity_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running VGG-16 CIFAR-10 {args.method} {args.ratio}")
    
    train_loader, test_loader = get_cifar10_dataloader(batch_size=128, num_workers=0)
    
    # Load Baseline (Random init for speed/demo, usually pretrained)
    model = vgg16(num_classes=10).to(device)
    
    # Prune
    if args.method == 'l1': scorer = L1ChannelScorer(model)
    else: scorer = GrayRelationalChannelScorer(rho=0.5)
    
    pruned_model = prune_model(model, scorer, args.ratio, args.method, train_loader, device, model_type='vgg16')
    
    # Finetune
    acc = train_model(pruned_model, train_loader, test_loader, epochs=40, lr=0.01, device=device)
    
    # Stats
    flops, params = get_model_complexity_info(pruned_model, (3, 32, 32))
    
    # Save
    res_file = 'vis/results_vgg.csv'
    is_new = not os.path.exists(res_file)
    with open(res_file, 'a') as f:
        if is_new:
            f.write("method,ratio,acc,flops,params\n")
        f.write(f"{args.method},{args.ratio},{acc:.2f},{flops},{params}\n")

if __name__ == "__main__":
    main()
