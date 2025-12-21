import os
import sys
import argparse
import torch
from datasets.tinyimagenet import get_tinyimagenet_dataloader
from models.resnet18_tiny import ResNet18Tiny as resnet18
from pruning.prune_model import prune_model
from pruning.l1_score import L1ChannelScorer
from pruning.gra_score import GrayRelationalChannelScorer
from train.train_supervised import train_model, test_model
from utils.flops_counter import get_model_complexity_info

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # TinyImageNet path
    data_root = r"C:\GRA-CNN\data\tiny-imagenet-200"
    save_dir = r"C:\GRA-CNN\experiments"
    
    print("Loading Tiny-ImageNet...")
    train_loader, test_loader = get_tinyimagenet_dataloader(data_root, batch_size=64, num_workers=0)
    
    # Load or Train Baseline
    model = resnet18(num_classes=200).to(device)
    baseline_ckpt = os.path.join(save_dir, "baseline_tiny_resnet18_60ep.pth")
    
    if os.path.exists(baseline_ckpt):
        print(f"Loading baseline from {baseline_ckpt}")
        model.load_state_dict(torch.load(baseline_ckpt))
    else:
        print("Baseline not found! Training from scratch (this will take a while)...")
        # train_model(model, train_loader, test_loader, epochs=60, lr=0.1, device=device, save_path=baseline_ckpt)
        # For now, assume it exists or we skip
        return

    # Evaluate Baseline
    flops_base, params_base = get_model_complexity_info(model, (3, 64, 64))
    acc_base = test_model(model, test_loader, device)
    print(f"Baseline: Acc={acc_base:.2f}%, FLOPs={flops_base/1e6:.2f}M, Params={params_base/1e6:.2f}M")
    
    # Experiments
    configs = [
        ('l1', 0.5),
        ('gra', 0.5)
    ]
    
    for method, ratio in configs:
        print(f"\n--- Running Tiny-ImageNet {method.upper()} Ratio {ratio} ---")
        
        # Reload baseline
        model.load_state_dict(torch.load(baseline_ckpt))
        
        if method == 'l1':
            scorer = L1ChannelScorer(model)
        elif method == 'gra':
            scorer = GrayRelationalChannelScorer(rho=0.5)
            
        pruned_model = prune_model(
            model, 
            scorer=scorer, 
            prune_ratio=ratio, 
            method=method, 
            dataloader=train_loader, 
            device=device,
            model_type='resnet18_tiny'
        )
        
        # Finetune
        ft_epochs = 25
        print(f"Finetuning for {ft_epochs} epochs...")
        save_path = os.path.join(save_dir, f"tiny_r18_{method}_{ratio}.pth")
        acc_pruned = train_model(pruned_model, train_loader, test_loader, epochs=ft_epochs, lr=0.01, device=device, save_path=save_path)
        
        # Metrics
        flops_pruned, params_pruned = get_model_complexity_info(pruned_model, (3, 64, 64))
        flops_red = 100 * (1 - flops_pruned/flops_base)
        params_red = 100 * (1 - params_pruned/params_base)
        
        print(f"Result {method} {ratio}: Acc={acc_pruned:.2f}%, FLOPs Red={flops_red:.2f}%, Params Red={params_red:.2f}%")
        
        # Save to results
        with open("vis/results.csv", "a") as f:
            f.write(f"Tiny-ImageNet,ResNet-18,{method.upper() if method=='l1' else 'GRA-CNN'},{ratio},0.5,{acc_pruned:.2f},{flops_pruned},{params_pruned},{flops_red:.2f},{params_red:.2f}\n")

if __name__ == "__main__":
    main()
