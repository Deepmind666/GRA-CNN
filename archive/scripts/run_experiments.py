import argparse
import os
import sys
import time
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.dirname(__file__))

from models.resnet_cifar import resnet20, resnet56, resnet110
from models.resnet18_tiny import ResNet18Tiny as resnet18
from datasets.cifar10 import get_cifar10_dataloader
from datasets.cifar100 import get_cifar100_dataloader
from datasets.tinyimagenet import get_tinyimagenet_dataloader
from pruning.l1_score import L1ChannelScorer
from pruning.gra_score import GrayRelationalChannelScorer
from pruning.fpgm_score import FPGMChannelScorer
from pruning.hrank_score import HRankChannelScorer
from pruning.prune_model import prune_model
from train.train_supervised import train_model, test_model
from utils.flops_counter import get_model_complexity_info

def get_scorer(method, model, rho=0.5):
    if method == 'l1':
        return L1ChannelScorer(model)
    elif method == 'fpgm':
        return FPGMChannelScorer(model)
    elif method == 'hrank':
        return HRankChannelScorer(model)
    elif method == 'gra':
        return GrayRelationalChannelScorer(rho=rho)
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    parser = argparse.ArgumentParser(description='Pruning Experiments')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('--model', type=str, required=True, choices=['resnet20', 'resnet56', 'resnet110', 'resnet18'])
    parser.add_argument('--method', type=str, required=True, choices=['l1', 'fpgm', 'hrank', 'gra'])
    parser.add_argument('--prune_ratio', type=float, required=True)
    parser.add_argument('--rho', type=float, default=0.5, help='GRA resolution coefficient')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='experiments')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.model}_{args.method}_{args.prune_ratio}")
    os.makedirs(save_path, exist_ok=True)

    print(f"Running Experiment: {args.dataset} | {args.model} | {args.method} | {args.prune_ratio}")

    # 1. Load Dataset
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_dataloader(batch_size=args.batch_size, num_workers=args.workers, mock=args.mock)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar100_dataloader(batch_size=args.batch_size, num_workers=args.workers, mock=args.mock)
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        # TinyImageNet needs correct path
        data_root = r"C:\GRA-CNN\data\tiny-imagenet-200" 
        train_loader, test_loader = get_tinyimagenet_dataloader(data_root, batch_size=args.batch_size, num_workers=args.workers, mock=args.mock)
        num_classes = 200

    # 2. Load Model
    if args.model == 'resnet20':
        model = resnet20(num_classes=num_classes).to(device)
    elif args.model == 'resnet56':
        model = resnet56(num_classes=num_classes).to(device)
    elif args.model == 'resnet110':
        model = resnet110(num_classes=num_classes).to(device)
    elif args.model == 'resnet18':
        model = resnet18(num_classes=num_classes).to(device) # Adapted for Tiny

    # 3. Train Baseline (or load)
    # Ideally we load a pretrained baseline to save time, or train if not exists.
    # For this comprehensive run, let's assume we might need to train.
    # To speed up, we check if baseline checkpoint exists.
    baseline_ckpt = os.path.join(args.save_dir, f"baseline_{args.dataset}_{args.model}.pth")
    
    if os.path.exists(baseline_ckpt):
        print(f"Loading baseline from {baseline_ckpt}")
        try:
            model.load_state_dict(torch.load(baseline_ckpt))
        except Exception:
            print("Baseline checkpoint invalid. Retraining baseline...")
            train_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device, save_path=baseline_ckpt)
    else:
        print("Training baseline...")
        train_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, device=device, save_path=baseline_ckpt)

    # Baseline Metrics
    flops_base, params_base = get_model_complexity_info(model, (3, 32, 32) if 'cifar' in args.dataset else (3, 64, 64))
    acc_base = test_model(model, test_loader, device)
    print(f"Baseline: Acc={acc_base:.2f}%, FLOPs={flops_base/1e6:.2f}M, Params={params_base/1e6:.2f}M")

    # 4. Apply Pruning
    scorer = get_scorer(args.method, model, rho=args.rho)
    
    # HRank and GRA need data to compute scores
    pruned_model = prune_model(
        model, 
        scorer=scorer, 
        prune_ratio=args.prune_ratio, 
        method=args.method, 
        dataloader=train_loader, 
        device=device, 
        rho=args.rho,
        model_type=args.model
    )
    
    # 5. Finetune
    print("Finetuning pruned model...")
    finetuned_ckpt = os.path.join(save_path, "finetuned.pth")
    train_model(pruned_model, train_loader, test_loader, epochs=args.finetune_epochs, lr=args.lr*0.1, device=device, save_path=finetuned_ckpt)
    
    # 6. Evaluate
    flops_pruned, params_pruned = get_model_complexity_info(pruned_model, (3, 32, 32) if 'cifar' in args.dataset else (3, 64, 64))
    acc_pruned = test_model(pruned_model, test_loader, device)
    
    flops_reduction = 100 * (1 - flops_pruned/flops_base)
    params_reduction = 100 * (1 - params_pruned/params_base)
    
    print(f"Result: Acc={acc_pruned:.2f}%, FLOPs Red={flops_reduction:.2f}%, Params Red={params_reduction:.2f}%")
    
    # 7. Save Results
    results_file = os.path.join(args.save_dir, "results_comprehensive.csv")
    file_exists = os.path.exists(results_file)
    
    with open(results_file, 'a') as f:
        if not file_exists:
            f.write("Dataset,Model,Method,PruningRatio,Rho,Accuracy,FLOPs,Params,FLOPs_Red,Params_Red\n")
        f.write(f"{args.dataset},{args.model},{args.method},{args.prune_ratio},{args.rho},{acc_pruned:.4f},{flops_pruned},{params_pruned},{flops_reduction:.2f},{params_reduction:.2f}\n")

if __name__ == '__main__':
    main()
