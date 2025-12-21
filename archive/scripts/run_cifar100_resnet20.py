import argparse
import os
import torch
from models.resnet_cifar import resnet20
from datasets.cifar100 import get_cifar100_dataloader
from train.train_supervised import train_model
from pruning.prune_model import prune_resnet, build_new_resnet
from utils.flops_counter import get_model_complexity_info

def main():
    parser = argparse.ArgumentParser(description='GRA-CNN CIFAR-100 Pruning')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--epochs', type=int, default=160, help='Training epochs') # Full training
    parser.add_argument('--finetune_epochs', type=int, default=100, help='Finetuning epochs')
    parser.add_argument('--ratio', type=float, default=0.5, help='Pruning ratio')
    parser.add_argument('--rho', type=float, default=0.5, help='GRA rho')
    parser.add_argument('--method', type=str, default='gra', choices=['gra', 'l1'], help='Pruning method')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default='./experiments/cifar100_r20')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Data
    print("Loading data...")
    trainloader, testloader = get_cifar100_dataloader(mock=args.mock, num_workers=args.workers)
    
    # 2. Model (Pretrain)
    print("Initializing ResNet-20 for CIFAR-100...")
    model = resnet20(num_classes=100).to(device)
    
    # Check if pretrained exists (Check in default dir to reuse pretrain)
    default_dir = './experiments/cifar100_r20'
    if not os.path.exists(default_dir): os.makedirs(default_dir)
    pretrained_path = os.path.join(default_dir, 'resnet20_cifar100.pth')
    
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path))
    else:
        print("Training from scratch...")
        train_model(model, trainloader, testloader, epochs=args.epochs, save_path=pretrained_path, device=device)
    
    # 3. Pruning
    print(f"Pruning with ratio {args.ratio} using {args.method} (rho={args.rho})...")
    flops_old, params_old = get_model_complexity_info(model, (3, 32, 32))
    
    new_cfg, mask_dict = prune_resnet(model, args.ratio, method=args.method, dataloader=trainloader, device=device, rho=args.rho)
    
    # 4. Rebuild
    print("Rebuilding model...")
    new_model = build_new_resnet(model, new_cfg, mask_dict, model_type='resnet20', num_classes=100).to(device)
    
    flops_new, params_new = get_model_complexity_info(new_model, (3, 32, 32))
    print(f"FLOPs Reduction: {(1-flops_new/flops_old):.2%}")
    print(f"Params Reduction: {(1-params_new/params_old):.2%}")
    
    # 5. Finetune
    print("Finetuning...")
    finetuned_path = os.path.join(args.save_dir, f'resnet20_cifar100_pruned_{args.method}_{args.ratio}_{args.rho}.pth')
    best_acc = train_model(new_model, trainloader, testloader, epochs=args.finetune_epochs, save_path=finetuned_path, device=device)
    
    print(f"Final Pruned Accuracy: {best_acc:.2f}%")
    
    # Save CSV
    csv_path = os.path.join(args.save_dir, f'results_cifar100_{args.method}_{args.ratio}_{args.rho}.csv')
    with open(csv_path, "w") as f:
        f.write(f"Architecture,Method,PruningRatio,Rho,Accuracy\n")
        f.write(f"resnet20,{args.method},{args.ratio},{args.rho},{best_acc:.2f}\n")

if __name__ == '__main__':
    main()
