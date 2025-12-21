import argparse
import os
import torch
from models.resnet18_tiny import resnet18_tiny
from datasets.tinyimagenet import get_tinyimagenet_dataloader
from train.train_supervised import train_model
from pruning.prune_model import prune_resnet, build_new_resnet
from utils.flops_counter import get_model_complexity_info

def main():
    parser = argparse.ArgumentParser(description='GRA-CNN Tiny-ImageNet Pruning')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs') # Reduced for time constraint, usually need more
    parser.add_argument('--finetune_epochs', type=int, default=40, help='Finetuning epochs')
    parser.add_argument('--ratio', type=float, default=0.4, help='Pruning ratio')
    parser.add_argument('--rho', type=float, default=0.5, help='GRA rho')
    parser.add_argument('--save_dir', type=str, default='./experiments/tiny_r18')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Data
    print("Loading data...")
    trainloader, testloader = get_tinyimagenet_dataloader(mock=args.mock, data_root='./data/tiny-imagenet-200')
    
    # 2. Model (Pretrain)
    print("Initializing ResNet-18 for Tiny-ImageNet...")
    model = resnet18_tiny(num_classes=200).to(device)
    
    # Check if pretrained exists
    pretrained_path = os.path.join(args.save_dir, 'resnet18_tiny.pth')
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path))
    else:
        print("Training from scratch...")
        train_model(model, trainloader, testloader, epochs=args.epochs, save_path=pretrained_path, device=device)
    
    # 3. Pruning
    print(f"Pruning with ratio {args.ratio} using GRA (rho={args.rho})...")
    flops_old, params_old = get_model_complexity_info(model, (3, 64, 64))
    
    new_cfg, mask_dict = prune_resnet(model, args.ratio, method='gra', dataloader=trainloader, device=device, rho=args.rho)
    
    # 4. Rebuild
    print("Rebuilding model...")
    new_model = build_new_resnet(model, new_cfg, mask_dict, model_type='resnet18_tiny', num_classes=200).to(device)
    
    flops_new, params_new = get_model_complexity_info(new_model, (3, 64, 64))
    print(f"FLOPs Reduction: {(1-flops_new/flops_old):.2%}")
    print(f"Params Reduction: {(1-params_new/params_old):.2%}")
    
    # 5. Finetune
    print("Finetuning...")
    finetuned_path = os.path.join(args.save_dir, f'resnet18_tiny_pruned_{args.ratio}.pth')
    best_acc = train_model(new_model, trainloader, testloader, epochs=args.finetune_epochs, save_path=finetuned_path, device=device)
    
    print(f"Final Pruned Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
