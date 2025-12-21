import os
import sys
import argparse
import torch
from datasets.tinyimagenet import get_tinyimagenet_dataloader
from models.resnet18_tiny import ResNet18Tiny as resnet18
from pruning.prune_model import prune_model
from pruning.gra_score import GrayRelationalChannelScorer
from train.train_supervised import train_model, test_model
from utils.flops_counter import get_model_complexity_info

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_root = r"C:\GRA-CNN\data\tiny-imagenet-200"
    save_dir = r"C:\GRA-CNN\experiments"
    
    print("Loading Tiny-ImageNet...")
    train_loader, test_loader = get_tinyimagenet_dataloader(batch_size=64, num_workers=0, data_root=data_root)
    
    baseline_ckpt = os.path.join(save_dir, "baseline_tiny_resnet18_60ep.pth")
    
    # GRA Ratio 0.5
    print("\n--- Running Tiny-ImageNet GRA Ratio 0.5 ---")
    model = resnet18(num_classes=200).to(device)
    if os.path.exists(baseline_ckpt):
        model.load_state_dict(torch.load(baseline_ckpt))
    
    scorer = GrayRelationalChannelScorer(rho=0.5)
    pruned_model = prune_model(
        model, 
        scorer=scorer, 
        prune_ratio=0.5, 
        method='gra', 
        dataloader=train_loader, 
        device=device,
        model_type='resnet18_tiny'
    )
    
    save_path = os.path.join(save_dir, "tiny_r18_gra_0.5.pth")
    train_model(pruned_model, train_loader, test_loader, epochs=25, lr=0.01, device=device, save_path=save_path)

if __name__ == "__main__":
    main()
