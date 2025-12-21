import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import csv
import numpy as np
from models.resnet_cifar import resnet20
from pruning.gra_score import GrayRelationalChannelScorer

def get_l1_scores(model):
    l1_scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filters = module.weight.data
            l1 = filters.abs().view(filters.size(0), -1).sum(dim=1)
            l1_scores[name] = l1.cpu().numpy()
    return l1_scores

def get_gra_scores(model, dataloader, device):
    gra_scores = {}
    
    # Define hooks to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(get_activation(name)))
            
    model.eval()
    with torch.no_grad():
        try:
            inputs, targets = next(iter(dataloader))
        except StopIteration:
            # Handle empty dataloader if needed, but unlikely here
            return gra_scores
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        logits = model(inputs)
        
        # Reference sequence: Target class logits
        batch_size = logits.size(0)
        ref_seq = logits[range(batch_size), targets] # Shape [B]
        
        # Normalize Reference
        ref_min = ref_seq.min()
        ref_max = ref_seq.max()
        if ref_max - ref_min > 1e-6:
            ref_seq = (ref_seq - ref_min) / (ref_max - ref_min)
        else:
            ref_seq = torch.zeros_like(ref_seq)
            
        # Compute GRA for each layer
        for name, feats in activations.items():
            # feats: [B, C, H, W] -> GAP -> [B, C]
            gap_feats = feats.mean(dim=[2, 3]) 
            num_channels = gap_feats.size(1)
            layer_gra = []
            
            for c in range(num_channels):
                comp_seq = gap_feats[:, c]
                
                # Normalize Comparison Sequence
                c_min = comp_seq.min()
                c_max = comp_seq.max()
                if c_max - c_min > 1e-6:
                    comp_seq = (comp_seq - c_min) / (c_max - c_min)
                else:
                    comp_seq = torch.zeros_like(comp_seq)
                
                # GRA Calculation
                delta = (ref_seq - comp_seq).abs()
                rho = 0.5
                delta_min = delta.min()
                delta_max = delta.max()
                
                gra_value = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-8)
                score = gra_value.mean().item()
                layer_gra.append(score)
            
            gra_scores[name] = np.array(layer_gra)
            
    for h in hooks:
        h.remove()
        
    return gra_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    print("Loading ResNet-20...")
    model = resnet20()
    
    # Try loading baseline
    baseline_path = 'experiments/baseline_cifar10_resnet20.pth'
    if os.path.exists(baseline_path):
        try:
            state_dict = torch.load(baseline_path, map_location=device)
            if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            # Remove module. prefix if present
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded baseline from {baseline_path}")
        except Exception as e:
            print(f"Failed to load baseline: {e}")
    else:
        print("Baseline not found, using random weights (Warning: Scores will be meaningless)")
        
    model.to(device)
    
    # Data
    print("Loading Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    except:
        print("Could not load CIFAR-10, using mock data")
        inputs = torch.randn(128, 3, 32, 32)
        targets = torch.randint(0, 10, (128,))
        trainloader = [(inputs, targets)]
    
    # Compute Scores
    print("Computing L1 Scores...")
    l1_scores = get_l1_scores(model)
    
    print("Computing GRA Scores...")
    gra_scores = get_gra_scores(model, trainloader, device)
    
    # Save
    os.makedirs('vis', exist_ok=True)
    out_path = 'vis/correlation_scores.csv'
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'channel_idx', 'l1_score', 'gra_score'])
        
        for name in l1_scores:
            if name in gra_scores:
                l1 = l1_scores[name]
                gra = gra_scores[name]
                
                # Normalize
                if l1.max() > l1.min():
                    l1_norm = (l1 - l1.min()) / (l1.max() - l1.min())
                else:
                    l1_norm = l1
                    
                if gra.max() > gra.min():
                    gra_norm = (gra - gra.min()) / (gra.max() - gra.min())
                else:
                    gra_norm = gra
                
                for i in range(len(l1)):
                    writer.writerow([name, i, l1_norm[i], gra_norm[i]])
                    
    print(f"Saved correlation scores to {out_path}")

if __name__ == '__main__':
    main()
