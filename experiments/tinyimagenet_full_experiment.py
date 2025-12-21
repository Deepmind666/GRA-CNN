"""
Complete Tiny-ImageNet-200 Experiment Suite for GRA-CNN
Runs comprehensive experiments: GRA vs L1 vs FPGM at multiple pruning ratios
Designed for RTX 5090 32GB - can run all experiments efficiently
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ImageNet normalization (also used for Tiny-ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_tiny_imagenet_loaders(data_dir, batch_size=128, num_workers=4):
    """Load Tiny-ImageNet-200 dataset"""
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Check if val directory needs restructuring (Tiny-ImageNet specific)
    val_images_dir = os.path.join(val_dir, 'images')
    if os.path.exists(val_images_dir):
        print("Restructuring validation directory...")
        restructure_val_dir(val_dir)
    
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    valset = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    
    return trainloader, valloader


def restructure_val_dir(val_dir):
    """Restructure Tiny-ImageNet validation directory"""
    import shutil
    
    val_images = os.path.join(val_dir, 'images')
    annotations = os.path.join(val_dir, 'val_annotations.txt')
    
    if not os.path.exists(annotations):
        return
        
    # Read annotations
    with open(annotations, 'r') as f:
        lines = f.readlines()
    
    # Create class directories and move images
    for line in lines:
        parts = line.strip().split('\t')
        img_name = parts[0]
        class_name = parts[1]
        
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        src = os.path.join(val_images, img_name)
        dst = os.path.join(class_dir, img_name)
        
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    
    # Clean up
    if os.path.exists(val_images):
        shutil.rmtree(val_images)
    if os.path.exists(annotations):
        os.remove(annotations)


class TinyImageNetResNet(nn.Module):
    """ResNet-18 adapted for Tiny-ImageNet (64x64 images, 200 classes)"""
    
    def __init__(self, pretrained=True, num_classes=200):
        super().__init__()
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for 64x64 images (smaller kernel, no stride)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove max pooling for smaller images
        self.model.maxpool = nn.Identity()
        
        # Modify final FC layer for 200 classes
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)


def calc_gra_score(activations, logits, targets, rho=0.5):
    """Calculate Gray Relational Analysis score"""
    # Global Average Pooling
    A = activations.mean(dim=[2, 3])  # [B, C]
    
    # Get logit of correct class
    Z = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
    
    A = A.cpu().numpy()
    Z = Z.cpu().numpy()
    
    # Min-max normalization
    A_min = A.min(axis=0, keepdims=True)
    A_max = A.max(axis=0, keepdims=True)
    A_norm = (A - A_min) / (A_max - A_min + 1e-8)
    
    Z_min = Z.min()
    Z_max = Z.max()
    Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-8)
    Z_norm = Z_norm[:, np.newaxis]
    
    # Gray relational coefficient
    diff = np.abs(A_norm - Z_norm)
    delta_min = diff.min()
    delta_max = diff.max()
    xi = (delta_min + rho * delta_max) / (diff + rho * delta_max + 1e-8)
    
    # Gray relational grade
    gamma = xi.mean(axis=0)
    return gamma


def get_gra_scores(model, dataloader, device, rho=0.5, num_batches=50):
    """Get GRA scores for all prunable conv layers"""
    model.eval()
    
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks on conv1 of each block (typical pruning targets)
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'conv1' in name and 'layer' in name:
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    all_scores = {}
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="GRA Scoring")):
            if batch_idx >= num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            for name, feats in activations.items():
                score = calc_gra_score(feats, logits, targets, rho)
                if name not in all_scores:
                    all_scores[name] = []
                all_scores[name].append(score)
    
    for h in hooks:
        h.remove()
    
    return {name: np.mean(scores, axis=0) for name, scores in all_scores.items()}


def get_l1_scores(model):
    """Get L1-norm scores"""
    scores = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'conv1' in name and 'layer' in name:
                weight = module.weight.data
                l1_norm = weight.abs().sum(dim=[1, 2, 3])
                scores[name] = l1_norm.cpu().numpy()
    return scores


def get_fpgm_scores(model):
    """Get FPGM scores (geometric median distance)"""
    scores = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'conv1' in name and 'layer' in name:
                weight = module.weight.data.cpu().numpy()
                n_filters = weight.shape[0]
                weight_flat = weight.reshape(n_filters, -1)
                
                # Calculate pairwise distances
                distances = np.zeros(n_filters)
                for i in range(n_filters):
                    dist = np.linalg.norm(weight_flat - weight_flat[i], axis=1)
                    distances[i] = dist.sum()
                
                scores[name] = distances
    return scores


def prune_model(model, scores, prune_ratio):
    """Apply pruning masks based on scores"""
    all_scores = np.concatenate([s.flatten() for s in scores.values()])
    threshold = np.percentile(all_scores, prune_ratio * 100)
    
    masks = {}
    pruned_count = 0
    total_count = 0
    
    for name, score in scores.items():
        mask = score > threshold
        # Ensure minimum channels
        min_channels = max(4, int(len(score) * 0.1))
        if mask.sum() < min_channels:
            top_indices = np.argsort(score)[-min_channels:]
            mask = np.zeros_like(score, dtype=bool)
            mask[top_indices] = True
        
        masks[name] = mask
        pruned_count += (~mask).sum()
        total_count += len(score)
    
    # Apply masks
    with torch.no_grad():
        for name, module in model.model.named_modules():
            if name in masks:
                mask_tensor = torch.tensor(masks[name], device=module.weight.device).float()
                module.weight.data *= mask_tensor.view(-1, 1, 1, 1)
    
    actual_ratio = pruned_count / total_count
    print(f"Pruned {pruned_count}/{total_count} channels ({actual_ratio:.1%})")
    
    return masks


def evaluate(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'loss': total_loss/total, 'acc': 100.*correct/total})
    
    return 100. * correct / total


def finetune(model, trainloader, valloader, device, epochs=50, lr=0.01):
    """Finetune pruned model"""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_acc = evaluate(model, valloader, device)
        scheduler.step()
        
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    return best_acc


def run_single_experiment(data_dir, method, prune_ratio, rho, finetune_epochs, device, save_dir):
    """Run a single pruning experiment"""
    print(f"\n{'='*60}")
    print(f"Experiment: {method.upper()}, Ratio={prune_ratio}, Rho={rho}")
    print(f"{'='*60}")
    
    # Load data
    trainloader, valloader = get_tiny_imagenet_loaders(data_dir, batch_size=128)
    
    # Create model
    model = TinyImageNetResNet(pretrained=True, num_classes=200)
    model = model.to(device)
    
    # Baseline
    print("\nEvaluating baseline...")
    baseline_acc = evaluate(model, valloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # First finetune baseline on Tiny-ImageNet
    print("\nFinetuning baseline on Tiny-ImageNet...")
    finetune(model, trainloader, valloader, device, epochs=10, lr=0.001)
    baseline_after = evaluate(model, valloader, device)
    print(f"Baseline after adaptation: {baseline_after:.2f}%")
    
    # Get scores
    print(f"\nComputing {method.upper()} scores...")
    if method == 'gra':
        scores = get_gra_scores(model, trainloader, device, rho)
    elif method == 'l1':
        scores = get_l1_scores(model)
    elif method == 'fpgm':
        scores = get_fpgm_scores(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Prune
    print(f"\nPruning with ratio {prune_ratio}...")
    prune_model(model, scores, prune_ratio)
    
    # Evaluate after pruning
    pruned_acc = evaluate(model, valloader, device)
    print(f"Accuracy after pruning: {pruned_acc:.2f}%")
    
    # Finetune
    print(f"\nFinetuning for {finetune_epochs} epochs...")
    final_acc = finetune(model, trainloader, valloader, device, finetune_epochs)
    print(f"Final accuracy: {final_acc:.2f}%")
    
    # Save results
    results = {
        'method': method,
        'prune_ratio': prune_ratio,
        'rho': rho,
        'baseline_acc': baseline_after,
        'pruned_acc': pruned_acc,
        'final_acc': final_acc,
        'finetune_epochs': finetune_epochs,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(save_dir, f'tinyimagenet_{method}_{prune_ratio}_{rho}.csv')
    pd.DataFrame([results]).to_csv(results_file, index=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Tiny-ImageNet-200 GRA-CNN Experiments')
    parser.add_argument('--data-dir', type=str, default='./data/tiny-imagenet-200')
    parser.add_argument('--save-dir', type=str, default='./experiments/tinyimagenet')
    parser.add_argument('--finetune-epochs', type=int, default=50)
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.3, 0.5, 0.7])
    parser.add_argument('--methods', type=str, nargs='+', default=['gra', 'l1', 'fpgm'])
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--single', action='store_true', help='Run single experiment')
    parser.add_argument('--method', type=str, default='gra')
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    all_results = []
    
    if args.single:
        # Run single experiment
        results = run_single_experiment(
            args.data_dir, args.method, args.ratio, args.rho,
            args.finetune_epochs, device, args.save_dir
        )
        all_results.append(results)
    else:
        # Run all experiments
        total = len(args.methods) * len(args.ratios)
        current = 0
        
        for method in args.methods:
            for ratio in args.ratios:
                current += 1
                print(f"\n\n{'#'*60}")
                print(f"Experiment {current}/{total}")
                print(f"{'#'*60}")
                
                try:
                    results = run_single_experiment(
                        args.data_dir, method, ratio, args.rho,
                        args.finetune_epochs, device, args.save_dir
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"Error in experiment: {e}")
                    continue
    
    # Save combined results
    if all_results:
        combined_file = os.path.join(args.save_dir, 'all_results.csv')
        pd.DataFrame(all_results).to_csv(combined_file, index=False)
        print(f"\n\nAll results saved to {combined_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
