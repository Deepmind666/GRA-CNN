"""
ImageNet-100 Subset Experiment for GRA-CNN
Uses PyTorch pretrained ResNet-50 and ImageNet-100 subset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_subset_loader(data_dir, batch_size=64, num_classes=100, num_workers=4, subset_size=None):
    """
    Load ImageNet or ImageNet-100 subset.
    If using full ImageNet, can specify subset_size for quick experiments.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Check if data directory exists
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir):
        print(f"ImageNet data not found at {data_dir}")
        print("Please download ImageNet or use --mock flag for testing")
        print("For ImageNet-100, you can use: https://www.kaggle.com/c/imagenet-object-localization-challenge")
        return None, None
    
    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    valset = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)
    
    # Subset to first N classes if needed
    if num_classes < 1000:
        # Get indices of first N classes
        class_to_idx = trainset.class_to_idx
        selected_classes = list(class_to_idx.keys())[:num_classes]
        selected_indices = [class_to_idx[c] for c in selected_classes]
        
        train_indices = [i for i, (_, label) in enumerate(trainset.samples) if label in selected_indices]
        val_indices = [i for i, (_, label) in enumerate(valset.samples) if label in selected_indices]
        
        trainset = torch.utils.data.Subset(trainset, train_indices)
        valset = torch.utils.data.Subset(valset, val_indices)
    
    # Further subset for quick experiments
    if subset_size and len(trainset) > subset_size:
        indices = np.random.choice(len(trainset), subset_size, replace=False)
        trainset = torch.utils.data.Subset(trainset, indices)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, valloader


def get_mock_imagenet_loader(batch_size=64, num_classes=100):
    """Mock dataloader for testing pipeline without ImageNet"""
    print("Using MOCK ImageNet data for pipeline testing...")
    trainset = torch.utils.data.TensorDataset(
        torch.randn(500, 3, 224, 224), 
        torch.randint(0, num_classes, (500,))
    )
    valset = torch.utils.data.TensorDataset(
        torch.randn(200, 3, 224, 224), 
        torch.randint(0, num_classes, (200,))
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


class PrunableResNet50(nn.Module):
    """ResNet-50 wrapper that supports channel pruning via cfg"""
    def __init__(self, num_classes=1000, pretrained=True, cfg=None):
        super().__init__()
        # Load pretrained ResNet-50
        self.model = models.resnet50(pretrained=pretrained)
        
        # Modify final FC for different num_classes
        if num_classes != 1000:
            self.model.fc = nn.Linear(2048, num_classes)
        
        self.cfg = cfg  # For pruned model reconstruction
        
    def forward(self, x):
        return self.model(x)
    
    def get_prunable_layers(self):
        """Return list of (name, module) tuples for prunable conv layers"""
        prunable = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Skip downsample layers and first conv
                if 'downsample' not in name and name != 'conv1':
                    prunable.append((name, module))
        return prunable


def calc_gra_score(activations, logits, targets, rho=0.5):
    """
    Calculate Gray Relational Analysis score for channel importance
    activations: [B, C, H, W]
    logits: [B, num_classes]
    targets: [B]
    """
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
    
    # Calculate difference sequence
    diff = np.abs(A_norm - Z_norm)
    
    # Gray relational coefficient
    delta_min = diff.min()
    delta_max = diff.max()
    xi = (delta_min + rho * delta_max) / (diff + rho * delta_max + 1e-8)
    
    # Gray relational grade (average over samples)
    gamma = xi.mean(axis=0)
    
    return gamma


def get_l1_scores_resnet50(model):
    """Get L1-norm scores for all prunable conv layers"""
    scores = {}
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'downsample' not in name and name != 'conv1':
                weight = module.weight.data
                l1_norm = weight.abs().sum(dim=[1, 2, 3])  # Sum over input channels and spatial
                scores[name] = l1_norm.cpu().numpy()
    return scores


def get_gra_scores_resnet50(model, dataloader, device, rho=0.5, num_batches=10):
    """Get GRA scores for all prunable conv layers"""
    model.eval()
    
    # Register hooks
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'downsample' not in name and name != 'conv1':
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Accumulate scores
    all_scores = {}
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Computing GRA scores")):
            if batch_idx >= num_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            for name, feats in activations.items():
                score = calc_gra_score(feats, logits, targets, rho)
                if name not in all_scores:
                    all_scores[name] = []
                all_scores[name].append(score)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Average scores across batches
    final_scores = {}
    for name, score_list in all_scores.items():
        final_scores[name] = np.mean(score_list, axis=0)
    
    return final_scores


def prune_resnet50(model, scores, prune_ratio=0.5):
    """
    Prune ResNet-50 based on scores
    Returns pruned model with new cfg
    """
    # Calculate global threshold
    all_scores = np.concatenate([s.flatten() for s in scores.values()])
    threshold = np.percentile(all_scores, prune_ratio * 100)
    
    # Generate mask for each layer
    masks = {}
    cfg = []
    
    for name, score in scores.items():
        mask = score > threshold
        # Ensure at least 10% channels remain
        min_channels = max(1, int(len(score) * 0.1))
        if mask.sum() < min_channels:
            # Keep top min_channels
            top_indices = np.argsort(score)[-min_channels:]
            mask = np.zeros_like(score, dtype=bool)
            mask[top_indices] = True
        
        masks[name] = mask
        cfg.append(int(mask.sum()))
    
    print(f"Original channels: {sum(len(s) for s in scores.values())}")
    print(f"Pruned channels: {sum(cfg)}")
    print(f"Actual pruning ratio: {1 - sum(cfg) / sum(len(s) for s in scores.values()):.2%}")
    
    return masks, cfg


def evaluate(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total


def finetune(model, trainloader, valloader, device, epochs=10, lr=0.001):
    """Finetune pruned model"""
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': train_loss/total, 'acc': 100.*correct/total})
        
        scheduler.step()
        
        # Evaluate
        val_acc = evaluate(model, valloader, device)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='ImageNet-100 GRA-CNN Pruning')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet', help='ImageNet data directory')
    parser.add_argument('--num-classes', type=int, default=100, help='Number of classes (100 for ImageNet-100)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--prune-ratio', type=float, default=0.5, help='Pruning ratio')
    parser.add_argument('--rho', type=float, default=0.5, help='GRA resolution coefficient')
    parser.add_argument('--finetune-epochs', type=int, default=10, help='Finetuning epochs')
    parser.add_argument('--method', type=str, default='gra', choices=['gra', 'l1'], help='Pruning method')
    parser.add_argument('--mock', action='store_true', help='Use mock data for testing')
    parser.add_argument('--save-dir', type=str, default='experiments/imagenet', help='Save directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    if args.mock:
        trainloader, valloader = get_mock_imagenet_loader(args.batch_size, args.num_classes)
    else:
        trainloader, valloader = get_imagenet_subset_loader(
            args.data_dir, args.batch_size, args.num_classes
        )
        if trainloader is None:
            print("Falling back to mock data...")
            trainloader, valloader = get_mock_imagenet_loader(args.batch_size, args.num_classes)
    
    # Load pretrained ResNet-50
    print("Loading pretrained ResNet-50...")
    model = PrunableResNet50(num_classes=args.num_classes, pretrained=True)
    model = model.to(device)
    
    # Baseline evaluation
    print("Evaluating baseline model...")
    baseline_acc = evaluate(model, valloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Get pruning scores
    print(f"Computing {args.method.upper()} scores...")
    if args.method == 'gra':
        scores = get_gra_scores_resnet50(model, trainloader, device, args.rho)
    else:
        scores = get_l1_scores_resnet50(model)
    
    # Prune
    print(f"Pruning with ratio {args.prune_ratio}...")
    masks, cfg = prune_resnet50(model, scores, args.prune_ratio)
    
    # Apply masks (zero out pruned channels)
    with torch.no_grad():
        for name, module in model.model.named_modules():
            if name in masks:
                mask = torch.tensor(masks[name], device=device).float()
                # Zero out pruned filters
                module.weight.data *= mask.view(-1, 1, 1, 1)
    
    # Evaluate after pruning (before finetuning)
    pruned_acc = evaluate(model, valloader, device)
    print(f"Accuracy after pruning (before finetune): {pruned_acc:.2f}%")
    
    # Finetune
    print("Finetuning...")
    final_acc = finetune(model, trainloader, valloader, device, args.finetune_epochs)
    print(f"Final accuracy after finetuning: {final_acc:.2f}%")
    
    # Save results
    results = {
        'method': args.method,
        'prune_ratio': args.prune_ratio,
        'baseline_acc': baseline_acc,
        'pruned_acc': pruned_acc,
        'final_acc': final_acc,
        'num_classes': args.num_classes
    }
    
    results_file = os.path.join(args.save_dir, f'imagenet_{args.method}_{args.prune_ratio}.csv')
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Save model
    model_file = os.path.join(args.save_dir, f'resnet50_{args.method}_{args.prune_ratio}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'cfg': cfg,
        'acc': final_acc
    }, model_file)
    print(f"Model saved to {model_file}")
    
    return results


if __name__ == '__main__':
    main()
