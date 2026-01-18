"""
基线模型训练器
==============
训练预训练模型作为剪枝实验的起点
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import time
from datetime import datetime

sys.path.insert(0, r'C:\GRA-CNN')

from models.resnet_cifar import resnet20, resnet56, resnet110
from models.vgg_cifar import vgg16

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\GRA-CNN\data'
EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'

def get_dataloader(dataset, batch_size=128, train=True):
    if dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if train:
            ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
        else:
            ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    elif dataset.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        if train:
            ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform_train)
        else:
            ds = torchvision.datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True)

def get_model(arch, dataset):
    num_classes = 100 if dataset.lower() == 'cifar100' else 10
    arch_lower = arch.lower().replace('-', '')
    
    if arch_lower == 'resnet20':
        model = resnet20(num_classes=num_classes)
    elif arch_lower == 'resnet32':
        from models.resnet_cifar import resnet32
        return resnet32(num_classes=num_classes)
    elif arch_lower == 'resnet44':
        from models.resnet_cifar import resnet44
        return resnet44(num_classes=num_classes)
    elif arch_lower == 'resnet56':
        model = resnet56(num_classes=num_classes)
    elif arch_lower == 'resnet110':
        model = resnet110(num_classes=num_classes)
    elif arch_lower == 'vgg16':
        model = vgg16(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model

def train_baseline(arch, dataset, epochs=200, lr=0.1):
    print(f"Training baseline: {arch} on {dataset}")
    print(f"Device: {DEVICE}")
    
    arch_key = arch.lower().replace('-', '')
    dataset_key = dataset.lower().replace('-', '')
    
    # Data
    train_loader = get_dataloader(dataset, train=True)
    test_loader = get_dataloader(dataset, train=False)
    
    # Model
    model = get_model(arch, dataset).to(DEVICE)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    best_acc = 0
    history = []
    
    for epoch in range(epochs):
        # ... (rest of the training loop) ...
        # [OMITTED FOR BREVITY - keep existing training logic]
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(EXPERIMENTS_DIR, f'baseline_{dataset_key}_{arch_key}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'acc': test_acc,
            }, save_path)
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, Best={best_acc:.2f}%")
    
    # Save history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(EXPERIMENTS_DIR, f'baseline_{dataset}_{arch}.csv')
    history_df.to_csv(history_path, index=False)
    
    print(f"\nBaseline training complete!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.join(EXPERIMENTS_DIR, f'baseline_{dataset}_{arch}.pth')}")
    
    return best_acc

def main():
    parser = argparse.ArgumentParser(description='Train Baseline Model')
    parser.add_argument('--arch', type=str, required=True, help='Architecture')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    
    args = parser.parse_args()
    
    train_baseline(args.arch, args.dataset, args.epochs)

if __name__ == '__main__':
    main()
