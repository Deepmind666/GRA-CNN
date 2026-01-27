"""
GRA-CNN v5.3 8-Hour Comprehensive Experiment Suite
===================================================
Validates v5.3 fixes:
1. BN freezing with model.eval()
2. ResNet FLOPs with correct INPUT sizes
3. try/finally protection for hooks

Run time: ~8 hours on RTX 5090
"""

import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# RTX 5090 compatibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

sys.path.insert(0, r'C:\GRA-CNN')

import torchvision
import torchvision.transforms as transforms

from models.resnet_cifar import resnet20, resnet56, resnet110
from models.vgg_cifar import vgg16

# Import v5.3 algorithm
from pruning.core_algorithm_v5 import (
    GRA_VERSION,
    compute_gra_final_score_v5,
    get_global_mask_iso_flops_v5,
    get_pruning_mask_simple,
)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r'C:\GRA-CNN\data'
EXPERIMENTS_DIR = r'C:\GRA-CNN\experiments'
RESULTS_FILE = os.path.join(EXPERIMENTS_DIR, f'v53_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

print(f"=" * 60)
print(f"GRA-CNN v{GRA_VERSION} 8-Hour Experiment Suite")
print(f"=" * 60)
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Results: {RESULTS_FILE}")
print(f"=" * 60)


def get_dataloaders(dataset, batch_size=32):
    """Get data loaders for CIFAR-10/100."""
    dataset_lower = dataset.lower().replace('-', '')

    if dataset_lower == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        DatasetClass = torchvision.datasets.CIFAR10
    elif dataset_lower == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        DatasetClass = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = DatasetClass(root=DATA_DIR, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = DatasetClass(root=DATA_DIR, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader, num_classes


def get_model(arch, num_classes):
    """Get model by architecture name."""
    arch = arch.lower().replace('-', '')
    if arch == 'resnet20': return resnet20(num_classes=num_classes)
    elif arch == 'resnet56': return resnet56(num_classes=num_classes)
    elif arch == 'resnet110': return resnet110(num_classes=num_classes)
    elif arch == 'vgg16': return vgg16(num_classes=num_classes)
    else: raise ValueError(f"Unknown architecture: {arch}")


def load_baseline(model, arch, dataset, device):
    """Load pretrained baseline weights."""
    arch_key = arch.lower().replace('-', '')
    dataset_key = dataset.lower().replace('-', '')
    ckpt_path = os.path.join(EXPERIMENTS_DIR, f'baseline_{dataset_key}_{arch_key}.pth')

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"  Loaded baseline: {ckpt_path}")
            return True
        except Exception as e:
            print(f"  Error loading baseline: {e}")
    return False


def evaluate(model, dataloader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def finetune(model, train_loader, test_loader, epochs, device, lr=0.01):
    """Finetune pruned model."""
    print(f"  [Finetune] Epochs={epochs}, LR={lr}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs}: Acc={acc:.2f}%, Best={best_acc:.2f}%")

    return best_acc


def run_v53_experiment(arch, dataset, ratio, use_iso_flops=True,
                       use_hessian=True, use_ngscs=True, epochs=40):
    """Run single v5.3 experiment."""
    print(f"\n{'='*60}")
    print(f"[v{GRA_VERSION}] {arch}/{dataset}/ratio={ratio}")
    print(f"Config: Iso-FLOPs={use_iso_flops}, Hessian={use_hessian}, NGSCS={use_ngscs}")
    print(f"{'='*60}")

    device = DEVICE

    # Load data
    train_loader, test_loader, num_classes = get_dataloaders(dataset)
    print(f"Data: {dataset}, Classes={num_classes}")

    # Load model
    model = get_model(arch, num_classes).to(device)
    if not load_baseline(model, arch, dataset, device):
        print("  SKIP: No baseline checkpoint")
        return None

    base_acc = evaluate(model, test_loader, device)
    print(f"Baseline Acc: {base_acc:.2f}%")

    # Compute v5.3 scores
    print(f"Computing v{GRA_VERSION} scores...")
    scores, metadata = compute_gra_final_score_v5(
        model, train_loader, device,
        num_batches=10,
        use_hessian=use_hessian,
        use_ngscs=use_ngscs,
        verbose=True
    )

    # Apply pruning
    if use_iso_flops:
        target_flops_ratio = 1.0 - ratio
        masks, actual_ratio = get_global_mask_iso_flops_v5(
            model, scores, target_flops_ratio,
            input_size=32, verbose=True
        )
    else:
        masks = get_pruning_mask_simple(scores, ratio, model)

    # Apply masks
    for name, module in model.named_modules():
        if name in masks and isinstance(module, nn.Conv2d):
            mask = torch.from_numpy(masks[name]).float().to(device)
            with torch.no_grad():
                module.weight.data *= mask.view(-1, 1, 1, 1)
                if module.bias is not None:
                    module.bias.data *= mask

    pruned_acc = evaluate(model, test_loader, device)
    print(f"Pruned Acc: {pruned_acc:.2f}%")

    # Finetune
    final_acc = finetune(model, train_loader, test_loader, epochs, device)
    print(f"Final Acc: {final_acc:.2f}%")

    return {
        'version': GRA_VERSION,
        'architecture': arch,
        'dataset': dataset,
        'ratio': ratio,
        'iso_flops': use_iso_flops,
        'use_hessian': use_hessian,
        'use_ngscs': use_ngscs,
        'baseline_acc': base_acc,
        'pruned_acc': pruned_acc,
        'final_acc': final_acc,
        'acc_drop': base_acc - final_acc,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def save_result(result):
    """Save single result to CSV."""
    df = pd.DataFrame([result])
    if os.path.exists(RESULTS_FILE):
        existing = pd.read_csv(RESULTS_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"  Saved to {RESULTS_FILE}")


def main():
    """Run 8-hour experiment suite."""
    print(f"\nStarting 8-hour v{GRA_VERSION} experiment suite...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Experiment configurations
    experiments = [
        # ResNet-20 experiments
        ('resnet20', 'cifar10', 0.3),
        ('resnet20', 'cifar10', 0.5),
        ('resnet20', 'cifar10', 0.7),

        # ResNet-56 experiments
        ('resnet56', 'cifar10', 0.3),
        ('resnet56', 'cifar10', 0.5),
        ('resnet56', 'cifar10', 0.7),

        # VGG-16 experiments
        ('vgg16', 'cifar10', 0.3),
        ('vgg16', 'cifar10', 0.5),
        ('vgg16', 'cifar10', 0.7),

        # CIFAR-100 experiments
        ('resnet56', 'cifar100', 0.3),
        ('resnet56', 'cifar100', 0.5),
        ('vgg16', 'cifar100', 0.5),
    ]

    total = len(experiments)
    completed = 0
    failed = 0

    for i, (arch, dataset, ratio) in enumerate(experiments):
        print(f"\n[{i+1}/{total}] Running experiment...")
        try:
            result = run_v53_experiment(
                arch, dataset, ratio,
                use_iso_flops=True,
                use_hessian=True,
                use_ngscs=True,
                epochs=40
            )
            if result:
                save_result(result)
                completed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Experiment Suite Complete!")
    print(f"Completed: {completed}/{total}")
    print(f"Failed: {failed}/{total}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {RESULTS_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
