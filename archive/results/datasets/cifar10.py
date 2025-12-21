import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar10_dataloader(data_root='./data', batch_size=128, num_workers=2, mock=False):
    if mock:
        # Create mock data
        train_data = torch.randn(100, 3, 32, 32)
        train_targets = torch.randint(0, 10, (100,))
        test_data = torch.randn(20, 3, 32, 32)
        test_targets = torch.randint(0, 10, (20,))
        
        train_set = torch.utils.data.TensorDataset(train_data, train_targets)
        test_set = torch.utils.data.TensorDataset(test_data, test_targets)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
        return train_loader, test_loader

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)

    return train_loader, test_loader
