import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def get_cifar100_dataloader(batch_size=128, num_workers=2, mock=False, data_root='./data'):
    if mock:
        # Mock data: 100 samples, 3x32x32
        train_x = torch.randn(100, 3, 32, 32)
        train_y = torch.randint(0, 100, (100,))
        test_x = torch.randn(50, 3, 32, 32)
        test_y = torch.randint(0, 100, (50,))
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    try:
        trainset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    except:
        print("Warning: CIFAR-100 download failed or not found. Using Mock Data.")
        return get_cifar100_dataloader(batch_size, num_workers, mock=True)
        
    return train_loader, test_loader
