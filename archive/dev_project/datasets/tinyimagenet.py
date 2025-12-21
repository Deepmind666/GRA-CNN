import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

def get_tinyimagenet_dataloader(batch_size=64, num_workers=2, mock=False, data_root='./data/tiny-imagenet-200'):
    if mock or not os.path.exists(data_root):
        if not mock:
            print(f"Tiny-ImageNet not found at {data_root}. Using Mock Data.")
        # Mock data: 100 samples, 3x64x64, 200 classes
        train_x = torch.randn(100, 3, 64, 64)
        train_y = torch.randint(0, 200, (100,))
        test_x = torch.randn(50, 3, 64, 64)
        test_y = torch.randint(0, 200, (50,))
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    # Tiny-ImageNet: 64x64 images
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    
    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    testset = datasets.ImageFolder(val_dir, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
