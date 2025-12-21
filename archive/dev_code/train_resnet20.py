import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from models import resnet20, resnet56

def get_dataloader(batch_size, num_workers=0, mock=False):
    if mock:
        print("Using MOCK dataset for testing pipeline...")
        # Create random tensors
        trainset = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
        testset = torch.utils.data.TensorDataset(torch.randn(50, 3, 32, 32), torch.randint(0, 10, (50,)))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, testloader

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

    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    except Exception as e:
        print(f"Failed to download CIFAR-10: {e}")
        print("Falling back to MOCK data.")
        return get_dataloader(batch_size, num_workers, mock=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def train(epoch, net, trainloader, criterion, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

def test(epoch, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc

def main():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=160, type=int, help='number of epochs')
    parser.add_argument('--depth', default=20, type=int, help='resnet depth: 20 or 56')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='save directory')
    parser.add_argument('--mock', action='store_true', help='use mock data')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.depth == 56:
        net = resnet56()
    else:
        net = resnet20()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    trainloader, testloader = get_dataloader(batch_size=128, mock=args.mock)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    best_acc = 0
    for epoch in range(args.epochs):
        train(epoch, net, trainloader, criterion, optimizer, device)
        acc = test(epoch, net, testloader, criterion, device)
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            print(f"Epoch {epoch}: New best accuracy: {best_acc:.2f}%")
            torch.save(net.state_dict(), os.path.join(args.save_dir, f'resnet{args.depth}_best.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Test Acc: {acc:.2f}%")

    print(f"Training finished. Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
