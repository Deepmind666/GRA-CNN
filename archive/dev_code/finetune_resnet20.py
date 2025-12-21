import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import pandas as pd
from models import resnet20, resnet56

def get_dataloader(batch_size, num_workers=2, mock=False):
    if mock:
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

    # Try default paths
    roots = ['./data', '../data', 'data']
    root = './data'
    for r in roots:
        if os.path.exists(r):
            root = r
            break
            
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

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
    return train_loss / (batch_idx + 1), 100. * correct / total

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
    return test_loss / (batch_idx + 1), acc

def main():
    parser = argparse.ArgumentParser(description='Finetune Pruned ResNet on CIFAR-10')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate') # Lower LR for finetuning
    parser.add_argument('--epochs', default=40, type=int, help='number of epochs') # Fewer epochs
    parser.add_argument('--depth', default=20, type=int, help='resnet depth: 20 or 56')
    parser.add_argument('--pruned-model', type=str, required=True, help='path to pruned model checkpoint')
    parser.add_argument('--save-dir', default='./finetuned', type=str, help='save directory')
    parser.add_argument('--mock', action='store_true', help='use mock data')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint to get cfg
    print(f"Loading pruned model from {args.pruned_model}")
    checkpoint = torch.load(args.pruned_model, map_location=device)
    
    cfg = None
    if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
        print(f"Found configuration: {cfg}")
    
    if args.depth == 56:
        net = resnet56(cfg=cfg)
    elif args.depth == 110:
        from models.resnet_cifar import resnet110
        net = resnet110(cfg=cfg)
    else:
        net = resnet20(cfg=cfg)
        
    # Load weights
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
    net.load_state_dict(state_dict)
    net.to(device)

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)

    trainloader, testloader = get_dataloader(batch_size=128, mock=args.mock)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    log_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train(epoch, net, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(epoch, net, testloader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{test_loss:.4f},{test_acc:.2f}\n")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            
    print(f"Finetuning finished. Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
