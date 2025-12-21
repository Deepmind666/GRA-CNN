import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import csv
from models import resnet20, resnet56
from train_resnet20 import get_dataloader, train, test

def main():
    parser = argparse.ArgumentParser(description='Finetune Pruned ResNet')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate') # Lower LR for finetuning
    parser.add_argument('--epochs', default=40, type=int, help='number of epochs') # Fewer epochs
    parser.add_argument('--depth', default=20, type=int, help='resnet depth: 20 or 56')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to pruned checkpoint')
    parser.add_argument('--save-dir', default='./checkpoints_finetuned', type=str)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
        cfg = checkpoint['cfg']
        state_dict = checkpoint['state_dict']
    else:
        print("Error: Checkpoint must contain 'cfg' and 'state_dict'")
        return

    if args.depth == 56:
        net = resnet56(cfg=cfg)
    else:
        net = resnet20(cfg=cfg)
        
    net.load_state_dict(state_dict)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)

    trainloader, testloader = get_dataloader(batch_size=128, mock=args.mock)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Setup log file
    log_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TestAccuracy'])

    best_acc = 0
    for epoch in range(args.epochs):
        train(epoch, net, trainloader, criterion, optimizer, device)
        acc = test(epoch, net, testloader, criterion, device)
        scheduler.step()
        
        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, acc])
        
        if acc > best_acc:
            best_acc = acc
            print(f"Epoch {epoch}: New best accuracy: {best_acc:.2f}%")
            torch.save({'state_dict': net.state_dict(), 'cfg': cfg, 'acc': acc}, 
                       os.path.join(args.save_dir, 'best.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Test Acc: {acc:.2f}%")

    print(f"Finetuning finished. Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
