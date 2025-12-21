
import argparse
import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import resnet20, resnet56
from gra import GrayRelationalChannelScorer

def get_args():
    parser = argparse.ArgumentParser(description='GRA-CNN Pruning on CIFAR-10')
    parser.add_argument('--arch', default='resnet20', type=str, help='model architecture')
    parser.add_argument('--epochs', default=160, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--prune-ratio', default=0.3, type=float, help='pruning ratio')
    parser.add_argument('--score-type', default='gra', type=str, choices=['l1', 'gra'], help='scoring method')
    parser.add_argument('--save-dir', default='checkpoints', type=str, help='save directory')
    parser.add_argument('--gpu', action='store_true', default=False, help='use GPU')
    parser.add_argument('--mock', action='store_true', help='use mock data')
    parser.add_argument('--rho', default=0.5, type=float, help='GRA coefficient')
    parser.add_argument('--csv-file', default='results.csv', type=str, help='output csv file')
    return parser.parse_args()

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = running_loss/len(train_loader)
    print(f'Epoch [{epoch+1}] Train Loss: {avg_loss:.4f} Acc: {acc:.2f}%')
    return avg_loss, acc

def validate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = running_loss/len(val_loader)
    print(f'Val Loss: {avg_loss:.4f} Acc: {acc:.2f}%')
    return avg_loss, acc

def prune_model(model, ratio, score_type, train_loader, device, rho=0.5):
    print(f"Pruning model with {score_type} score, ratio={ratio}, rho={rho}...")
    
    # Gather all prunable modules (Conv2d followed by BN)
    # In ResNet, we usually prune the first conv of the basic block?
    # Or prune based on BN scales?
    # GRA requires feature maps, so we need hooks.
    
    gra_scorer = GrayRelationalChannelScorer(rho=rho)
    
    # Dictionary to store scores: layer_name -> tensor of scores
    scores = {}
    hooks = []

    def get_activation_hook(name):
        def hook(model, input, output):
            # output is feature map [N, C, H, W]
            # We need logits for GRA.
            # This is tricky: we need the final logits of the model for the SAME batch.
            # So we probably need to run a forward pass and capture everything.
            pass 
        return hook

    # Simple approach: Iterate over layers, run a forward pass to get features, compute score.
    
    model.eval()
    
    # We only prune the first conv of each block for simplicity in this demo
    # Or prune the BN weights directly if using L1.
    
    prunable_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # Check if it's a prunable layer (e.g. not the first or last if needed)
            prunable_layers.append((name, m))
    
    if score_type == 'l1':
        # L1 norm of BN weights (gamma)
        for name, m in prunable_layers:
            score = m.weight.data.abs()
            scores[name] = score
            
    elif score_type == 'gra':
        # Need to run forward pass
        # We accumulate GRA scores over a few batches
        
        # Register hooks to capture features
        features = {}
        def get_hook(name):
            def hook(module, input, output):
                features[name] = output
            return hook
            
        for name, m in prunable_layers:
            hooks.append(m.register_forward_hook(get_hook(name)))
            scores[name] = torch.zeros(m.num_features).to(device)
            
        # Run one epoch (or less) of calibration
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(train_loader):
                if i > 10: break # Only use 10 batches for speed
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) # Logits
                
                for name, m in prunable_layers:
                    f_map = features[name] # [N, C, H, W]
                    batch_score = gra_scorer.compute_score(f_map, targets, outputs)
                    scores[name] += batch_score
                    
        # Average scores
        for name in scores:
            scores[name] /= 10.0 # divided by num batches
            
        # Remove hooks
        for h in hooks:
            h.remove()

    # Now apply pruning (Global or Local?)
    # Let's do Global pruning for simplicity: Sort ALL channels in the network and kill bottom X%.
    
    all_scores = []
    for name, score in scores.items():
        all_scores.append(score)
    
    all_scores = torch.cat(all_scores)
    threshold_index = int(len(all_scores) * ratio)
    threshold_val, _ = torch.sort(all_scores)
    threshold = threshold_val[threshold_index]
    
    print(f"Global Pruning Threshold: {threshold:.4f}")
    
    # Apply masks
    pruned_params = 0
    total_params = 0
    
    for name, m in prunable_layers:
        score = scores[name]
        mask = score.gt(threshold).float()
        
        # Apply mask to BN weight and bias
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        
        # Also need to mask the preceding Conv layer weights!
        # This requires knowing the graph. 
        # For this demo, we just zero out BN parameters. 
        # In a real "Pruning" paper, we would physically remove channels.
        # Zeroing BN effectively kills the channel output.
        
        pruned = mask.eq(0).sum().item()
        total = mask.numel()
        pruned_params += pruned
        total_params += total
        
        # print(f"Layer {name}: Pruned {pruned}/{total} ({pruned/total:.2%})")

    print(f"Total Pruned Channels: {pruned_params}/{total_params} ({pruned_params/total_params:.2%})")
    return model

def main():
    args = get_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    # Data
    print('==> Preparing data..')
    if args.mock:
        print("Using MOCK data (random noise).")
        trainset = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
        testset = torch.utils.data.TensorDataset(torch.randn(20, 3, 32, 32), torch.randint(0, 10, (20,)))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False)
    else:
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
            trainset = torchvision.datasets.CIFAR10(root='../project/data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='../project/data', train=False, download=True, transform=transform_test)
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Falling back to mock data.")
            trainset = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
            testset = torch.utils.data.TensorDataset(torch.randn(20, 3, 32, 32), torch.randint(0, 10, (20,)))
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print(f'==> Building model {args.arch}..')
    if args.arch == 'resnet20':
        model = resnet20()
    elif args.arch == 'resnet56':
        model = resnet56()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Log file for convergence
    log_file = os.path.join(args.save_dir, f'log_{args.arch}_{args.score_type}_{args.prune_ratio}_{args.rho}.csv')
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')

    # 1. Pre-training (or load checkpoint)
    # For demo, we train for a few epochs
    print("==> Starting Training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, device)
        test_loss, test_acc = validate(testloader, model, criterion, device)
        
        with open(log_file, 'a') as f:
            f.write(f'{epoch},{train_loss},{train_acc},{test_loss},{test_acc}\n')
        
        # Prune halfway? Or after full training?
        # Usually: Train Full -> Prune -> Finetune
        # Let's just prune at the end for this script.
        
    # 2. Pruning
    model = prune_model(model, args.prune_ratio, args.score_type, trainloader, device, rho=args.rho)
    
    # 3. Fine-tuning
    print("==> Fine-tuning...")
    optimizer = optim.SGD(model.parameters(), lr=args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(5): # Short finetune
        train(trainloader, model, criterion, optimizer, epoch, device)
        acc = validate(testloader, model, criterion, device)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.arch}_{args.score_type}_pruned.pth'))
    
    # Write to CSV
    with open(args.csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.arch, args.score_type, args.prune_ratio, args.rho, acc])

if __name__ == '__main__':
    main()
