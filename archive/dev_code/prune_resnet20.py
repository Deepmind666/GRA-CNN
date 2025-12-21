import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from models import resnet20, resnet56
from train_resnet20 import get_dataloader
from prune_utils import get_model_complexity_info
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression

# -----------------------------------------------------------------------------
# Scoring Logic (Merged from MetricCalculator)
# -----------------------------------------------------------------------------

def calc_gra(A, Z, rho=0.5):
    # A: [N, C], Z: [N]
    # Normalize columns of A
    A_min = A.min(axis=0, keepdims=True)
    A_max = A.max(axis=0, keepdims=True)
    A_norm = (A - A_min) / (A_max - A_min + 1e-8)
    
    # Normalize Z
    Z_min = Z.min()
    Z_max = Z.max()
    Z_norm = (Z - Z_min) / (Z_max - Z_min + 1e-8)
    Z_norm = Z_norm[:, np.newaxis]
    
    diff = np.abs(A_norm - Z_norm)
    
    # Global max/min diff
    delta_min = diff.min()
    delta_max = diff.max()
    
    # GRA Coefficient
    xi = (delta_min + rho * delta_max) / (diff + rho * delta_max)
    
    # Average over samples -> [C]
    gamma = xi.mean(axis=0)
    return gamma

def calc_cosine(A, Z):
    # A: [N, C], Z: [N]
    A_norm = np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    Z_norm = np.linalg.norm(Z) + 1e-8
    dot = Z.dot(A) 
    scores = dot / (Z_norm * A_norm)
    return scores.flatten()

def calc_correlation(A, Z):
    # A: [N, C], Z: [N]
    A_mean = A.mean(axis=0, keepdims=True)
    A_centered = A - A_mean
    Z_mean = Z.mean()
    Z_centered = Z - Z_mean
    cov = np.dot(Z_centered, A_centered)
    A_std = np.sqrt((A_centered**2).sum(axis=0)) + 1e-8
    Z_std = np.sqrt((Z_centered**2).sum()) + 1e-8
    corr = cov / (Z_std * A_std)
    return np.abs(corr)

def calc_mi(A, Z):
    # A: [N, C], Z: [N]
    # Use small subset for MI to save time if N is large
    if A.shape[0] > 2000:
        indices = np.random.choice(A.shape[0], 2000, replace=False)
        A = A[indices]
        Z = Z[indices]
    mi = mutual_info_regression(A, Z, discrete_features=False)
    return mi

def get_scores(model, dataloader, method='gra', device='cuda', rho=0.5, num_batches=10):
    model.eval()
    scores = {}
    
    if method == 'l1':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'conv1' in name and 'layer' in name:
                    filters = module.weight.data
                    s = filters.abs().view(filters.size(0), -1).sum(dim=1)
                    scores[name] = s.cpu().numpy()
        return scores

    # For activation-based metrics
    print(f"Collecting activations for {method} (batches={num_batches})...")
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # GAP
            if output.dim() == 4:
                act = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
            else:
                act = output.view(output.size(0), -1)
            
            if name not in activations:
                activations[name] = []
            activations[name].append(act.detach().cpu())
        return hook
        
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if 'conv1' in name and 'layer' in name:
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    logits_gt = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            
            # GT Logit
            gt_logits = out.gather(1, targets.view(-1, 1)).squeeze()
            logits_gt.append(gt_logits.detach().cpu())
            
    for h in hooks:
        h.remove()
        
    # Concatenate
    Z = torch.cat(logits_gt, dim=0).numpy()
    
    for name, acts_list in activations.items():
        A = torch.cat(acts_list, dim=0).numpy()
        
        if method == 'gra':
            s = calc_gra(A, Z, rho)
        elif method == 'cosine':
            s = calc_cosine(A, Z)
        elif method == 'correlation':
            s = calc_correlation(A, Z)
        elif method == 'mi':
            s = calc_mi(A, Z)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        scores[name] = s
        
    return scores

# -----------------------------------------------------------------------------
# Pruning Logic
# -----------------------------------------------------------------------------

def prune_model(model, ratio, method='gra', dataloader=None, device='cuda', rho=0.5):
    print(f"Calculating scores using {method}...")
    scores = get_scores(model, dataloader, method, device, rho, num_batches=10)
    
    # Flatten all scores to find global threshold
    all_scores = np.concatenate([s for s in scores.values()])
    threshold = np.percentile(all_scores, ratio * 100)
    print(f"Pruning threshold: {threshold:.4f}")
    
    new_cfg = []
    mask_dict = {} 
    
    pruned_channels = 0
    total_channels = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
             if 'conv1' in name and 'layer' in name:
                 s = scores[name]
                 mask = s >= threshold
                 
                 # Ensure at least one channel is kept
                 if mask.sum() == 0:
                     idx = np.argmax(s)
                     mask[idx] = True
                     
                 mask_dict[name] = mask
                 new_cfg.append(int(mask.sum()))
                 
                 pruned_channels += len(mask) - mask.sum()
                 total_channels += len(mask)
                 
    print(f"Pruned {pruned_channels}/{total_channels} channels. Real Ratio: {pruned_channels/total_channels:.2%}")
    return new_cfg, mask_dict

def build_new_model(old_model, new_cfg, mask_dict, depth):
    if depth == 56:
        new_model = resnet56(cfg=new_cfg)
    elif depth == 110:
        from models.resnet_cifar import resnet110
        new_model = resnet110(cfg=new_cfg)
    else:
        new_model = resnet20(cfg=new_cfg)
        
    old_modules = list(old_model.named_modules())
    new_modules = dict(new_model.named_modules())
    
    last_mask = None
    
    for name, old_module in old_modules:
        if isinstance(old_module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            new_module = new_modules[name]
            
            if 'conv1' in name and 'layer' in name:
                mask = mask_dict[name]
                mask_indices = np.where(mask)[0]
                
                if isinstance(old_module, nn.Conv2d):
                    new_module.weight.data = old_module.weight.data[mask_indices, :, :, :]
                    last_mask = mask
                    
            elif 'bn1' in name and 'layer' in name:
                mask = last_mask
                mask_indices = np.where(mask)[0]
                new_module.weight.data = old_module.weight.data[mask_indices]
                new_module.bias.data = old_module.bias.data[mask_indices]
                new_module.running_mean.data = old_module.running_mean.data[mask_indices]
                new_module.running_var.data = old_module.running_var.data[mask_indices]
                
            elif 'conv2' in name and 'layer' in name:
                mask = last_mask
                mask_indices = np.where(mask)[0]
                if isinstance(old_module, nn.Conv2d):
                    new_module.weight.data = old_module.weight.data[:, mask_indices, :, :]
                    
            else:
                new_module.weight.data = old_module.weight.data.clone()
                if old_module.bias is not None:
                    new_module.bias.data = old_module.bias.data.clone()
                if isinstance(old_module, nn.BatchNorm2d):
                    new_module.running_mean.data = old_module.running_mean.data.clone()
                    new_module.running_var.data = old_module.running_var.data.clone()

    return new_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=20)
    parser.add_argument('--resume', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--save', type=str, default='./pruned_model.pth')
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--method', type=str, default='gra', choices=['gra', 'l1', 'cosine', 'correlation', 'mi'])
    parser.add_argument('--rho', type=float, default=0.5, help='GRA resolution coefficient')
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.depth == 56:
        model = resnet56()
    elif args.depth == 110:
        from models.resnet_cifar import resnet110
        model = resnet110()
    else:
        model = resnet20()
        
    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    trainloader, testloader = get_dataloader(batch_size=128, mock=args.mock)
    
    flops_old, params_old = get_model_complexity_info(model, (3, 32, 32))
    print(f"Original: FLOPs: {flops_old/1e6:.2f}M, Params: {params_old/1e6:.2f}M")
    
    new_cfg, mask_dict = prune_model(model, args.ratio, args.method, trainloader, device, rho=args.rho)
    
    new_model = build_new_model(model, new_cfg, mask_dict, args.depth)
    new_model.to(device)
    
    flops_new, params_new = get_model_complexity_info(new_model, (3, 32, 32))
    print(f"Pruned:   FLOPs: {flops_new/1e6:.2f}M, Params: {params_new/1e6:.2f}M")
    print(f"Reduction: FLOPs: {(1-flops_new/flops_old):.2%}, Params: {(1-params_new/params_old):.2%}")
    
    torch.save({'state_dict': new_model.state_dict(), 'cfg': new_cfg}, args.save)
    print(f"Saved pruned model to {args.save}")

if __name__ == '__main__':
    main()
