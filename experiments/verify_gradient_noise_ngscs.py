import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import numpy as np
from models.resnet_cifar import resnet20, resnet56

# ============================================================================
# Normalized Gradient-Semantic Consistency Score (NGSCS)
# Formula: Weighted Cosine Consistency with Energy Gating
# ============================================================================

@torch.no_grad()
def _flatten(t):
    return t.reshape(t.size(0), -1)

def compute_ngscs_for_layer(model, dataloader, target_layer_name, device, w_max=3.0, eps=1e-8):
    """
    Computes the Normalized GSCS for a specific layer.
    """
    model.eval()
    acts = {"x": None}
    
    # Hook to capture activations
    def fwd_hook(m, inp, out):
        acts["x"] = out
        out.retain_grad() # Enable gradient on non-leaf tensor

    # Register hook
    handle = None
    for n, m in model.named_modules():
        if n == target_layer_name:
            handle = m.register_forward_hook(fwd_hook)
            break
            
    if handle is None:
        print(f"Error: Layer {target_layer_name} not found.")
        return 0.0

    print(f"Scanning {target_layer_name} for Gradient Noise...")
    g_by_class = defaultdict(list)
    norm_by_class = defaultdict(list)
    
    # Collect gradients
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx > 5: break # Limit samples for speed (approx 500-600 images)
        images, labels = images.to(device), labels.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        # Backward to get dL/dA
        loss.backward()
        
        # Retrieve gradient at the hooked layer
        if acts["x"].grad is None:
            continue
            
        g = acts["x"].grad.detach()              # [B, C, H, W]
        g = _flatten(g)                          # [B, D]
        g_norm = g.norm(dim=1)                   # [B]

        # Group by class
        for i in range(g.size(0)):
            c = int(labels[i].item())
            g_by_class[c].append(g[i].cpu())
            norm_by_class[c].append(g_norm[i].cpu())

    handle.remove()

    # Compute NGSCS per class
    scores = []
    
    for c, gs in g_by_class.items():
        if len(gs) < 2: continue
        
        G = torch.stack(gs, dim=0)               # [Nc, D]
        norms = torch.stack(norm_by_class[c])    # [Nc]

        # 1. Energy Gating (Weighting)
        # Avoid zero gradients dominating or random noise
        med = norms.median()
        w = (norms / (med + eps)).clamp(0, w_max) # [Nc]

        # 2. Whitening / Normalization
        # Direction unit vectors
        G_hat = G / (G.norm(dim=1, keepdim=True) + eps) 

        # 3. Weighted Cosine Consistency
        # Mean Vector (Weighted)
        v = (w.unsqueeze(1) * G_hat).sum(dim=0) # [D]
        
        # NGSCS = Norm(Weighted_Sum) / (Sum_Weights)
        score_c = v.norm() / (w.sum() + eps)
        scores.append(score_c)

    if len(scores) == 0: return 0.0
    
    final_ngscs = torch.stack(scores).mean().item()
    return final_ngscs

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Model (ResNet-56)
    model = resnet56(num_classes=10).to(device)
    # Note: In real usage, load trained weights! 
    # model.load_state_dict(torch.load('checkpoints/resnet56_cifar10.pth.tar')['state_dict'])
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
    
    # Target Layers (One from each stage)
    targets = [
        'layer1.0.conv1', # Stage 1 (Shallow)
        'layer1.4.conv1',
        'layer2.0.conv1', # Stage 2
        'layer2.4.conv1',
        'layer3.0.conv1', # Stage 3
        'layer3.4.conv1'  # Stage 3 (Deep)
    ]
    
    print("\n=== Normalized GSCS Analysis (Theoretical Verification) ===")
    print("Hypothesis: NGSCS should increase with depth (Semantic Consistency).")
    
    results = {}
    for layer in targets:
        score = compute_ngscs_for_layer(model, loader, layer, device)
        results[layer] = score
        print(f"Layer {layer}: NGSCS = {score:.4f}")
        
    # Save for plotting
    import json
    with open('experiments/ngscs_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
