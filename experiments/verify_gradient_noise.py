import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models.resnet_cifar import resnet20, resnet56

def compute_gscs(model, dataloader, device, target_layers):
    """
    Computes Gradient-Semantic Consistency Score (GSCS).
    GSCS = ||Mean(Grad)|| / Mean(||Grad||)
    """
    model.eval()
    gradients = {name: [] for name in target_layers}
    
    def get_grad_hook(name):
        def hook(m, grad_in, grad_out):
            gradients[name].append(grad_out[0].detach().cpu())
        return hook

    hooks = []
    for name, m in model.named_modules():
        if name in target_layers:
            # Full Backward Hook gets gradient w.r.t output
            hooks.append(m.register_full_backward_hook(get_grad_hook(name)))
            
    print("Computing GSCS on single-class batch...")
    
    # 1. Filter for a single class (e.g., Class 0)
    target_class = 0
    consistent_batch = []
    for inputs, targets in dataloader:
        mask = targets == target_class
        if mask.sum() > 0:
            consistent_batch.append(inputs[mask])
        if len(consistent_batch) * 128 > 100: break # Collect ~100 images
        
    inputs = torch.cat(consistent_batch, dim=0)[:50].to(device) # Use 50 images
    targets = torch.full((50,), target_class, device=device)
    
    # 2. Per-sample gradient computation (inefficient loop but safe)
    # Ideally use functorch, but loop is fine for 50 samples
    for i in range(len(inputs)):
        model.zero_grad()
        out = model(inputs[i:i+1])
        loss = F.cross_entropy(out, targets[i:i+1])
        loss.backward()
        
    # 3. Calculate GSCS
    results = {}
    for name in target_layers:
        grads = torch.stack(gradients[name]) # [N, C, H, W]
        grads_flat = grads.view(grads.shape[0], -1)
        
        # Numerator: Norm of Mean Gradient
        mean_grad = grads_flat.mean(dim=0)
        norm_of_mean = mean_grad.norm()
        
        # Denominator: Mean of Gradient Norms
        mean_of_norms = grads_flat.norm(dim=1).mean()
        
        gscs = norm_of_mean / (mean_of_norms + 1e-8)
        results[name] = gscs.item()
        
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load a pretrained model (dummy init for check or load path)
    model = resnet56(num_classes=10).to(device)
    # Ideally load weights: model.load_state_dict(...)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
    
    # Target 3 stages
    targets = ['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1']
    
    scores = compute_gscs(model, loader, device, targets)
    
    print("\n=== Gradient-Semantic Consistency Score (GSCS) ===")
    print("Hypothesis: Shallow < Deep")
    for name in targets:
        print(f"{name}: {scores[name]:.4f}")

if __name__ == '__main__':
    main()
