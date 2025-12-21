import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add code dir
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from models import resnet20
from train_resnet20 import get_dataloader
from utils.metric_calculator import MetricCalculator

def normalize_map(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet20()
    baseline_path = 'experiments/baseline_cifar10_resnet20.pth'
    if os.path.exists(baseline_path):
        ckpt = torch.load(baseline_path, map_location=device)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
            
    model.to(device)
    model.eval()
    
    loader, _ = get_dataloader(batch_size=128)
    
    # Calculate Scores
    layer_name = 'layer2.0.conv1'
    calc = MetricCalculator(model, loader, device)
    
    print("Computing L1 and GRA...")
    l1_scores = calc.compute_scores(layer_name, metric='l1')
    gra_scores = calc.compute_scores(layer_name, metric='gra', num_batches=10)
    
    # Normalize scores to [0,1]
    l1_norm = (l1_scores - l1_scores.min()) / (l1_scores.max() - l1_scores.min())
    gra_norm = (gra_scores - gra_scores.min()) / (gra_scores.max() - gra_scores.min())
    
    # Find interesting channels
    # High GRA, Low L1 (Top Left Quadrant)
    # Low GRA, High L1 (Bottom Right Quadrant)
    
    diff = gra_norm - l1_norm
    top_hidden_gem = np.argmax(diff) # Max (GRA - L1)
    top_false_positive = np.argmin(diff) # Max (L1 - GRA) -> Min (GRA - L1)
    
    print(f"Hidden Gem Channel (High GRA, Low L1): {top_hidden_gem}, L1={l1_norm[top_hidden_gem]:.2f}, GRA={gra_norm[top_hidden_gem]:.2f}")
    print(f"Redundant Channel (High L1, Low GRA): {top_false_positive}, L1={l1_norm[top_false_positive]:.2f}, GRA={gra_norm[top_false_positive]:.2f}")
    
    # Get Activations for a sample image
    # We need to hook again to get the spatial map (not GAP)
    activations = {}
    def hook(module, input, output):
        activations['act'] = output.detach().cpu()
        
    layer_module = calc.get_layer(layer_name)
    handle = layer_module.register_forward_hook(hook)
    
    data_iter = iter(loader)
    images, targets = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        _ = model(images)
    
    handle.remove()
    
    act = activations['act'] # [B, C, H, W]
    
    # Select a sample where the Hidden Gem is active (if possible)
    # Or just pick the first sample
    sample_idx = 0
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original Image (Inverse Normalize)
    img = images[sample_idx].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    axes[0].imshow(img)
    axes[0].set_title(f"Input Image (Class {targets[sample_idx]})")
    axes[0].axis('off')
    
    # Hidden Gem Map
    gem_map = act[sample_idx, top_hidden_gem].numpy()
    # gem_map = cv2.resize(gem_map, (32, 32))
    axes[1].imshow(gem_map, cmap='jet', extent=[0, 32, 32, 0], interpolation='bilinear')
    axes[1].set_title(f"High GRA / Low L1\n(Ch {top_hidden_gem})")
    axes[1].axis('off')
    
    # False Positive Map
    fp_map = act[sample_idx, top_false_positive].numpy()
    # fp_map = cv2.resize(fp_map, (32, 32))
    axes[2].imshow(fp_map, cmap='jet', extent=[0, 32, 32, 0], interpolation='bilinear')
    axes[2].set_title(f"High L1 / Low GRA\n(Ch {top_false_positive})")
    axes[2].axis('off')
    
    plt.tight_layout()
    out_path = 'vis/semantic_visualization.pdf'
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
