"""
MobileNetV2 for CIFAR/Tiny-ImageNet
===================================
Adapted MobileNetV2 for 32x32 and 64x64 input images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    """MobileNetV2 Inverted Residual Block"""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Pointwise
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    """
    MobileNetV2 adapted for CIFAR (32x32) and Tiny-ImageNet (64x64).
    
    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier (0.5, 0.75, 1.0, 1.4)
        input_size: 32 for CIFAR, 64 for Tiny-ImageNet
    """
    
    def __init__(self, num_classes=10, width_mult=1.0, input_size=32):
        super().__init__()
        
        # MobileNetV2 configuration: [expand_ratio, channels, num_blocks, stride]
        if input_size == 32:  # CIFAR
            # Smaller strides for 32x32
            cfg = [
                [1, 16, 1, 1],
                [6, 24, 2, 1],  # stride 1 instead of 2
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            first_stride = 1
        else:  # Tiny-ImageNet (64x64)
            cfg = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            first_stride = 2
        
        # First layer
        input_channels = int(32 * width_mult)
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channels, 3, first_stride, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU6(inplace=True)
        )]
        
        # Inverted residual blocks
        for t, c, n, s in cfg:
            output_channels = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channels, output_channels, stride, t))
                input_channels = output_channels
        
        # Last layer
        last_channels = int(1280 * width_mult)
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(last_channels, num_classes)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def mobilenetv2_cifar(num_classes=10, width_mult=1.0):
    """MobileNetV2 for CIFAR-10/100 (32x32)"""
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult, input_size=32)


def mobilenetv2_tiny(num_classes=200, width_mult=1.0):
    """MobileNetV2 for Tiny-ImageNet (64x64)"""
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult, input_size=64)


if __name__ == '__main__':
    # Test
    model = mobilenetv2_cifar(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"CIFAR input: {x.shape}, output: {y.shape}")
    
    model = mobilenetv2_tiny(num_classes=200)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    print(f"Tiny-ImageNet input: {x.shape}, output: {y.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
