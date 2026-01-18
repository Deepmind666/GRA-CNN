"""
科研严谨性修复工具: FLOPs 和吞吐量真实测量
============================================
使用 thop 库计算模型的真实 FLOPs 和参数量
"""

import torch
import torch.nn as nn
import time
import sys
import os
import json

sys.path.insert(0, r'C:\GRA-CNN')

# 尝试导入 thop，如果没有则给出提示
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Run: pip install thop")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_flops_params(model, input_size=(1, 3, 64, 64)):
    """
    计算模型的 FLOPs 和参数量
    
    Args:
        model: PyTorch 模型
        input_size: 输入张量大小 (batch, channels, height, width)
    
    Returns:
        dict: {'flops': GFLOPs, 'params': M params, 'flops_raw': raw flops, 'params_raw': raw params}
    """
    if not THOP_AVAILABLE:
        # 手动计算参数量
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'flops': 'N/A (thop not installed)',
            'params': f'{params/1e6:.2f}M',
            'params_raw': params,
            'flops_raw': None
        }
    
    model = model.to(DEVICE)
    model.eval()
    
    dummy_input = torch.randn(*input_size).to(DEVICE)
    
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    flops_str, params_str = clever_format([flops, params], "%.2f")
    
    return {
        'flops': flops_str,
        'params': params_str,
        'flops_gflops': flops / 1e9,
        'params_m': params / 1e6,
        'flops_raw': flops,
        'params_raw': params
    }


def measure_throughput(model, input_size=(1, 3, 64, 64), warmup=50, iterations=200):
    """
    测量模型的 GPU 推理吞吐量
    
    Args:
        model: PyTorch 模型
        input_size: 输入张量大小
        warmup: 预热迭代次数
        iterations: 测量迭代次数
    
    Returns:
        dict: {'throughput_fps': images/sec, 'latency_ms': ms per image}
    """
    model = model.to(DEVICE)
    model.eval()
    
    batch_size = input_size[0]
    dummy_input = torch.randn(*input_size).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    total_time = end - start
    latency_per_batch = total_time / iterations
    latency_per_image = latency_per_batch / batch_size
    throughput = batch_size * iterations / total_time
    
    return {
        'throughput_fps': throughput,
        'latency_ms': latency_per_image * 1000,
        'batch_size': batch_size,
        'iterations': iterations,
        'total_time_s': total_time
    }


def measure_model_comprehensive(model, dataset='CIFAR-10', batch_size=128):
    """
    综合测量模型的所有效率指标
    """
    # 确定输入大小
    if 'tiny' in dataset.lower() or 'imagenet' in dataset.lower():
        input_h, input_w = 64, 64
    else:  # CIFAR
        input_h, input_w = 32, 32
    
    input_size = (batch_size, 3, input_h, input_w)
    single_input_size = (1, 3, input_h, input_w)
    
    # 计算 FLOPs (单张图片)
    flops_info = count_flops_params(model, single_input_size)
    
    # 测量吞吐量 (batch)
    throughput_info = measure_throughput(model, input_size, warmup=30, iterations=100)
    
    result = {
        **flops_info,
        **throughput_info,
        'dataset': dataset,
        'device': DEVICE
    }
    
    return result


# ============================================================================
# 测试部分
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("科研严谨性工具: FLOPs 和吞吐量测量")
    print("=" * 60)
    
    # 测试基本功能
    from models.resnet_cifar import resnet56
    
    model = resnet56(num_classes=10)
    print(f"\nDevice: {DEVICE}")
    print(f"THOP available: {THOP_AVAILABLE}")
    
    # 测量 CIFAR 模型
    print("\n--- ResNet-56 on CIFAR-10 ---")
    result = measure_model_comprehensive(model, 'CIFAR-10', batch_size=128)
    
    print(f"FLOPs: {result['flops']}")
    print(f"Params: {result['params']}")
    print(f"Throughput: {result['throughput_fps']:.1f} images/sec")
    print(f"Latency: {result['latency_ms']:.3f} ms/image")
    
    print("\n" + "=" * 60)
    print("测量工具就绪!")
    print("=" * 60)
