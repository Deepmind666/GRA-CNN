# RTX 5090 (Blackwell) 环境配置笔记

**日期**：2026-01-23  
**GPU 型号**：NVIDIA GeForce RTX 5090 (Compute Capability: 12.0 / sm_120)  
**系统**：Windows 11 / CUDA 驱动 570+

## 核心挑战
RTX 5090 采用 Blackwell 架构（sm_120），算力核心领先于目前主流的 PyTorch 官方稳定版（如 2.4/2.5，通常构建支持至 sm_90）。直接运行可能导致以下错误：
> `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`

## 解决方案：安装适配版 PyTorch
必须使用包含 CUDA 12.8 支持的 PyTorch Nightly 或预览版本。

### 1. 基础环境
- **Conda 环境名**：`gra311`
- **Python 版本**：3.11

### 2. 安装指令 (Nightly 通道)
在 Conda 环境下执行：
```bash
# 卸载不兼容版本
pip uninstall torch torchvision torchaudio -y

# 安装 cu128 预览版
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. 环境验证
运行以下脚本确认 GPU 已激活：
```python
import torch
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# 核心测试：张量移动与计算
x = torch.ones(1).cuda()
print(f"Success! Tensor on: {x.device}")
```

## 注意事项
- **显存保留**：由于 5090 拥有 32GB 专用显存，多任务并行时需关注内存池分配。
- **驱动要求**：确保 NVIDIA 驱动版本至少为 572.xx 以上，以获得对 sm_120 的内核级支持。
