# ImageNet-100 实验指南

## 概述
本文档说明如何运行 GRA-CNN 在 ImageNet-100 子集上的实验。

## 数据准备

### 选项 1: ImageNet-100 子集 (推荐)
从完整 ImageNet 中选择前100类：

```bash
# 假设完整 ImageNet 在 /data/imagenet
python scripts/create_imagenet100.py --source /data/imagenet --target ./data/imagenet100
```

### 选项 2: 完整 ImageNet
直接使用完整 ImageNet-1K 数据集：
- 下载链接: https://www.image-net.org/download.php
- 目录结构:
  ```
  data/imagenet/
  ├── train/
  │   ├── n01440764/
  │   ├── n01443537/
  │   └── ...
  └── val/
      ├── n01440764/
      ├── n01443537/
      └── ...
  ```

## 运行实验

### 单次实验
```bash
cd C:\GRA-CNN

# GRA 剪枝 50%
python experiments/imagenet_experiment.py \
    --data-dir ./data/imagenet100 \
    --num-classes 100 \
    --method gra \
    --prune-ratio 0.5 \
    --finetune-epochs 10

# L1 剪枝 50%
python experiments/imagenet_experiment.py \
    --data-dir ./data/imagenet100 \
    --num-classes 100 \
    --method l1 \
    --prune-ratio 0.5 \
    --finetune-epochs 10
```

### 批量实验
```bash
# Windows
experiments\run_imagenet_experiments.bat

# Linux/Mac
bash experiments/run_imagenet_experiments.sh
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `./data/imagenet` | ImageNet 数据目录 |
| `--num-classes` | 100 | 类别数 (100 = ImageNet-100) |
| `--batch-size` | 64 | 批大小 |
| `--prune-ratio` | 0.5 | 剪枝率 |
| `--rho` | 0.5 | GRA 分辨系数 |
| `--finetune-epochs` | 10 | 微调轮数 |
| `--method` | gra | 剪枝方法 (gra/l1) |
| `--mock` | False | 使用模拟数据测试 |

## 预期结果

基于类似工作的估计，在 ImageNet-100 上的预期结果：

| Method | Ratio=0.3 | Ratio=0.5 | Ratio=0.7 |
|--------|-----------|-----------|-----------|
| L1-Norm | ~74.8% | ~73.5% | ~70.2% |
| GRA-CNN | ~75.2% | ~74.1% | ~71.3% |

## 将结果添加到论文

1. 运行实验后，结果保存在 `experiments/imagenet/`
2. 更新 `vis/table_imagenet.tex` 中的数值
3. 在 `manuscript_apin.tex` 中添加：
   ```latex
   \input{vis/table_imagenet.tex}
   ```
4. 重新编译 PDF

## 硬件要求

- GPU: 至少 8GB 显存 (RTX 3070+)
- RAM: 32GB+
- 存储: ImageNet-100 约 15GB, 完整 ImageNet 约 150GB
- 预计时间: 
  - ImageNet-100: 每个实验约 2-3 小时
  - 完整 ImageNet: 每个实验约 8-12 小时
