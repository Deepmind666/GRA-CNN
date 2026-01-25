# GRA-CNN: Structured Pruning via Gray Relational Analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

This repository serves as the official implementation of the paper **"GRA-CNN: Structured Channel Pruning of CNNs via Gray Relational Analysis"**.

## 🚀 Key Features

GRA-CNN introduces a novel pruning criterion based on **Semantic Alignment**. Instead of removing filters with small weights (Magnitude) or low rank (Geometric), GRA-CNN identifies filters that are functionally aligned with the network's final decision logic (logits).

- **Core Metric**: Gray Relational Analysis (GRA) measuring trend similarity between activation sequences and output logits.
- **Fusion Engine**: Combines GRA with Fisher Information and Orthogonality for robust selection.
- **Advantage**: Superior performance on **Deep Networks** (ResNet-110) and **High Sparsity** (90% pruning) regimes where traditional methods collapse.

## 📊 Performance Highlights

| Architecture | Dataset | Ratio | Metric | Advantage |
|--------------|---------|-------|--------|-----------|
| **ResNet-110** | CIFAR-10 | **90%** | Acc (%) | **GRA (76.4%)** >> L1 (<20%) |
| **VGG-16** | CIFAR-100 | **70%** | Acc (%) | **GRA (66.5%)** > L1 (64.2%) |
| **MobileNetV2**| Tiny-ImgNet| **50%** | Speedup | **1.66x** w/ +2.4% Acc gain |

> **Note**: GRA is designed for "Hard Mode". While it performs comparably to L1 on shallow networks (e.g., ResNet-20 @ 30%), its true power emerges in deep architectures and extreme compression scenarios.

## 🛠️ Usage

### 1. Installation
```bash
git clone https://github.com/Deepmind666/GRA-CNN.git
cd GRA-CNN
pip install -r requirements.txt
```

### 2. Core Algorithm
The implementation of the GRA-Fisher scoring mechanism is located in:
`pruning/core_algorithm.py`

### 3. Running Experiments
To reproduce the ResNet-110 extreme sparsity result:
```bash
python experiments/run_real_pruning.py --arch resnet110 --dataset cifar10 --method gra --ratio 0.9
```

## 📂 Repository Structure

- `pruning/`: Core importance scoring algorithms (GRA, Fisher, L1).
- `models/`: CIFAR-10/100 optimized model definitions (ResNet, VGG, MobileNet).
- `experiments/`: Training and pruning scripts.
- `vis/`: Visualization scripts for reproducing paper figures.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{gra_cnn_2026,
  title={GRA-CNN: Structured Channel Pruning of CNNs via Gray Relational Analysis},
  author={Zhang, Xiaobo and Li, Kangrui and et al.},
  journal={arXiv preprint},
  year={2026}
}
```
