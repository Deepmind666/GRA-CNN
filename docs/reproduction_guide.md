# GRA-CNN: Structured Channel Pruning via Gray Relational Analysis

This project implements **GRA-CNN**, a novel structured pruning framework guided by Gray Relational Analysis (GRA) to measure semantic alignment between internal activations and decision-level outputs.

## 📂 Project Structure

```text
GRA-CNN/
├── APIN_Submission/      # Submission-ready manuscript (LaTeX) and figures
│   ├── manuscript_apin_final.pdf  # Final submission version
│   ├── vis/                       # High-quality result tables and visualizations
│   └── cover_letter.pdf           # Submission cover letter
├── experiments/          # Industrial-grade experimental suites
│   ├── tinyimagenet_full_experiment.py  # Standalone suite for Tiny-ImageNet-200
│   └── run_imagenet_experiments.bat      # Batch runner for ImageNet-100/1000
├── pruning/              # Core Algorithmic Library (Modular)
│   ├── gra_score.py      # Core GRA implementation (Semantic Alignment)
│   ├── l1_score.py       # Magnitude-based baseline
│   ├── fpgm_score.py     # Geometric Median baseline
│   └── prune_model.py    # Pruning engine for multiple architectures
├── models/               # Architectural Definitions
│   ├── resnet_cifar.py   # ResNet-20/56/110 for CIFAR datasets
│   ├── resnet18_tiny.py  # Optimized ResNet-18 for Tiny-ImageNet
│   └── vgg_cifar.py      # VGG-16 for CIFAR datasets
├── archive/              # Historical drafts, development code, and logs
└── run_all.py            # Master script to reproduce CIFAR experiments
```

## 🚀 Getting Started

### 1. Requirements
Ensure you have a PyTorch environment (CUDA 12.x recommended for RTX 5090).
```bash
pip install torch torchvision numpy pandas matplotlib tqdm
```

### 2. Reproducing CIFAR Experiments
The `run_all.py` script automates the Baseline Training -> GRA Pruning -> Fine-tuning pipeline.
```bash
python run_all.py
```
*   This will iterate through ResNet-20/56 and various pruning ratios (0.3 - 0.7).
*   Results will be consolidated in `archive/results/experiment_results_sota.csv`.

### 3. Reproducing Tiny-ImageNet-200 Experiments
Run the standalone suite designed for deep validation:
```bash
python experiments/tinyimagenet_full_experiment.py --ratios 0.3 0.5 0.7 --methods gra l1 fpgm
```

## 🔬 Core Innovation: Gray Relational Analysis
The importance score $\gamma_i$ for channel $i$ is computed as:
$$ \gamma_i = \text{mean}_j \left( \frac{\Delta_{\text{min}} + \rho \Delta_{\text{max}}}{\Delta_{i,j} + \rho \Delta_{\text{max}}} \right) $$
Where $\Delta_{i,j}$ represents the semantic distance between the $i$-th channel activation and the ground-truth logit for sample $j$.

## 📧 Contact
For questions regarding the implementation or the APIN manuscript, please refer to the corresponding author in `manuscript_apin_final.pdf`.
