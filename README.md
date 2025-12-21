# GRA-CNN: Structured Channel Pruning of CNNs via Gray Relational Analysis

[![Paper](https://img.shields.io/badge/Paper-APIN--Submission-blue)](APIN_Submission/manuscript_apin_final.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GRA-CNN** is a structured pruning framework that optimizes Deep Convolutional Neural Networks by prioritizing **Semantic Alignment**. By leveraging Gray Relational Analysis (GRA), we preserve channels that are functionally coupled with the model's decision-making logic, rather than just high-magnitude weights.

## 🌟 Key Features
- **Semantic Filtering**: Identifies indispensable channels using decision-level logit correlation.
- **Superior Performance**: Achieves **+2.42% accuracy gain** over L1-Norm on Tiny-ImageNet-200 at 50% pruning.
- **Robustness**: Guided by the $\rho$ parameter for stable ranking across diverse architectures (ResNet, VGG).
- **Efficiency**: Modular implementation compatible with CUDA 12 and modern GPUs like RTX 5090.

## 📈 Quick Results
| Dataset | Architecture | Ratio | GRA-CNN (Ours) | Baseline (L1) | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tiny-ImageNet-200 | ResNet-18 | 50% | **58.12%** | 55.70% | +2.42% |
| CIFAR-100 | ResNet-56 | 70% | **59.84%** | 58.52% | +1.32% |

## 🛠️ Usage
Please refer to [reproduction_guide.md](reproduction_guide.md) for detailed instructions on environment setup and running experiments.

## 📚 Citation
If you find this work useful, please cite our APIN submission:
```bibtex
@article{gra_cnn2024,
  title={GRA-CNN: Structured Channel Pruning of CNNs via Gray Relational Analysis},
  author={GDUT-Automation},
  journal={Applied Intelligence},
  year={2024}
}
```
