# GRA-CNN: Global Relevant Analysis for Lightweight Model Pruning

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-85C000.svg?style=for-the-badge&logo=NVIDIA&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-RTX_5090_(Blackwell)-85C000.svg?style=for-the-badge&logo=NVIDIA&logoColor=white)

This repository contains the official implementation of **GRA-CNN** (Global Relevant Analysis), a novel filter pruning method that preserves global semantic information during model compression.

## 🚀 Features

- **Global Relevant Analysis (GRA)**: A metric that correlates filter activations with class-wise margins to identify semantically critical neurons.
- **MORF Fusion**: Multi-Objective Relevant Fusion strategy combining Fisher information, Orthogonality, and GRA scores.
- **Turbo Mode**: Optimized for NVIDIA RTX 5090, supporting 8x parallel experiment execution.
- **Broad Architecture Support**: ResNet-20/56/110, VGG-16, MobileNetV2.

## 🛠️ Environment Setup

### Prerequisites
- Python 3.11+
- CUDA 12.8 (Required for Blackwell / sm_120 support)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YourUsername/GRA-CNN.git
    cd GRA-CNN
    ```

2.  **Install Dependencies**
    Note: For RTX 5090, you **must** use the PyTorch Nightly build.
    ```bash
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install pandas numpy scipy matplotlib tabulate tqdm
    ```

## ⚡ Quick Start (Turbo Mode)

We provide a high-throughput runner optimized for high-end GPUs (e.g., RTX 4090/5090).

**To run the full experiment matrix (98+ configs):**

```bash
python experiments/master_experiment_runner.py
```

*   **Default**: 8x Parallelization (Configured for 24GB+ VRAM).
*   **Output**: Results are saved to `experiments/supplementary_results.csv` and `experiments/master_run_log.txt`.

**To run a single pruning experiment:**

```bash
python experiments/run_real_pruning.py --arch resnet56 --dataset cifar10 --method gra --ratio 0.5
```

## 📊 Directory Structure

- `experiments/`: Core experiment scripts and runners.
- `models/`: Model definitions (ResNet, VGG, MobileNet).
- `pruning/`: Implementations of GRA, L1, FPGM, and HRank prune methods.
- `docs/`: Detailed logs and environmental setup notes.
- `APIN_Submission/`: LaTeX source for the manuscript.

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{gra_cnn_2026,
  title={GRA-CNN: Global Relevant Analysis for Lightweight Model Pruning},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
