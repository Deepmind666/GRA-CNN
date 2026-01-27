# GRA v5.3 vs Classic Methods Comparison

## ResNet-56 / CIFAR-100

| Method | Ratio 30% | Ratio 50% | Ratio 70% |
|--------|-----------|-----------|-----------|
| **GRA v5.3** | **72.06%** | **70.53%** | - |
| GRA (old) | 60.25% | 60.04% | 59.75% |
| L1-norm | 59.86% | 59.50% | 60.03% |
| FPGM | 59.99% | 59.98% | 59.99% |
| HRank | 60.14% | 59.54% | 59.61% |
| Baseline | 72.56% | 72.56% | 72.56% |

**Key Finding**: GRA v5.3 @ 50% achieves 70.53%, outperforming all classic methods by ~10.5 percentage points.

## ResNet-56 / CIFAR-10

| Method | Ratio 30% | Ratio 50% | Ratio 70% |
|--------|-----------|-----------|-----------|
| **GRA v5.3** | **93.74%** | **92.12%** | **91.93%** |
| Baseline | 93.63% | 93.63% | 93.63% |

**Key Finding**: GRA v5.3 @ 30% exceeds baseline (+0.11%), indicating effective redundancy removal.

## VGG16 / CIFAR-10

| Method | Ratio 30% | Ratio 50% | Ratio 70% |
|--------|-----------|-----------|-----------|
| **GRA v5.3** | **93.87%** | **92.66%** | **88.59%** |
| GRA (old) | 92.10% | 92.12% | 91.88% |
| L1-norm | 91.90% | 91.83% | 91.81% |
| Baseline | 94.01% | 94.01% | 94.01% |

## Data Sources
- v5.3 results: `experiments/v53_results_20260127_010241.csv`
- Classic methods: `experiments/final_consolidated_results.csv`
