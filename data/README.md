# Dataset Layout

Raw public datasets are not committed to this repository. CIFAR-10 and
CIFAR-100 can be downloaded by `torchvision` when the experiment runner is
used. Tiny-ImageNet must be prepared locally.

Expected local layout:

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
  tiny-imagenet-200/
    train/
    val/
```

For Tiny-ImageNet, pass the path explicitly when needed:

```bash
python experiments/run_real_pruning.py --arch resnet18 --dataset tinyimagenet --data-dir data/tiny-imagenet-200 --method GRA-CNN --target_ratio 0.7 --seed 42 --epochs 30
```

The tracked result data used by the submitted paper is in `results/`. Source
CSV tables for the IVC figures/tables are in `paper/tables/source_csv/`.
