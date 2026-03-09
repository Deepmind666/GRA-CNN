# Checkpoint Placeholders

The public repository does not bundle pretrained baseline weights.

If you want to run the public experiment scripts end-to-end, place the required checkpoints in this directory using filenames compatible with the mappings defined in:

- `experiments/run_chip_worker.py`
- `experiments/run_metric_ablation_worker.py`

Typical examples include:

- `baseline_cifar100_resnet56.pth`
- `baseline_cifar100_vgg16.pth`
- `baseline_cifar10_resnet110.pth`
- `baseline_tiny_resnet18_60ep.pth`
