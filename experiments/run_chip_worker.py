"""Public single-task runner for GRA-CNN and structured pruning baselines."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    from pruning.gra_chip import compute_gra_chip_v31_scores, compute_gra_chip_v32_scores
except Exception:
    compute_gra_chip_v31_scores = None
    compute_gra_chip_v32_scores = None

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
RESULT_DIR = ROOT / "results"

CHECKPOINTS = {
    "resnet56": {
        "cifar10": "checkpoints/resnet56_best_new.pth",
        "cifar100": "checkpoints/baseline_cifar100_resnet56.pth",
    },
    "resnet110": {
        "cifar10": "checkpoints/baseline_cifar10_resnet110.pth",
        "cifar100": "checkpoints/baseline_cifar100_resnet110.pth",
    },
    "resnet18": {
        "tinyimagenet": "checkpoints/baseline_tiny_resnet18_60ep.pth",
    },
    "vgg16": {
        "cifar10": "checkpoints/baseline_cifar10_vgg16.pth",
        "cifar100": "checkpoints/baseline_cifar100_vgg16.pth",
    },
}

ALL_METHODS = [
    "GRA-CNN",
    "L1",
    "Taylor",
    "CHIP",
    "FPGM",
    "HRank",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_WORKER_SEED = 42


# ----------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------
def default_loader_workers() -> int:
    cpu_count = os.cpu_count() or 4
    if cpu_count <= 1:
        return 0
    return max(4, min(12, cpu_count // 2))


def default_scoring_workers() -> int:
    cpu_count = os.cpu_count() or 4
    if cpu_count <= 1:
        return 0
    return max(2, min(8, cpu_count // 3))


def default_train_batch_size(dataset: str) -> int:
    if dataset == "tinyimagenet":
        return 32
    return 1024


def default_scoring_batch_size(dataset: str) -> int:
    if dataset == "tinyimagenet":
        return 32
    return 512


def build_loader_kwargs(workers: int) -> Dict[str, Any]:
    workers = max(int(workers), 0)
    kwargs: Dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def resolve_data_root(data_dir: str, default_subdir: str) -> Path:
    root = Path(data_dir) if data_dir else (ROOT / default_subdir)
    if not root.is_absolute():
        root = ROOT / root
    return root.resolve()


def tune_workers_for_dataset(dataset: str, workers: int, *, scoring: bool) -> int:
    workers = max(int(workers), 0)
    if os.name == "nt" and dataset == "tinyimagenet":
        return min(workers, 4 if not scoring else 2)
    return workers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(_WORKER_SEED + worker_id)
    np.random.seed(_WORKER_SEED + worker_id)
    random.seed(_WORKER_SEED + worker_id)


# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------
def get_dataloaders(
    dataset: str,
    batch_size: int | None = None,
    workers: int | None = None,
    seed: int = 42,
    data_dir: str = "",
) -> Tuple:
    global _WORKER_SEED
    _WORKER_SEED = seed
    if batch_size is None:
        batch_size = default_train_batch_size(dataset)
    if workers is None:
        workers = default_loader_workers()
    workers = tune_workers_for_dataset(dataset, workers, scoring=False)

    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        dset_cls = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        dset_cls = torchvision.datasets.CIFAR100
    elif dataset == "tinyimagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        num_classes = 200
        train_tf = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        root = resolve_data_root(data_dir, "data/tiny-imagenet-200")
        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet path invalid: {root}")

        train_ds = torchvision.datasets.ImageFolder(str(train_dir), transform=train_tf)
        test_ds = torchvision.datasets.ImageFolder(str(val_dir), transform=test_tf)

        g = torch.Generator()
        g.manual_seed(seed)
        loader_kwargs = build_loader_kwargs(workers)
        train_ld = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            worker_init_fn=_worker_init_fn, generator=g,
            **loader_kwargs,
        )
        test_ld = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            **loader_kwargs,
        )
        return train_ld, test_ld, num_classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_root = str(resolve_data_root(data_dir, "data"))
    train_ds = dset_cls(root=data_root, train=True, download=True, transform=train_tf)
    test_ds = dset_cls(root=data_root, train=False, download=True, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs = build_loader_kwargs(workers)
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        worker_init_fn=_worker_init_fn, generator=g,
        **loader_kwargs,
    )
    test_ld = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
    )
    return train_ld, test_ld, num_classes


def get_scoring_dataloader(
    dataset: str,
    batch_size: int | None = None,
    workers: int | None = None,
    seed: int = 42,
    data_dir: str = "",
) -> torch.utils.data.DataLoader:
    """Deterministic loader for scoring (no augmentation)."""
    if batch_size is None:
        batch_size = default_scoring_batch_size(dataset)
    if workers is None:
        workers = default_scoring_workers()
    workers = tune_workers_for_dataset(dataset, workers, scoring=True)
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dset_cls = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        dset_cls = torchvision.datasets.CIFAR100
    elif dataset == "tinyimagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        tf = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        root = resolve_data_root(data_dir, "data/tiny-imagenet-200")
        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Tiny-ImageNet path invalid: {root}")
        ds = torchvision.datasets.ImageFolder(str(train_dir), transform=tf)
        g = torch.Generator()
        g.manual_seed(seed)
        loader_kwargs = build_loader_kwargs(workers)
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            generator=g,
            **loader_kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    data_root = str(resolve_data_root(data_dir, "data"))
    ds = dset_cls(root=data_root, train=True, download=False, transform=tf)
    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs = build_loader_kwargs(workers)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        generator=g,
        **loader_kwargs,
    )


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------
def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet56":
        from models.resnet_cifar import resnet56
        return resnet56(num_classes=num_classes)
    if arch == "resnet110":
        from models.resnet_cifar import resnet110
        return resnet110(num_classes=num_classes)
    if arch == "resnet18":
        from models.resnet18_tiny import resnet18_tiny
        return resnet18_tiny(num_classes=num_classes)
    if arch == "vgg16":
        from models.vgg_cifar import vgg16
        return vgg16(num_classes=num_classes)
    raise ValueError(f"Unsupported arch: {arch}")


def load_checkpoint(model: nn.Module, arch: str, dataset: str) -> None:
    if arch not in CHECKPOINTS or dataset not in CHECKPOINTS[arch]:
        raise ValueError(f"No checkpoint mapping for arch={arch}, dataset={dataset}")
    ckpt_path = str(ROOT / CHECKPOINTS[arch][dataset])
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)


# ----------------------------------------------------------------
# Eval / Finetune
# ----------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += model(x).argmax(1).eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module, loader, device: torch.device, num_classes: int
) -> Dict[str, Any]:
    """Evaluate top-1 + class-level robustness metrics."""
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        y_np = y.cpu().numpy()
        p_np = pred.cpu().numpy()
        for t, p in zip(y_np, p_np):
            conf[int(t), int(p)] += 1

    top1 = 100.0 * correct / max(total, 1)

    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)
    pred_count = conf.sum(axis=0).astype(np.float64)
    per_class_acc = np.divide(tp, np.maximum(support, 1.0)) * 100.0

    precision = np.divide(tp, np.maximum(pred_count, 1.0))
    recall = np.divide(tp, np.maximum(support, 1.0))
    f1 = np.divide(2.0 * precision * recall, np.maximum(precision + recall, 1e-12))
    macro_f1 = float(np.mean(f1) * 100.0)
    balanced_acc = float(np.mean(per_class_acc))
    worst_class_acc = float(np.min(per_class_acc))
    k = min(10, num_classes)
    worst10_mean_acc = float(np.mean(np.sort(per_class_acc)[:k]))
    collapse_5 = int(np.sum(per_class_acc < 5.0))
    collapse_10 = int(np.sum(per_class_acc < 10.0))

    return {
        "top1": float(top1),
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "worst_class_acc": worst_class_acc,
        "worst10_mean_acc": worst10_mean_acc,
        "collapse_class_count_5pct": collapse_5,
        "collapse_class_count_10pct": collapse_10,
    }


def finetune(
    model: nn.Module, train_ld, test_ld, device: torch.device, epochs: int = 30, lr: float = 0.01
) -> float:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    best_acc = 0.0

    for ep in range(epochs):
        model.train()
        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            acc = evaluate(model, test_ld, device)
            best_acc = max(best_acc, acc)
    return best_acc


# ----------------------------------------------------------------
# Taylor scoring (self-contained)
# ----------------------------------------------------------------
def compute_taylor_scores(
    model: nn.Module, loader, device: torch.device, target_layers: List[str], num_batches: int = 12
) -> Dict[str, np.ndarray]:
    """First-order Taylor: mean(|act * grad|) per channel."""
    model.eval()
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}
    hooks = []

    target_set = set(target_layers)
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Conv2d):
            continue

        def _make_hook(ln):
            def _hook(_m, _inp, out):
                activations[ln] = out
                if isinstance(out, torch.Tensor):
                    out.register_hook(lambda g, _ln=ln: gradients.__setitem__(_ln, g))
            return _hook
        hooks.append(module.register_forward_hook(_make_hook(name)))

    accum: Dict[str, np.ndarray] = {}
    count = 0
    try:
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            activations.clear()
            gradients.clear()
            model.zero_grad(set_to_none=True)
            logits = model(x)
            F.cross_entropy(logits, y).backward()

            for ln in target_layers:
                act = activations.get(ln)
                grad = gradients.get(ln)
                if act is None or grad is None:
                    continue
                sc = torch.abs(act.detach() * grad.detach()).mean(dim=[0, 2, 3]).cpu().numpy()
                if ln not in accum:
                    accum[ln] = np.zeros_like(sc, dtype=np.float64)
                accum[ln] += sc.astype(np.float64)
            count += 1
    finally:
        for h in hooks:
            h.remove()

    return {k: v / max(count, 1) for k, v in accum.items()}


def compute_fisher_scores(
    model: nn.Module, loader, device: torch.device, target_layers: List[str], num_batches: int = 12
) -> Dict[str, np.ndarray]:
    """Activation Fisher proxy: mean((act * grad)^2) per channel."""
    model.eval()
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}
    hooks = []

    target_set = set(target_layers)
    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Conv2d):
            continue

        def _make_hook(ln):
            def _hook(_m, _inp, out):
                activations[ln] = out
                if isinstance(out, torch.Tensor):
                    out.register_hook(lambda g, _ln=ln: gradients.__setitem__(_ln, g))
            return _hook
        hooks.append(module.register_forward_hook(_make_hook(name)))

    accum: Dict[str, np.ndarray] = {}
    count = 0
    try:
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            activations.clear()
            gradients.clear()
            model.zero_grad(set_to_none=True)
            logits = model(x)
            F.cross_entropy(logits, y).backward()

            for ln in target_layers:
                act = activations.get(ln)
                grad = gradients.get(ln)
                if act is None or grad is None:
                    continue
                fisher = ((act.detach() * grad.detach()) ** 2).mean(dim=[0, 2, 3]).cpu().numpy()
                if ln not in accum:
                    accum[ln] = np.zeros_like(fisher, dtype=np.float64)
                accum[ln] += fisher.astype(np.float64)
            count += 1
    finally:
        for h in hooks:
            h.remove()

    return {k: v / max(count, 1) for k, v in accum.items()}


# ----------------------------------------------------------------
# L1 scoring
# ----------------------------------------------------------------
def compute_l1_scores(model: nn.Module) -> Dict[str, np.ndarray]:
    scores = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            scores[name] = m.weight.data.abs().sum(dim=(1, 2, 3)).cpu().numpy()
    return scores


def compute_fpgm_scores(model: nn.Module, metric: str = "l2") -> Dict[str, np.ndarray]:
    """FPGM: geometric median distance as importance (larger = more unique)."""
    p = 2 if metric.lower() == "l2" else 1
    scores = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.data.view(m.out_channels, -1)
            if m.out_channels < 2:
                scores[name] = np.ones(m.out_channels, dtype=np.float32)
                continue
            dist = torch.cdist(w, w, p=p)
            uniq = dist.sum(dim=1) / max(m.out_channels - 1, 1)
            scores[name] = uniq.detach().cpu().numpy().astype(np.float32)
    return scores


def compute_hrank_scores(
    model: nn.Module, loader, device: torch.device,
    target_layers: List[str], num_batches: int = 5,
) -> Dict[str, np.ndarray]:
    """HRank: average feature map rank as importance (higher rank = more important)."""
    rank_sums: Dict[str, np.ndarray] = {}
    count = 0
    features: Dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(layer_name):
        def hook_fn(module, inp, out):
            features[layer_name] = out.detach()
        return hook_fn

    for name, m in model.named_modules():
        if name in target_layers and isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device, non_blocking=True)
            features.clear()
            model(images)
            for lname, feat in features.items():
                B, C, H, W = feat.shape
                ranks = np.zeros(C, dtype=np.float32)
                for b in range(B):
                    for c in range(C):
                        fm = feat[b, c].cpu().float().numpy()
                        ranks[c] += np.linalg.matrix_rank(fm)
                if lname not in rank_sums:
                    rank_sums[lname] = ranks
                else:
                    rank_sums[lname] += ranks
            count += images.size(0)

    for h in hooks:
        h.remove()

    scores = {}
    for lname, rs in rank_sums.items():
        scores[lname] = (rs / max(count, 1)).astype(np.float32)
    return scores


# ----------------------------------------------------------------
# CHIP / GRA-CHIP scoring (from pruning/gra_chip.py)
# ----------------------------------------------------------------
def _collect_feature_maps(
    model: nn.Module, loader, device: torch.device,
    num_batches: int = 8, target_layers: List[str] | None = None,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """Collect activation maps and labels from target conv layers."""
    features: Dict[str, List[torch.Tensor]] = defaultdict(list)
    labels_list: List[torch.Tensor] = []
    hooks = []

    target_set = set(target_layers) if target_layers else None
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if target_set is not None and name not in target_set:
            continue

        def make_hook(n):
            def hook_fn(module, inp, out):
                features[n].append(out.detach().cpu())
            return hook_fn
        hooks.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(device, non_blocking=True))
            labels_list.append(y)

    for h in hooks:
        h.remove()

    feats = {k: torch.cat(v, dim=0) for k, v in features.items()}
    labels = torch.cat(labels_list, dim=0).numpy()
    return feats, labels


def compute_chip_ci(feat: torch.Tensor) -> np.ndarray:
    """CHIP original: Channel Independence via nuclear norm drop.
    Higher score = more important channel."""
    N, C, H, W = feat.shape
    mat = feat.reshape(N, C, -1)  # [N, C, HW]
    ci = np.zeros(C, dtype=np.float64)
    n_sub = min(N, 40)  # subsample for speed

    for n in range(n_sub):
        m_n = mat[n].float()  # [C, HW], use float32 for speed
        orig_norm = torch.linalg.norm(m_n, ord="nuc").item()
        for j in range(C):
            reduced = m_n.clone()
            reduced[j] = 0
            ci[j] += orig_norm - torch.linalg.norm(reduced, ord="nuc").item()

    ci /= n_sub
    return ci


def compute_gra_redundancy(feat: torch.Tensor, rho: float = 0.5) -> np.ndarray:
    """GRA inter-channel redundancy. Higher = more redundant (should prune)."""
    N, C, H, W = feat.shape
    profile = feat.mean(dim=(2, 3)).numpy().astype(np.float64)  # [N, C]

    p_min = profile.min(axis=0, keepdims=True)
    p_max = profile.max(axis=0, keepdims=True)
    profile = (profile - p_min) / (p_max - p_min + 1e-8)

    redundancy = np.zeros(C, dtype=np.float64)
    for i in range(C):
        delta = np.abs(profile[:, i:i + 1] - profile)  # [N, C]
        d_min = delta.min()
        d_max = delta.max()
        grc = (d_min + rho * d_max) / (delta + rho * d_max + 1e-8)
        grc_mean = grc.mean(axis=0)
        grc_mean[i] = 0
        redundancy[i] = grc_mean.sum() / max(C - 1, 1)
    return redundancy


def compute_ca_gra_redundancy(feat: torch.Tensor, labels: np.ndarray, rho: float = 0.5) -> np.ndarray:
    """Class-Aware GRA: channel is redundant only if redundant across ALL classes."""
    C = feat.shape[1]
    profile = feat.mean(dim=(2, 3)).numpy().astype(np.float64)

    classes = np.unique(labels)
    per_class_red = []

    for cls in classes:
        mask = labels == cls
        if mask.sum() < 2:
            continue
        p = profile[mask]
        p_min = p.min(axis=0, keepdims=True)
        p_max = p.max(axis=0, keepdims=True)
        p = (p - p_min) / (p_max - p_min + 1e-8)

        red = np.zeros(C, dtype=np.float64)
        for i in range(C):
            delta = np.abs(p[:, i:i + 1] - p)
            d_min, d_max = delta.min(), delta.max()
            grc = (d_min + rho * d_max) / (delta + rho * d_max + 1e-8)
            grc_mean = grc.mean(axis=0)
            grc_mean[i] = 0
            red[i] = grc_mean.sum() / max(C - 1, 1)
        per_class_red.append(red)

    if not per_class_red:
        return compute_gra_redundancy(feat, rho)
    return np.min(per_class_red, axis=0)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    r = x.max() - x.min()
    return (x - x.min()) / (r + 1e-8) if r > 1e-8 else np.ones_like(x) * 0.5


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Quick Spearman rank correlation between two 1D arrays."""
    if len(a) < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    except ImportError:
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        n = len(a)
        d = ra - rb
        return float(1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


# ---- Scoring entry points ----

def score_chip_only(
    model: nn.Module, loader, device: torch.device,
    target_layers: List[str], num_batches: int = 8,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    feats, labels = _collect_feature_maps(model, loader, device, num_batches, target_layers)
    scores: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}

    for name in target_layers:
        feat = feats.get(name)
        if feat is None or feat.shape[1] < 2:
            continue
        ci = compute_chip_ci(feat)
        scores[name] = _minmax(ci)
        meta[name] = {"raw_ci_std": float(np.std(ci))}

    meta["__global__"] = {"version": "CHIP", "method": "CHIP"}
    return scores, meta


def score_gra_chip_base(
    model: nn.Module, loader, device: torch.device,
    target_layers: List[str], num_batches: int = 8,
    alpha: float = 0.5, rho: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    feats, labels = _collect_feature_maps(model, loader, device, num_batches, target_layers)
    scores: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}

    for name in target_layers:
        feat = feats.get(name)
        if feat is None or feat.shape[1] < 2:
            continue
        ci_raw = compute_chip_ci(feat)
        gra_raw = compute_gra_redundancy(feat, rho)
        ci_n = _minmax(ci_raw)
        gra_n = _minmax(gra_raw)
        scores[name] = alpha * ci_n + (1 - alpha) * (1 - gra_n)
        meta[name] = {
            "spearman_ci_gra": _spearman_corr(ci_n, 1 - gra_n),
        }

    meta["__global__"] = {"version": "GRA-CHIP-base", "method": "GRA-CHIP-base", "alpha": alpha, "rho": rho}
    return scores, meta


def score_gra_chip_v2(
    model: nn.Module, loader, device: torch.device,
    target_layers: List[str], num_batches: int = 8,
    alpha: float = 0.5, rho: float = 0.5,
    use_frd: bool = True, use_ca: bool = True, use_caf: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    feats, labels = _collect_feature_maps(model, loader, device, num_batches, target_layers)
    scores: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    frd_penalties = []
    agreement_means = []

    for name in target_layers:
        feat = feats.get(name)
        if feat is None or feat.shape[1] < 2:
            continue

        ci_raw = compute_chip_ci(feat)
        ci = _minmax(ci_raw)

        if use_ca:
            gra_red = _minmax(compute_ca_gra_redundancy(feat, labels, rho))
        else:
            gra_red = _minmax(compute_gra_redundancy(feat, rho))
        gra_imp = 1.0 - gra_red

        frd_pen = 0.0
        if use_frd:
            disagreement = np.clip(ci - gra_imp, 0, None)
            frd_pen = float(disagreement.mean())
            ci = ci - 0.3 * disagreement

        if use_caf:
            agreement = 1.0 - np.abs(ci - gra_imp)
            w_gra = alpha * agreement
            w_ci = 1.0 - w_gra
            scores[name] = w_ci * ci + w_gra * gra_imp
            agreement_means.append(float(agreement.mean()))
        else:
            scores[name] = alpha * ci + (1 - alpha) * gra_imp

        meta[name] = {
            "spearman_ci_gra": _spearman_corr(_minmax(ci_raw), gra_imp),
            "frd_penalty_mean": frd_pen,
        }
        frd_penalties.append(frd_pen)

    meta["__global__"] = {
        "version": "GRA-CHIP-v2",
        "method": "GRA-CHIP-v2",
        "alpha": alpha,
        "rho": rho,
        "use_frd": use_frd,
        "use_ca": use_ca,
        "use_caf": use_caf,
        "mean_frd_penalty": float(np.mean(frd_penalties)) if frd_penalties else 0.0,
        "mean_agreement": float(np.mean(agreement_means)) if agreement_means else 0.0,
    }
    return scores, meta


def score_gra_chip_v3(
    model: nn.Module,
    loader,
    device: torch.device,
    target_layers: List[str],
    pruning_ratio: float,
    num_batches: int = 8,
    rho: float = 0.5,
    use_ca: bool = True,
    enable_boundary: bool = True,
    enable_quality_gate: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """v3: CHIP/Taylor/Fisher anchor + GRA boundary-only refinement."""
    taylor_scores = compute_taylor_scores(model, loader, device, target_layers, num_batches=num_batches)
    fisher_scores = compute_fisher_scores(model, loader, device, target_layers, num_batches=num_batches)
    feats, labels = _collect_feature_maps(model, loader, device, num_batches, target_layers)

    r = float(pruning_ratio)
    low_q, high_q = (0.40, 0.60) if r >= 0.7 else (0.35, 0.65)

    layer_pack: Dict[str, Dict[str, Any]] = {}
    for name in target_layers:
        feat = feats.get(name)
        if feat is None or feat.shape[1] < 2:
            continue

        chip_n = _minmax(compute_chip_ci(feat))
        taylor_n = _minmax(taylor_scores.get(name, chip_n))
        fisher_n = _minmax(fisher_scores.get(name, chip_n))
        anchor = _minmax(0.60 * chip_n + 0.25 * taylor_n + 0.15 * fisher_n)

        if use_ca:
            gra_red = _minmax(compute_ca_gra_redundancy(feat, labels, rho=rho))
        else:
            gra_red = _minmax(compute_gra_redundancy(feat, rho=rho))
        sem_imp = 1.0 - gra_red

        r_score = float(np.clip(1.0 - np.mean(np.abs(anchor - sem_imp)), 0.0, 1.0))
        u_score = float(np.clip(1.0 - np.std(sem_imp), 0.0, 1.0))
        c_score = float(np.clip(np.percentile(sem_imp, 90) - np.percentile(sem_imp, 10), 0.0, 1.0))
        quality = float(np.clip(0.45 * r_score + 0.35 * (1.0 - u_score) + 0.20 * c_score, 0.0, 1.0))

        layer_pack[name] = {
            "anchor": anchor,
            "semantic": sem_imp,
            "quality": quality,
            "R": r_score,
            "U": u_score,
            "C": c_score,
        }

    q_values = [v["quality"] for v in layer_pack.values()]
    if q_values:
        tau_candidate = max(0.58, float(np.percentile(q_values, 70.0)))
        gate_on_candidate = float(np.mean(np.asarray(q_values) >= tau_candidate))
        tau_dyn = float(np.percentile(q_values, 55.0)) if gate_on_candidate < 0.40 else tau_candidate
    else:
        tau_candidate = 0.58
        gate_on_candidate = 0.0
        tau_dyn = 0.58

    scores: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {}
    gate_on_layers = 0
    semantic_layers = 0
    total_swaps = 0

    for name, pack in layer_pack.items():
        base = np.asarray(pack["anchor"], dtype=np.float64)
        sem = np.asarray(pack["semantic"], dtype=np.float64)
        quality = float(pack["quality"])
        c_num = len(base)
        keep_k = int(np.clip(round((1.0 - r) * c_num), 1, c_num))
        keep_floor = max(4, int(np.ceil(0.10 * c_num)))
        keep_k = max(keep_k, min(keep_floor, c_num))

        order = np.argsort(base)[::-1]
        keep_mask = np.zeros(c_num, dtype=bool)
        keep_mask[order[:keep_k]] = True

        gate_on = True
        if enable_quality_gate:
            gate_on = quality >= tau_dyn

        swaps = 0
        q_conf = 0.0
        if gate_on:
            gate_on_layers += 1
            if enable_boundary:
                # Boundary zone must be centered on keep/prune threshold, not global rank.
                band_width = max(2, int(round((high_q - low_q) * c_num)))
                lo = max(0, keep_k - band_width // 2)
                hi = min(c_num, keep_k + band_width // 2 + 1)
                if hi <= lo:
                    hi = min(c_num, lo + 2)
                candidate = order[lo:hi]
            else:
                candidate = order

            kept_candidate = [idx for idx in candidate if keep_mask[idx]]
            pruned_candidate = [idx for idx in candidate if not keep_mask[idx]]
            if kept_candidate and pruned_candidate:
                semantic_layers += 1
                kept_candidate.sort(key=lambda idx: (sem[idx], base[idx]))
                pruned_candidate.sort(key=lambda idx: (-sem[idx], -base[idx]))

                q_conf = float(np.clip((quality - tau_dyn) / max(1.0 - tau_dyn, 1e-8), 0.0, 1.0))
                raw_swaps = int(round(0.10 * c_num * (1.0 + q_conf)))
                max_swaps = max(2, raw_swaps)
                max_swaps = min(max_swaps, len(kept_candidate), len(pruned_candidate))

                for i in range(max_swaps):
                    out_idx = kept_candidate[i]
                    in_idx = pruned_candidate[i]
                    sem_gain = float(sem[in_idx] - sem[out_idx])
                    anchor_gap = float(base[out_idx] - base[in_idx])
                    if sem_gain >= 0.08 and anchor_gap <= 0.06:
                        keep_mask[out_idx] = False
                        keep_mask[in_idx] = True
                        swaps += 1
                # Fallback: if strict condition triggers zero swap, apply tiny conservative swap.
                if swaps == 0 and max_swaps > 0:
                    out_idx = kept_candidate[0]
                    in_idx = pruned_candidate[0]
                    sem_gain = float(sem[in_idx] - sem[out_idx])
                    anchor_gap = float(base[out_idx] - base[in_idx])
                    if sem_gain >= 0.03 and anchor_gap <= 0.08:
                        keep_mask[out_idx] = False
                        keep_mask[in_idx] = True
                        swaps = 1
        total_swaps += swaps
        final = np.array(base, copy=True)
        final[keep_mask] += 1.0
        scores[name] = final

        meta[name] = {
            "quality": quality,
            "R": float(pack["R"]),
            "U": float(pack["U"]),
            "C": float(pack["C"]),
            "gate_on": bool(gate_on),
            "swaps": int(swaps),
            "q_conf": float(q_conf),
            "keep_k": int(keep_k),
        }

    n_layers = max(len(scores), 1)
    meta["__global__"] = {
        "version": "GRA-CHIP-v3",
        "method": "GRA-CHIP-v3",
        "anchor_weights": {"chip": 0.60, "taylor": 0.25, "fisher": 0.15},
        "tau_base": 0.58,
        "tau_candidate": float(tau_candidate),
        "tau_dyn": float(tau_dyn),
        "gate_quality_percentile": 70.0,
        "gate_on_rate_candidate": float(gate_on_candidate),
        "min_gate_on_rate": 0.40,
        "boundary_quantiles": [float(low_q), float(high_q)],
        "gate_on_rate": float(gate_on_layers / n_layers),
        "semantic_applied_rate": float(semantic_layers / n_layers),
        "total_swaps": int(total_swaps),
        "avg_swaps_per_layer": float(total_swaps / n_layers),
        "quality_mean": float(np.mean(q_values)) if q_values else 0.0,
        "rho": float(rho),
        "use_ca": bool(use_ca),
        "enable_boundary": bool(enable_boundary),
        "enable_quality_gate": bool(enable_quality_gate),
    }
    return scores, meta


def score_gra_chip_v31(
    model: nn.Module,
    loader,
    device: torch.device,
    target_layers: List[str],
    pruning_ratio: float,
    architecture: str,
    num_batches: int = 8,
    use_ca: bool = True,
    enable_boundary: bool = True,
    enable_quality_gate: bool = True,
    enable_consensus: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """v3.1: architecture-aware gate + consensus-constrained boundary swaps."""
    if compute_gra_chip_v31_scores is None:
        raise ImportError("compute_gra_chip_v31_scores is unavailable in pruning.gra_chip")
    taylor_scores = compute_taylor_scores(model, loader, device, target_layers, num_batches=num_batches)
    fisher_scores = compute_fisher_scores(model, loader, device, target_layers, num_batches=num_batches)
    scores, meta = compute_gra_chip_v31_scores(
        model=model,
        dataloader=loader,
        device=device,
        architecture=architecture,
        num_batches=num_batches,
        pruning_ratio=pruning_ratio,
        target_layers=target_layers,
        taylor_scores=taylor_scores,
        fisher_scores=fisher_scores,
        use_ca=use_ca,
        enable_boundary=enable_boundary,
        enable_quality_gate=enable_quality_gate,
        enable_consensus=enable_consensus,
    )
    return scores, meta


def score_gra_chip_v32(
    model: nn.Module,
    loader,
    device: torch.device,
    target_layers: List[str],
    pruning_ratio: float,
    architecture: str,
    num_batches: int = 8,
    use_ca: bool = True,
    enable_boundary: bool = True,
    enable_quality_gate: bool = True,
    enable_consensus: bool = True,
    enable_risk: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """v3.2: v3 anchor + CCR risk-constrained boundary refinement."""
    if compute_gra_chip_v32_scores is None:
        raise ImportError("compute_gra_chip_v32_scores is unavailable in pruning.gra_chip")
    taylor_scores = compute_taylor_scores(model, loader, device, target_layers, num_batches=num_batches)
    fisher_scores = compute_fisher_scores(model, loader, device, target_layers, num_batches=num_batches)
    scores, meta = compute_gra_chip_v32_scores(
        model=model,
        dataloader=loader,
        device=device,
        architecture=architecture,
        num_batches=num_batches,
        pruning_ratio=pruning_ratio,
        target_layers=target_layers,
        taylor_scores=taylor_scores,
        fisher_scores=fisher_scores,
        use_ca=use_ca,
        enable_boundary=enable_boundary,
        enable_quality_gate=enable_quality_gate,
        enable_consensus=enable_consensus,
        enable_risk=enable_risk,
    )
    return scores, meta


# ----------------------------------------------------------------
# Unified scoring dispatch
# ----------------------------------------------------------------
def get_target_layers(model: nn.Module) -> List[str]:
    """Determine prunable layers based on architecture."""
    layers: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if hasattr(model, "layer1") and hasattr(model, "layer2") and hasattr(model, "layer3"):
            # ResNet: only prune conv1 inside residual blocks (stage_mid_only)
            if ("layer" in name) and name.endswith("conv1"):
                layers.append(name)
        elif hasattr(model, "features"):
            # VGG: all conv layers in features
            if name.startswith("features."):
                layers.append(name)
        else:
            layers.append(name)
    return layers


def compute_scores(
    model: nn.Module, method: str, scoring_loader, device: torch.device,
    target_layers: List[str], num_batches: int = 8, pruning_ratio: float = 0.7,
    architecture: str = "resnet56", eval_loader=None,
    hrank_batches: int = 5, fpgm_metric: str = "l2", gra_rho: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    t0 = time.time()

    if method == "L1":
        all_l1 = compute_l1_scores(model)
        s = {k: all_l1[k] for k in target_layers if k in all_l1}
        m = {"__global__": {"version": "L1", "method": "L1"}}
    elif method == "Taylor":
        s = compute_taylor_scores(model, scoring_loader, device, target_layers, num_batches)
        m = {"__global__": {"version": "Taylor", "method": "Taylor"}}
    elif method == "CHIP":
        s, m = score_chip_only(model, scoring_loader, device, target_layers, num_batches)
    elif method == "GRA-CHIP-base":
        s, m = score_gra_chip_base(model, scoring_loader, device, target_layers, num_batches)
    elif method == "GRA-CHIP-v2":
        s, m = score_gra_chip_v2(model, scoring_loader, device, target_layers, num_batches,
                                  use_frd=True, use_ca=True, use_caf=True)
    elif method == "GRA-CHIP-v2-noFRD":
        s, m = score_gra_chip_v2(model, scoring_loader, device, target_layers, num_batches,
                                  use_frd=False, use_ca=True, use_caf=True)
    elif method == "GRA-CHIP-v2-noCA":
        s, m = score_gra_chip_v2(model, scoring_loader, device, target_layers, num_batches,
                                  use_frd=True, use_ca=False, use_caf=True)
    elif method == "GRA-CHIP-v2-noCAF":
        s, m = score_gra_chip_v2(model, scoring_loader, device, target_layers, num_batches,
                                  use_frd=True, use_ca=True, use_caf=False)
    elif method in ("GRA-CHIP-v3", "GRA-CNN"):
        s, m = score_gra_chip_v3(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            rho=gra_rho,
            use_ca=True, enable_boundary=True, enable_quality_gate=True,
        )
        if method == "GRA-CNN":
            m.setdefault("__global__", {})["version"] = "GRA-CNN"
    elif method == "GRA-CHIP-v3-noCA":
        s, m = score_gra_chip_v3(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            use_ca=False, enable_boundary=True, enable_quality_gate=True,
        )
    elif method == "GRA-CHIP-v3-noBoundary":
        s, m = score_gra_chip_v3(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            use_ca=True, enable_boundary=False, enable_quality_gate=True,
        )
    elif method == "GRA-CHIP-v3-noGate":
        s, m = score_gra_chip_v3(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            use_ca=True, enable_boundary=True, enable_quality_gate=False,
        )
    elif method == "GRA-CHIP-v3.1":
        s, m = score_gra_chip_v31(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, architecture=architecture, num_batches=num_batches,
            use_ca=True, enable_boundary=True, enable_quality_gate=True, enable_consensus=True,
        )
    elif method == "GRA-CHIP-v3.1-noConsensus":
        s, m = score_gra_chip_v31(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, architecture=architecture, num_batches=num_batches,
            use_ca=True, enable_boundary=True, enable_quality_gate=True, enable_consensus=False,
        )
    elif method == "GRA-CHIP-v3.1-noBoundary":
        s, m = score_gra_chip_v31(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, architecture=architecture, num_batches=num_batches,
            use_ca=True, enable_boundary=False, enable_quality_gate=True, enable_consensus=True,
        )
    elif method == "GRA-CHIP-v3.1-noGate":
        s, m = score_gra_chip_v31(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, architecture=architecture, num_batches=num_batches,
            use_ca=True, enable_boundary=True, enable_quality_gate=False, enable_consensus=True,
        )
    elif method == "GRA-CHIP-v3.2":
        s, m = score_gra_chip_v32(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, architecture=architecture, num_batches=num_batches,
            use_ca=True, enable_boundary=True, enable_quality_gate=True,
            enable_consensus=True, enable_risk=True,
        )
    elif method == "GRA-CHIP-v3.2-noRisk":
        s, m = score_gra_chip_v32(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, architecture=architecture, num_batches=num_batches,
            use_ca=True, enable_boundary=True, enable_quality_gate=True,
            enable_consensus=True, enable_risk=False,
        )
    elif method == "GRA-CHIP-v4":
        from pruning.gra_chip_v4 import score_gra_chip_v4
        taylor_s = compute_taylor_scores(model, scoring_loader, device, target_layers, num_batches)
        fisher_s = compute_fisher_scores(model, scoring_loader, device, target_layers, num_batches)
        s, m = score_gra_chip_v4(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            taylor_scores=taylor_s, fisher_scores=fisher_s,
        )
    elif method == "GRA-CHIP-v4-noGRA":
        from pruning.gra_chip_v4 import score_gra_chip_v4
        taylor_s = compute_taylor_scores(model, scoring_loader, device, target_layers, num_batches)
        fisher_s = compute_fisher_scores(model, scoring_loader, device, target_layers, num_batches)
        s, m = score_gra_chip_v4(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            w_chip=0.55, w_gra=0.0, w_taylor=0.25, w_fisher=0.20,
            taylor_scores=taylor_s, fisher_scores=fisher_s,
        )
    elif method in ("GRA-CHIP-v4C", "GRA-CHIP-v4C-noProtect", "GRA-CHIP-v4C-fixedBudget"):
        from pruning.gra_chip_v4c import score_gra_chip_v4c
        taylor_s = compute_taylor_scores(model, scoring_loader, device, target_layers, num_batches)
        fisher_s = compute_fisher_scores(model, scoring_loader, device, target_layers, num_batches)
        enable_prot = method != "GRA-CHIP-v4C-noProtect"
        adapt_lam = method != "GRA-CHIP-v4C-fixedBudget"
        s, m, _pmasks = score_gra_chip_v4c(
            model, scoring_loader, device, target_layers,
            pruning_ratio=pruning_ratio, num_batches=num_batches,
            architecture=architecture,
            taylor_scores=taylor_s, fisher_scores=fisher_s,
            enable_protection=enable_prot, adaptive_lambda=adapt_lam,
            eval_loader=eval_loader,
        )
    elif method == "FPGM":
        all_fpgm = compute_fpgm_scores(model, metric=fpgm_metric)
        s = {k: all_fpgm[k] for k in target_layers if k in all_fpgm}
        m = {"__global__": {"version": "FPGM", "method": "FPGM", "fpgm_metric": fpgm_metric}}
    elif method == "HRank":
        s = compute_hrank_scores(model, scoring_loader, device, target_layers, num_batches=hrank_batches)
        m = {"__global__": {"version": "HRank", "method": "HRank", "hrank_batches": int(hrank_batches)}}
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0
    m["__global__"]["score_time_s"] = float(elapsed)
    m["__global__"]["num_scoring_batches"] = int(num_batches)
    return s, m


# ----------------------------------------------------------------
# Structural pruning
# ----------------------------------------------------------------
def scores_to_masks(
    scores: Dict[str, np.ndarray], ratio: float, target_layers: List[str],
) -> Dict[str, np.ndarray]:
    """Convert importance scores to binary keep-masks for target layers."""
    masks: Dict[str, np.ndarray] = {}
    for name in target_layers:
        s = scores.get(name)
        if s is None:
            continue
        n_ch = len(s)
        n_prune = int(round(n_ch * ratio))
        n_prune = min(n_prune, n_ch - 1)  # keep at least 1
        if n_prune <= 0:
            masks[name] = np.ones(n_ch, dtype=bool)
            continue
        idx_prune = np.argsort(s)[:n_prune]
        mask = np.ones(n_ch, dtype=bool)
        mask[idx_prune] = False
        masks[name] = mask
    return masks


def apply_pruning(
    model: nn.Module, masks: Dict[str, np.ndarray],
    arch: str, num_classes: int, device: torch.device,
) -> nn.Module:
    from pruning.prune_model import build_new_resnet_from_mask, build_new_vgg_from_mask

    if arch.startswith("resnet"):
        pruned = build_new_resnet_from_mask(model, masks)
    elif arch.startswith("vgg"):
        pruned = build_new_vgg_from_mask(model, masks)
    else:
        raise ValueError(f"Unsupported arch for pruning: {arch}")
    return pruned.to(device)


# ----------------------------------------------------------------
# Signal quality diagnostic
# ----------------------------------------------------------------
def run_signal_diagnostic(
    model: nn.Module, scoring_loader, device: torch.device, target_layers: List[str],
) -> Dict[str, Any]:
    """Check correlation between CHIP, GRA, L1, Taylor signals.
    This runs BEFORE the full experiment to verify signal orthogonality."""
    print("[DIAG] Computing signal correlations...")

    l1_scores = compute_l1_scores(model)
    taylor_scores = compute_taylor_scores(model, scoring_loader, device, target_layers, 8)
    feats, labels = _collect_feature_maps(model, scoring_loader, device, 8, target_layers)

    results: Dict[str, Dict] = {}
    all_corrs = []

    for name in target_layers:
        feat = feats.get(name)
        if feat is None or feat.shape[1] < 4:
            continue

        l1 = _minmax(l1_scores.get(name, np.zeros(feat.shape[1])))
        taylor = _minmax(taylor_scores.get(name, np.zeros(feat.shape[1])))
        ci = _minmax(compute_chip_ci(feat))
        gra_red = _minmax(compute_gra_redundancy(feat))
        gra_imp = 1.0 - gra_red

        c_chip_gra = _spearman_corr(ci, gra_imp)
        c_chip_l1 = _spearman_corr(ci, l1)
        c_chip_taylor = _spearman_corr(ci, taylor)
        c_gra_l1 = _spearman_corr(gra_imp, l1)
        c_gra_taylor = _spearman_corr(gra_imp, taylor)
        c_l1_taylor = _spearman_corr(l1, taylor)

        results[name] = {
            "n_ch": feat.shape[1],
            "chip_vs_gra": c_chip_gra,
            "chip_vs_l1": c_chip_l1,
            "chip_vs_taylor": c_chip_taylor,
            "gra_vs_l1": c_gra_l1,
            "gra_vs_taylor": c_gra_taylor,
            "l1_vs_taylor": c_l1_taylor,
        }
        all_corrs.append(c_chip_gra)

    mean_chip_gra = float(np.mean(all_corrs)) if all_corrs else 0.0
    print(f"[DIAG] mean Spearman(CHIP, GRA_imp) = {mean_chip_gra:.3f}")
    print("[DIAG] If < 0.7, the signals are complementary (good)")
    print("[DIAG] If > 0.9, the signals are redundant (bad)")

    return {
        "per_layer": results,
        "mean_chip_gra_corr": mean_chip_gra,
        "signal_complementary": mean_chip_gra < 0.7,
    }


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Public experiment runner for GRA-CNN and baseline structured pruning methods")
    parser.add_argument("--arch", required=True, choices=["resnet56", "resnet110", "resnet18", "vgg16"])
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--data-dir", type=str, default="data/tiny-imagenet-200")
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--target_ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_scoring_batches", type=int, default=8)
    parser.add_argument("--gra-rho", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--scoring-batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_loader_workers())
    parser.add_argument("--scoring-workers", type=int, default=default_scoring_workers())
    parser.add_argument("--hrank_batches", type=int, default=5)
    parser.add_argument("--fpgm_metric", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--diagnose", action="store_true",
                        help="Run signal diagnostic before scoring")
    args = parser.parse_args()

    set_seed(args.seed)
    device = DEVICE
    train_batch_size = int(args.batch_size) if int(args.batch_size) > 0 else default_train_batch_size(args.dataset)
    scoring_batch_size = (
        int(args.scoring_batch_size)
        if int(args.scoring_batch_size) > 0
        else default_scoring_batch_size(args.dataset)
    )
    print(f"[CHIP-worker] {args.arch}/{args.dataset}/{args.method} "
          f"r={args.target_ratio} seed={args.seed} epochs={args.epochs}")
    print(f"[CHIP-worker] device={device}")
    print(f"[CHIP-worker] loader_workers={args.workers} scoring_workers={args.scoring_workers}")
    print(f"[CHIP-worker] train_batch_size={train_batch_size} scoring_batch_size={scoring_batch_size}")

    train_ld, test_ld, num_classes = get_dataloaders(
        args.dataset,
        batch_size=train_batch_size,
        workers=args.workers,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    scoring_ld = get_scoring_dataloader(
        args.dataset,
        batch_size=scoring_batch_size,
        workers=args.scoring_workers,
        seed=42,
        data_dir=args.data_dir,
    )

    model = build_model(args.arch, num_classes)
    load_checkpoint(model, args.arch, args.dataset)
    model.to(device)

    t0 = time.time()
    baseline_acc = evaluate(model, test_ld, device)
    print(f"[CHIP-worker] baseline_acc={baseline_acc:.2f}%")

    target_layers = get_target_layers(model)
    print(f"[CHIP-worker] target_layers={len(target_layers)}")

    # Optional signal diagnostic
    diag_result = None
    if args.diagnose:
        diag_result = run_signal_diagnostic(model, scoring_ld, device, target_layers)

    # Score
    scores, score_meta = compute_scores(
        model, args.method, scoring_ld, device, target_layers, args.num_scoring_batches, args.target_ratio, args.arch,
        eval_loader=test_ld, hrank_batches=args.hrank_batches, fpgm_metric=args.fpgm_metric,
        gra_rho=args.gra_rho,
    )
    score_time = score_meta.get("__global__", {}).get("score_time_s", 0.0)
    print(f"[CHIP-worker] scoring done in {score_time:.1f}s")

    # Prune
    masks = scores_to_masks(scores, args.target_ratio, target_layers)
    pruned_model = apply_pruning(model, masks, args.arch, num_classes, device)

    pruned_acc = evaluate(pruned_model, test_ld, device)
    print(f"[CHIP-worker] pruned_acc={pruned_acc:.2f}%")

    # Finetune
    final_acc = finetune(pruned_model, train_ld, test_ld, device, epochs=args.epochs)
    print(f"[CHIP-worker] final_acc={final_acc:.2f}%")
    final_detail = evaluate_detailed(pruned_model, test_ld, device, num_classes)
    print(
        "[CHIP-worker] detail "
        f"macro_f1={final_detail['macro_f1']:.2f}% "
        f"worst_class={final_detail['worst_class_acc']:.2f}% "
        f"collapse<5%={final_detail['collapse_class_count_5pct']}"
    )

    elapsed = time.time() - t0
    params_before = sum(p.numel() for p in model.parameters())
    params_after = sum(p.numel() for p in pruned_model.parameters())

    result = {
        "architecture": args.arch,
        "dataset": args.dataset,
        "method": args.method,
        "mode": "oneshot",
        "target_ratio": float(args.target_ratio),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "num_scoring_batches": int(args.num_scoring_batches),
        "batch_size": int(train_batch_size),
        "scoring_batch_size": int(scoring_batch_size),
        "baseline_acc": float(baseline_acc),
        "pruned_acc": float(pruned_acc),
        "final_acc": float(final_acc),
        "macro_f1": float(final_detail["macro_f1"]),
        "balanced_acc": float(final_detail["balanced_acc"]),
        "worst_class_acc": float(final_detail["worst_class_acc"]),
        "worst10_mean_acc": float(final_detail["worst10_mean_acc"]),
        "collapse_class_count_5pct": int(final_detail["collapse_class_count_5pct"]),
        "collapse_class_count_10pct": int(final_detail["collapse_class_count_10pct"]),
        "params_before": int(params_before),
        "params_after": int(params_after),
        "param_reduction_pct": float(100.0 * (1.0 - params_after / max(params_before, 1))),
        "compression_ratio": float(params_before / max(params_after, 1)),
        "score_time_s": float(score_time),
        "total_time_s": float(round(elapsed, 1)),
        "score_meta": score_meta.get("__global__", {}),
        "method_version": score_meta.get("__global__", {}).get("version", args.method),
        "gate_on_rate": float(score_meta.get("__global__", {}).get("gate_on_rate", 0.0)),
        "semantic_applied_rate": float(score_meta.get("__global__", {}).get("semantic_applied_rate", 0.0)),
        "total_swaps": int(score_meta.get("__global__", {}).get("total_swaps", 0)),
        "avg_swaps_per_layer": float(score_meta.get("__global__", {}).get("avg_swaps_per_layer", 0.0)),
        "tau_candidate": float(score_meta.get("__global__", {}).get("tau_candidate", 0.0)),
        "tau_dyn": float(score_meta.get("__global__", {}).get("tau_dyn", 0.0)),
        "quality_mean": float(score_meta.get("__global__", {}).get("quality_mean", 0.0)),
        "anchor_weights": score_meta.get("__global__", {}).get("anchor_weights", {}),
        "boundary_quantiles": score_meta.get("__global__", {}).get("boundary_quantiles", []),
        "risk_guard_rate": float(score_meta.get("__global__", {}).get("risk_guard_rate", 0.0)),
        "risk_protect_ratio": float(score_meta.get("__global__", {}).get("risk_protect_ratio", 0.0)),
        "risk_tol": float(score_meta.get("__global__", {}).get("risk_tol", 0.0)),
        "hard_class_count": int(score_meta.get("__global__", {}).get("hard_class_count", 0)),
        "hard_class_ids": score_meta.get("__global__", {}).get("hard_class_ids", []),
        "hard_class_mean_acc": float(score_meta.get("__global__", {}).get("hard_class_mean_acc", 0.0)),
        "lambda_severity": float(score_meta.get("__global__", {}).get("lambda_severity", 0.0)),
        "protected_channels_total": int(score_meta.get("__global__", {}).get("total_protected", 0)),
        "protected_channels_ratio": float(score_meta.get("__global__", {}).get("protect_ratio", 0.0)),
        "protected_channels_ratio_per_layer": score_meta.get("__global__", {}).get(
            "protected_channels_ratio_per_layer", {}
        ),
        "budget_mode": score_meta.get("__global__", {}).get("budget_mode", ""),
        "iso_flops_target": None,
        "iso_flops_actual": None,
        "iso_flops_ratio": None,
        "timestamp": datetime.now().isoformat(),
    }
    if diag_result is not None:
        result["signal_diagnostic"] = {
            "mean_chip_gra_corr": diag_result["mean_chip_gra_corr"],
            "signal_complementary": diag_result["signal_complementary"],
        }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fname = (f"{args.arch}_{args.dataset}_{args.method}"
             f"_r{args.target_ratio}_s{args.seed}.json")
    out_path = RESULT_DIR / fname
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[CHIP-worker] result -> {out_path}")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
