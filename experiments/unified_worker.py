"""Unified experiment worker: single task for the authoritative result matrix.

Supports: L1, FPGM, GRA-v5.2, GRA-v6, GRA-v7, GRA-v5.51 (with A/B/C ablation flags).
All methods share the same checkpoint, transforms, seeds, and finetune config.

Usage:
  python experiments/unified_worker.py \
    --arch resnet56 --method GRA-v5.51-ABC --ratio 0.7 --seed 42
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "unified_results"

# Authoritative checkpoint map — dataset-aware
CHECKPOINTS = {
    "resnet20": {
        "cifar10": "checkpoints/resnet20_cifar10_retrained.pth",
    },
    "resnet56": {
        "cifar10": "checkpoints/resnet56_best_new.pth",
    },
    "vgg16": {
        "cifar10": "experiments/baseline_cifar10_vgg16.pth",
        "cifar100": "experiments/baseline_cifar100_vgg16.pth",
    },
}

METHODS = [
    "L1", "FPGM",
    "GRA-v5.2",
    "GRA-v6",
    "GRA-v7",
    "GRA-v5.51-ABC",  # all three changes
    "GRA-v5.51-A",    # only rank+zscore norm
    "GRA-v5.51-B",    # only continuous weights
    "GRA-v5.51-C",    # only reliability fallback
    "GRA-v5.51-none", # v5.2 logic via v5.51 code (all flags off)
]


def build_model(arch, num_classes=10):
    if arch == "resnet20":
        from models.resnet_cifar import resnet20
        return resnet20(num_classes=num_classes)
    if arch == "resnet56":
        from models.resnet_cifar import resnet56
        return resnet56(num_classes=num_classes)
    if arch == "vgg16":
        from models.vgg_cifar import vgg16
        return vgg16(num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def compute_scores(model, method, scoring_loader, device, ratio):
    """Compute pruning scores for the given method."""
    if method == "L1":
        from experiments.run_full_matrix_v4 import compute_l1_scores
        return compute_l1_scores(model), None
    if method == "FPGM":
        from experiments.run_full_matrix_v4 import compute_fpgm_scores
        return compute_fpgm_scores(model), None
    if method == "GRA-v5.2":
        from pruning.core_algorithm_v5_improved import compute_gra_v5_scores
        scores, meta = compute_gra_v5_scores(
            model, scoring_loader, device,
            num_batches=20, pruning_ratio=ratio, verbose=False)
        return scores, meta
    if method == "GRA-v6":
        from pruning.core_algorithm_v6_c2srp import compute_gra_v6_scores
        scores, meta = compute_gra_v6_scores(
            model, scoring_loader, device,
            num_batches=12, pruning_ratio=ratio, verbose=False,
            enable_knockout_calibration=False,
            knockout_batches=2,
            knockout_channels_per_layer=8,
            random_seed=42)
        return scores, meta
    if method == "GRA-v7":
        from pruning.core_algorithm_v7 import compute_gra_v7_scores
        scores, meta = compute_gra_v7_scores(
            model, scoring_loader, device,
            num_batches=12, pruning_ratio=ratio, verbose=False)
        return scores, meta
    # GRA-v5.51 variants
    from pruning.core_algorithm_v551 import compute_gra_v551_scores
    flags = method.replace("GRA-v5.51-", "")
    use_a = "A" in flags
    use_b = "B" in flags
    use_c = "C" in flags
    if flags == "none":
        use_a = use_b = use_c = False
    scores, meta = compute_gra_v551_scores(
        model, scoring_loader, device,
        num_batches=20, pruning_ratio=ratio, verbose=False,
        use_rank_zscore=use_a,
        use_continuous_weights=use_b,
        use_reliability_fallback=use_c)
    return scores, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["resnet20", "resnet56", "vgg16"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    args = parser.parse_args()

    from experiments.run_full_matrix_v4 import (
        apply_structural_pruning, get_dataloaders,
        get_scoring_dataloader, set_seed, evaluate, finetune,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    num_classes = 100 if args.dataset == "cifar100" else 10

    # Load model + checkpoint
    model = build_model(args.arch, num_classes)
    ckpt = CHECKPOINTS[args.arch][args.dataset]
    sd = torch.load(ckpt, map_location="cpu")
    # Handle both raw state_dict and wrapped format
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)

    train_loader, test_loader, _ = get_dataloaders(args.dataset, seed=args.seed)
    scoring_loader = get_scoring_dataloader(args.dataset, seed=42)

    t0 = time.time()

    # Baseline
    baseline_acc = evaluate(model, test_loader, device)

    # Score + prune
    scores, scoring_meta = compute_scores(model, args.method, scoring_loader, device, args.ratio)
    # VGG uses full-network pruning; ResNet uses stage_mid_only
    pruning_scope = "stage_mid_only" if "resnet" in args.arch else None
    pruned, _ = apply_structural_pruning(
        model, scores, args.ratio, args.arch, num_classes, device, pruning_scope)
    pruned_acc = evaluate(pruned, test_loader, device)

    # Finetune (40 epochs, same for all methods)
    final_acc = finetune(pruned, train_loader, test_loader, device, epochs=40)

    elapsed = time.time() - t0
    params_before = sum(p.numel() for p in model.parameters())
    params_after = sum(p.numel() for p in pruned.parameters())

    result = {
        "architecture": args.arch,
        "dataset": args.dataset,
        "method": args.method,
        "ratio": args.ratio,
        "seed": args.seed,
        "baseline_acc": round(baseline_acc, 2),
        "pruned_acc": round(pruned_acc, 2),
        "final_acc": round(final_acc, 2),
        "params_before": params_before,
        "params_after": params_after,
        "compression_ratio": round(params_before / max(params_after, 1), 4),
        "time_s": round(elapsed, 1),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arch}_{args.dataset}_{args.method}_r{args.ratio}_s{args.seed}"
    with open(RESULT_DIR / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save scoring metadata sidecar (for reproducibility and paper auditing)
    if scoring_meta is not None:
        meta_path = RESULT_DIR / f"{tag}_meta.json"
        # Convert numpy/tensor values to JSON-serializable types
        def _serialize(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(i) for i in obj]
            return obj
        try:
            with open(meta_path, "w") as f:
                json.dump(_serialize(scoring_meta), f, indent=2)
        except Exception:
            pass  # metadata is best-effort, never block the result

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
