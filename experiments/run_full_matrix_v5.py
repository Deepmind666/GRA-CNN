from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiments.run_full_matrix_v4 import (
    CHECKPOINTS,
    apply_structural_pruning,
    compute_fpgm_scores,
    compute_gra_v1_scores,
    compute_gra_v4_scores,
    compute_hrank_scores,
    compute_l1_scores,
    compute_random_scores,
    compute_taylor_scores,
    evaluate,
    finetune,
    get_dataloaders,
    get_scoring_dataloader,
    set_seed,
)
from pruning.core_algorithm_v5_improved import compute_gra_v5_scores, GRA_VERSION as GRA_V5_VERSION


def _build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet56":
        from models.resnet_cifar import resnet56
        return resnet56(num_classes=num_classes)
    if arch == "resnet20":
        from models.resnet_cifar import resnet20
        return resnet20(num_classes=num_classes)
    if arch == "resnet110":
        from models.resnet_cifar import resnet110
        return resnet110(num_classes=num_classes)
    if arch == "vgg16":
        from models.vgg_cifar import vgg16_bn
        return vgg16_bn(num_classes=num_classes)
    if arch == "mobilenetv2":
        from models.mobilenetv2 import mobilenetv2_cifar
        return mobilenetv2_cifar(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def _load_checkpoint(model: nn.Module, arch: str) -> None:
    ckpt = CHECKPOINTS.get(arch)
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found for {arch}: {ckpt}")

    sd = torch.load(ckpt, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)


def _compute_scores(
    model: nn.Module,
    method: str,
    scoring_loader,
    device: torch.device,
    ratio: float,
) -> Tuple[Dict[str, np.ndarray], str]:
    if method == "L1":
        return compute_l1_scores(model), "none"
    if method == "FPGM":
        return compute_fpgm_scores(model), "none"
    if method == "HRank":
        return compute_hrank_scores(model, scoring_loader, device, num_batches=5), "none"
    if method == "Taylor":
        return compute_taylor_scores(model, scoring_loader, device), "none"
    if method == "GRA-v1":
        return compute_gra_v1_scores(model, scoring_loader, device), "1.0"
    if method == "GRA-v4":
        return compute_gra_v4_scores(model, scoring_loader, device, full_fusion=True, pruning_ratio=ratio), "4.0"
    if method == "GRA-v5":
        scores, metadata = compute_gra_v5_scores(
            model,
            scoring_loader,
            device,
            num_batches=20,
            pruning_ratio=ratio,
            verbose=False,
        )
        return scores, metadata.get("version", GRA_V5_VERSION)
    if method == "Random":
        return compute_random_scores(model), "none"

    raise ValueError(f"Unknown method: {method}")


def run_single_experiment_v5(
    arch: str,
    method: str,
    ratio: float,
    seed: int,
    device: torch.device,
    pruning_scope: str = "stage_mid_only",
    dataset: str = "cifar10",
    finetune_epochs: int = 40,
) -> Dict[str, Any]:
    set_seed(seed)

    train_loader, test_loader, num_classes = get_dataloaders(dataset, seed=seed)
    scoring_loader = get_scoring_dataloader(dataset, seed=42)

    model = _build_model(arch, num_classes)
    _load_checkpoint(model, arch)
    model.to(device)

    baseline = evaluate(model, test_loader, device)

    scores, gra_version = _compute_scores(model, method, scoring_loader, device, ratio)

    pruned_model, _mask_dict = apply_structural_pruning(
        model=model,
        scores=scores,
        ratio=ratio,
        arch=arch,
        num_classes=num_classes,
        device=device,
        pruning_scope=pruning_scope,
    )

    pruned_acc = evaluate(pruned_model, test_loader, device)
    final_acc = finetune(
        pruned_model,
        train_loader,
        test_loader,
        device,
        epochs=finetune_epochs,
    )
    # Guard: legacy finetune() only updates best_acc every 10 epochs.
    # For short smoke runs (epochs < 10), return a real measured accuracy.
    if finetune_epochs < 10 and final_acc <= 0.0:
        final_acc = evaluate(pruned_model, test_loader, device)

    params_before = sum(p.numel() for p in model.parameters())
    params_after = sum(p.numel() for p in pruned_model.parameters())

    result = {
        "architecture": arch,
        "dataset": dataset,
        "method": method,
        "ratio": float(ratio),
        "iso_flops": False,
        "seed": int(seed),
        "gra_version": gra_version,
        "baseline_acc": float(baseline),
        "pruned_acc": float(pruned_acc),
        "final_acc": float(final_acc),
        "params_before": int(params_before),
        "params_after": int(params_after),
        "compression_ratio": float(params_before / max(params_after, 1)),
        "pruning_scope": pruning_scope,
        "timestamp": datetime.now().isoformat(),
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one v5 pruning experiment")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pruning_scope", default="stage_mid_only", choices=["resnet_conv1_only", "stage_mid_only"])
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--finetune_epochs", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_single_experiment_v5(
        arch=args.arch,
        method=args.method,
        ratio=args.ratio,
        seed=args.seed,
        device=device,
        pruning_scope=args.pruning_scope,
        dataset=args.dataset,
        finetune_epochs=args.finetune_epochs,
    )
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
