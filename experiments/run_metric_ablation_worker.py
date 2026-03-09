"""Public worker for the semantic-metric sensitivity experiment.

This script keeps the class-aware refinement pipeline fixed and replaces only
the semantic similarity metric with cosine similarity or Pearson correlation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_chip_worker as worker
from pruning.core_algorithm_metric_ablation import compute_metric_ablation_scores

RESULT_DIR = ROOT / "results"
LOCK_DIR = RESULT_DIR / "locks"

CHECKPOINTS = {
    "resnet56": {
        "cifar100": "checkpoints/baseline_cifar100_resnet56.pth",
    },
    "vgg16": {
        "cifar100": "checkpoints/baseline_cifar100_vgg16.pth",
    },
}

METHODS = ["Cosine-Semantic", "Correlation-Semantic"]
METRIC_MAP = {
    "Cosine-Semantic": "cosine",
    "Correlation-Semantic": "correlation",
}


def build_model(arch: str, num_classes: int = 100):
    if arch == "resnet56":
        from models.resnet_cifar import resnet56

        return resnet56(num_classes=num_classes)
    if arch == "vgg16":
        from models.vgg_cifar import vgg16

        return vgg16(num_classes=num_classes)
    raise ValueError(f"Unsupported arch for metric ablation: {arch}")


def apply_structural_pruning(model, scores, ratio: float, arch: str, num_classes: int, device):
    target_layers = worker.get_target_layers(model)
    masks = worker.scores_to_masks(scores, ratio, target_layers)
    pruned = worker.apply_pruning(model, masks, arch, num_classes, device)
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser(description="Public metric-sensitivity runner for GRA-CNN")
    parser.add_argument("--arch", required=True, choices=["resnet56", "vgg16"])
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--finetune_epochs", type=int, default=40)
    parser.add_argument("--num_scoring_batches", type=int, default=12)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    tag = f"{args.arch}_{args.dataset}_{args.method}_r{args.ratio}_s{args.seed}"
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / f"{tag}.lock"
    result_path = RESULT_DIR / f"{tag}.json"
    lock_fd = None

    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if data.get("final_acc") not in (None, "", 0):
                print(json.dumps({"task": tag, "status": "skipped_exists"}), flush=True)
                return
        except Exception:
            pass

    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode("utf-8"))
    except FileExistsError:
        print(json.dumps({"task": tag, "status": "skipped_locked"}), flush=True)
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        worker.set_seed(args.seed)
        num_classes = 100

        model = build_model(args.arch, num_classes)
        ckpt_path = ROOT / CHECKPOINTS[args.arch][args.dataset]
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device)

        train_loader, test_loader, _ = worker.get_dataloaders(args.dataset, seed=args.seed, data_dir=args.data_dir)
        scoring_loader = worker.get_scoring_dataloader(args.dataset, seed=42, data_dir=args.data_dir)

        t0 = time.time()
        baseline_acc = worker.evaluate(model, test_loader, device)

        metric = METRIC_MAP[args.method]
        scores, scoring_meta = compute_metric_ablation_scores(
            model,
            scoring_loader,
            device,
            num_batches=args.num_scoring_batches,
            pruning_ratio=args.ratio,
            verbose=True,
            metric=metric,
        )

        pruned = apply_structural_pruning(model, scores, args.ratio, args.arch, num_classes, device)
        pruned_acc = worker.evaluate(pruned, test_loader, device)
        final_acc = worker.finetune(pruned, train_loader, test_loader, device, epochs=args.finetune_epochs)
        if args.finetune_epochs < 10 and final_acc <= 0:
            final_acc = worker.evaluate(pruned, test_loader, device)

        elapsed = time.time() - t0
        params_before = sum(p.numel() for p in model.parameters())
        params_after = sum(p.numel() for p in pruned.parameters())
        result = {
            "architecture": args.arch,
            "dataset": args.dataset,
            "method": args.method,
            "target_ratio": args.ratio,
            "seed": args.seed,
            "baseline_acc": round(baseline_acc, 2),
            "pruned_acc": round(pruned_acc, 2),
            "final_acc": round(final_acc, 2),
            "params_before": params_before,
            "params_after": params_after,
            "compression_ratio": round(params_before / max(params_after, 1), 4),
            "total_time_s": round(elapsed, 1),
            "finetune_epochs": int(args.finetune_epochs),
            "scoring_metric": metric,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        meta_path = RESULT_DIR / f"{tag}_meta.json"

        def _serialize(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_serialize(v) for v in obj]
            return obj

        try:
            meta_path.write_text(json.dumps(_serialize(scoring_meta), indent=2), encoding="utf-8")
        except Exception:
            pass

        print(json.dumps(result), flush=True)
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except Exception:
                pass
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
