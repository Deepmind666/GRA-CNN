"""Equal-budget CHIP experiment: compare CHIP vs GRA-CHIP-v3 at matched FLOPs.

For each cell (arch/dataset/ratio), we:
  1. Run the REFERENCE method (CHIP) at the nominal ratio → get actual FLOPs
  2. Binary-search the ratio for GRA-CHIP-v3 to match those FLOPs (<=2% error)
  3. Finetune both, record iso_flops metrics

Supported cells:
  - resnet56/cifar100/r=0.7
  - resnet56/cifar100/r=0.9
  - vgg16/cifar100/r=0.7
  - vgg16/cifar100/r=0.9

Usage:
  python experiments/run_chip_equal_budget.py --seed 42
  python experiments/run_chip_equal_budget.py --seed 42 --dry-run   # CPU-only, no finetune
  python experiments/run_chip_equal_budget.py --seed 42 --rewrite-raw --strict-iso
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

# Reuse from chip worker (scoring, pruning, data, model)
from experiments.run_chip_worker import (
    CHECKPOINTS,
    RESULT_DIR as CHIP_RESULT_DIR,
    build_model,
    load_checkpoint,
    get_dataloaders,
    get_scoring_dataloader,
    get_target_layers,
    compute_scores,
    scores_to_masks,
    apply_pruning,
    evaluate,
    finetune,
    set_seed,
)

RESULT_DIR = ROOT / "experiments" / "chip_results"

CELLS = [
    {"arch": "resnet56", "dataset": "cifar100", "ratio": 0.7},
    {"arch": "resnet56", "dataset": "cifar100", "ratio": 0.9},
    {"arch": "vgg16",    "dataset": "cifar100", "ratio": 0.7},
    {"arch": "vgg16",    "dataset": "cifar100", "ratio": 0.9},
]

METHODS = ["CHIP", "GRA-CHIP-v3"]


# ----------------------------------------------------------------
# FLOPs computation
# ----------------------------------------------------------------
def compute_model_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """Compute total FLOPs for conv+linear layers via dummy forward pass.

    Uses hooks to capture actual spatial dimensions per layer.
    Returns total multiply-accumulate operations (MACs) * 2 = FLOPs.
    """
    device = next(model.parameters()).device
    flops_list: List[int] = []
    hooks = []

    def _conv_hook(module: nn.Conv2d, inp, out):
        # FLOPs = 2 * C_out * C_in/groups * Kh * Kw * H_out * W_out
        batch, c_out, h_out, w_out = out.shape
        c_in = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        f = 2 * c_out * (c_in // groups) * kh * kw * h_out * w_out
        flops_list.append(f)

    def _linear_hook(module: nn.Linear, inp, out):
        f = 2 * module.in_features * module.out_features
        flops_list.append(f)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    model.eval()
    with torch.no_grad():
        x = torch.randn(*input_size, device=device)
        model(x)

    for h in hooks:
        h.remove()

    return sum(flops_list)


def find_ratio_for_target_flops(
    model: nn.Module,
    scores: Dict[str, np.ndarray],
    target_flops: int,
    target_layers: List[str],
    arch: str,
    num_classes: int,
    device: torch.device,
    tolerance: float = 0.02,
    max_iter: int = 16,
) -> Tuple[float, int, nn.Module]:
    """Binary search for pruning ratio that yields target FLOPs (within tolerance).

    Returns (best_ratio, actual_flops, pruned_model).
    """
    lo, hi = 0.05, 0.95
    best_ratio = None
    best_flops = None
    best_model = None
    best_err = float("inf")

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        try:
            masks = scores_to_masks(scores, mid, target_layers)
            pruned = apply_pruning(model, masks, arch, num_classes, device)
            actual = compute_model_flops(pruned)
        except Exception:
            if mid > 0.5:
                hi = mid
            else:
                lo = mid
            continue

        err = abs(actual - target_flops) / max(target_flops, 1)
        if err < best_err:
            best_err = err
            best_ratio = mid
            best_flops = actual
            best_model = pruned

        if err <= tolerance:
            break

        if actual > target_flops:
            lo = mid  # need more pruning
        else:
            hi = mid  # need less pruning

    return best_ratio, best_flops, best_model


# ----------------------------------------------------------------
# Single cell runner
# ----------------------------------------------------------------
def run_cell(
    arch: str, dataset: str, ratio: float, seed: int,
    device: torch.device, epochs: int = 30, dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run equal-budget comparison for one cell. Returns list of result dicts."""
    set_seed(seed)
    num_classes = 100 if dataset == "cifar100" else 10
    label = f"{arch}/{dataset}/r{ratio}/s{seed}"

    print(f"\n{'='*60}")
    print(f"[EB] {label}")
    print(f"{'='*60}")

    # Load model
    model = build_model(arch, num_classes)
    load_checkpoint(model, arch, dataset)
    model.to(device)

    train_ld, test_ld, _ = get_dataloaders(dataset, seed=seed)
    scoring_ld = get_scoring_dataloader(dataset, seed=42)
    target_layers = get_target_layers(model)

    baseline_acc = evaluate(model, test_ld, device)
    flops_before = compute_model_flops(model)
    params_before = sum(p.numel() for p in model.parameters())
    print(f"[EB] baseline_acc={baseline_acc:.2f}% flops={flops_before:,} params={params_before:,}")

    results = []

    # --- Phase 1: Run CHIP at nominal ratio (reference) ---
    print(f"\n[EB] Phase 1: CHIP @ r={ratio} (reference)")
    t0 = time.time()
    set_seed(seed)
    ref_scores, ref_meta = compute_scores(
        model, "CHIP", scoring_ld, device, target_layers, 8, ratio)
    ref_score_time = ref_meta.get("__global__", {}).get("score_time_s", 0.0)

    ref_masks = scores_to_masks(ref_scores, ratio, target_layers)
    ref_pruned = apply_pruning(model, ref_masks, arch, num_classes, device)
    ref_flops = compute_model_flops(ref_pruned)
    ref_params = sum(p.numel() for p in ref_pruned.parameters())
    ref_pruned_acc = evaluate(ref_pruned, test_ld, device)

    if dry_run:
        ref_final_acc = ref_pruned_acc
    else:
        ref_final_acc = finetune(ref_pruned, train_ld, test_ld, device, epochs=epochs)
    ref_time = time.time() - t0

    ref_result = {
        "architecture": arch,
        "dataset": dataset,
        "method": "CHIP",
        "role": "reference",
        "nominal_ratio": ratio,
        "actual_ratio": ratio,
        "seed": seed,
        "baseline_acc": round(baseline_acc, 2),
        "pruned_acc": round(ref_pruned_acc, 2),
        "final_acc": round(ref_final_acc, 2),
        "flops_before": flops_before,
        "flops_after": ref_flops,
        "flops_reduction_pct": round(100.0 * (1 - ref_flops / max(flops_before, 1)), 2),
        "params_before": params_before,
        "params_after": ref_params,
        "iso_flops_target": ref_flops,
        "iso_flops_actual": ref_flops,
        "iso_flops_ratio": 1.0,
        "iso_flops_error_pct": 0.0,
        "score_time_s": round(ref_score_time, 2),
        "total_time_s": round(ref_time, 1),
        "epochs": epochs if not dry_run else 0,
        "timestamp": datetime.now().isoformat(),
    }
    results.append(ref_result)
    print(f"[EB] CHIP: flops={ref_flops:,} ({ref_result['flops_reduction_pct']:.1f}% red) "
          f"acc={ref_final_acc:.2f}% time={ref_time:.0f}s")

    # --- Phase 2: GRA-CHIP-v3 matched to CHIP's FLOPs ---
    print(f"\n[EB] Phase 2: GRA-CHIP-v3 → target_flops={ref_flops:,}")
    t0 = time.time()
    set_seed(seed)
    v3_scores, v3_meta = compute_scores(
        model, "GRA-CHIP-v3", scoring_ld, device, target_layers, 8, ratio)
    v3_score_time = v3_meta.get("__global__", {}).get("score_time_s", 0.0)

    found_ratio, v3_flops, v3_pruned = find_ratio_for_target_flops(
        model, v3_scores, ref_flops, target_layers, arch, num_classes, device)

    if v3_pruned is None:
        print(f"[EB] WARNING: binary search failed for GRA-CHIP-v3")
        return results

    v3_params = sum(p.numel() for p in v3_pruned.parameters())
    v3_pruned_acc = evaluate(v3_pruned, test_ld, device)

    if dry_run:
        v3_final_acc = v3_pruned_acc
    else:
        v3_final_acc = finetune(v3_pruned, train_ld, test_ld, device, epochs=epochs)
    v3_time = time.time() - t0

    iso_ratio = v3_flops / max(ref_flops, 1)
    iso_err = abs(iso_ratio - 1.0) * 100

    v3_result = {
        "architecture": arch,
        "dataset": dataset,
        "method": "GRA-CHIP-v3",
        "role": "matched",
        "nominal_ratio": ratio,
        "actual_ratio": round(found_ratio, 4),
        "seed": seed,
        "baseline_acc": round(baseline_acc, 2),
        "pruned_acc": round(v3_pruned_acc, 2),
        "final_acc": round(v3_final_acc, 2),
        "flops_before": flops_before,
        "flops_after": v3_flops,
        "flops_reduction_pct": round(100.0 * (1 - v3_flops / max(flops_before, 1)), 2),
        "params_before": params_before,
        "params_after": v3_params,
        "iso_flops_target": ref_flops,
        "iso_flops_actual": v3_flops,
        "iso_flops_ratio": round(iso_ratio, 4),
        "iso_flops_error_pct": round(iso_err, 2),
        "score_time_s": round(v3_score_time, 2),
        "total_time_s": round(v3_time, 1),
        "epochs": epochs if not dry_run else 0,
        "timestamp": datetime.now().isoformat(),
    }
    results.append(v3_result)
    print(f"[EB] GRA-CHIP-v3: ratio_found={found_ratio:.4f} flops={v3_flops:,} "
          f"iso_err={iso_err:.2f}% acc={v3_final_acc:.2f}% time={v3_time:.0f}s")

    return results


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def _key_of_record(r: Dict[str, Any]) -> Tuple[str, str, float, str, int]:
    return (
        str(r["architecture"]),
        str(r["dataset"]),
        float(r["nominal_ratio"]),
        str(r["method"]),
        int(r["seed"]),
    )


def _validate_iso_record(r: Dict[str, Any]) -> None:
    required = ["iso_flops_target", "iso_flops_actual", "iso_flops_ratio"]
    for k in required:
        if r.get(k, None) in (None, "", "None"):
            raise ValueError(f"missing required iso field: {k}")


def _load_existing_raw(raw_csv: Path) -> Dict[Tuple[str, str, float, str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, str, float, str, int], Dict[str, Any]] = {}
    if not raw_csv.exists():
        return out
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["nominal_ratio"] = float(row["nominal_ratio"])
                row["seed"] = int(row["seed"])
                # Keep numeric fields parseable for downstream analyzers.
                for k in [
                    "baseline_acc",
                    "pruned_acc",
                    "final_acc",
                    "flops_before",
                    "flops_after",
                    "flops_reduction_pct",
                    "params_before",
                    "params_after",
                    "iso_flops_target",
                    "iso_flops_actual",
                    "iso_flops_ratio",
                    "iso_flops_error_pct",
                    "score_time_s",
                    "total_time_s",
                    "actual_ratio",
                ]:
                    if k in row and row[k] not in ("", None):
                        row[k] = float(row[k])
            except Exception:
                continue
            out[_key_of_record(row)] = row
    return out


def _write_raw_upsert(raw_csv: Path, new_rows: List[Dict[str, Any]]) -> int:
    merged = _load_existing_raw(raw_csv)
    for r in new_rows:
        merged[_key_of_record(r)] = r
    if not merged:
        return 0

    fieldnames = list(next(iter(merged.values())).keys())
    stable_rows = sorted(
        merged.values(),
        key=lambda x: (
            str(x["architecture"]),
            str(x["dataset"]),
            float(x["nominal_ratio"]),
            str(x["method"]),
            int(x["seed"]),
        ),
    )
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(stable_rows)
    return len(stable_rows)


def main():
    parser = argparse.ArgumentParser(description="CHIP equal-budget runner")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only, skip finetune (for self-test)")
    parser.add_argument("--cells", type=str, default=None,
                        help="Comma-separated cell indices (0,1,2,3). Default: all")
    parser.add_argument("--rewrite-raw", action="store_true",
                        help="Delete existing equal_budget_raw.csv before this run")
    parser.add_argument("--strict-iso", action="store_true",
                        help="Fail if any record misses iso fields or iso error > 2.0%%")
    parser.add_argument("--avg-task-sec", type=float, default=427.0,
                        help="ETA basis per single (cell,method) task")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    print(f"[EB] device={device} seed={args.seed} epochs={args.epochs} dry_run={args.dry_run}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = RESULT_DIR / "equal_budget_raw.csv"
    if args.rewrite_raw and raw_csv.exists():
        raw_csv.unlink()
        print(f"[EB] rewrite-raw enabled: removed {raw_csv}")

    # Select cells
    if args.cells:
        indices = [int(x) for x in args.cells.split(",")]
        cells = [CELLS[i] for i in indices]
    else:
        cells = CELLS

    est_tasks = len(cells) * len(METHODS)
    est_total = est_tasks * float(args.avg_task_sec)
    print("=" * 64)
    print(f"[EB] estimated_total_time={est_total:.1f}s ({est_total/3600:.2f}h)")
    print(f"[EB] avg_task_time_sec={float(args.avg_task_sec):.1f}")
    print("[EB] eta_basis=historical mean per (cell,method) task")
    print("=" * 64)

    all_results: List[Dict] = []
    had_failure = False
    for cell in cells:
        cell_results = run_cell(
            arch=cell["arch"], dataset=cell["dataset"], ratio=cell["ratio"],
            seed=args.seed, device=device, epochs=args.epochs, dry_run=args.dry_run,
        )
        if len(cell_results) != len(METHODS):
            had_failure = True
            print(f"[EB] WARNING: incomplete cell results for {cell} (got {len(cell_results)})")
        all_results.extend(cell_results)

    # Validate records
    for r in all_results:
        _validate_iso_record(r)
        if args.strict_iso and float(r.get("iso_flops_error_pct", 99.0)) > 2.0:
            raise RuntimeError(
                f"strict-iso violation: {r['architecture']}/{r['dataset']}/"
                f"r{r['nominal_ratio']}/{r['method']}/s{r['seed']} "
                f"iso_err={r['iso_flops_error_pct']}"
            )

    # Write raw CSV via upsert
    if all_results:
        n_rows = _write_raw_upsert(raw_csv, all_results)
        print(f"\n[EB] Raw results upserted to {raw_csv} (rows={n_rows})")

    # Write per-run JSON
    for r in all_results:
        tag = f"eb_{r['architecture']}_{r['dataset']}_{r['method']}_r{r['nominal_ratio']}_s{r['seed']}"
        jpath = RESULT_DIR / f"{tag}.json"
        jpath.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print(f"\n[EB] Done. {len(all_results)} results written.")
    for r in all_results:
        print(f"  {r['method']:16s} {r['architecture']}/{r['dataset']}/r{r['nominal_ratio']} "
              f"acc={r['final_acc']:.2f} iso_err={r['iso_flops_error_pct']:.2f}%")

    if args.strict_iso and had_failure:
        raise RuntimeError("strict-iso enabled and at least one cell is incomplete")


if __name__ == "__main__":
    main()
