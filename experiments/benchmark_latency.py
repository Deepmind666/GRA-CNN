"""Latency benchmark utility for the public GRA-CNN research release."""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_chip_worker as worker

OUTPUT_PATH = ROOT / "results" / "latency_benchmark.csv"
DEVICE = worker.DEVICE

CONFIGS = [
    {"arch": "resnet56", "dataset": "cifar100", "method": "unpruned", "ratio": None, "seed": None},
    {"arch": "resnet56", "dataset": "cifar100", "method": "GRA-CNN", "ratio": 0.7, "seed": 42},
    {"arch": "resnet56", "dataset": "cifar100", "method": "GRA-CNN", "ratio": 0.9, "seed": 42},
    {"arch": "vgg16", "dataset": "cifar100", "method": "unpruned", "ratio": None, "seed": None},
    {"arch": "vgg16", "dataset": "cifar100", "method": "GRA-CNN", "ratio": 0.7, "seed": 42},
    {"arch": "vgg16", "dataset": "cifar100", "method": "GRA-CNN", "ratio": 0.9, "seed": 42},
    {"arch": "resnet18", "dataset": "tinyimagenet", "method": "unpruned", "ratio": None, "seed": None},
    {"arch": "resnet18", "dataset": "tinyimagenet", "method": "GRA-CNN", "ratio": 0.7, "seed": 42},
]


def _num_classes(dataset: str) -> int:
    if dataset == "cifar100":
        return 100
    if dataset == "tinyimagenet":
        return 200
    raise ValueError(f"unsupported dataset: {dataset}")


def _input_shape(dataset: str, batch_size: int) -> Tuple[int, int, int, int]:
    side = 64 if dataset == "tinyimagenet" else 32
    return (batch_size, 3, side, side)


def _device_info() -> Tuple[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        name, driver = [part.strip() for part in result.stdout.strip().split(",", 1)]
        return name, driver
    except Exception:
        return str(DEVICE), "unknown"


def _load_baseline_model(arch: str, dataset: str) -> torch.nn.Module:
    num_classes = _num_classes(dataset)
    model = worker.build_model(arch, num_classes)
    worker.load_checkpoint(model, arch, dataset)
    return model.to(DEVICE).eval()


def _build_pruned_model(arch: str, dataset: str, ratio: float, seed: int) -> torch.nn.Module:
    num_classes = _num_classes(dataset)
    model = worker.build_model(arch, num_classes)
    worker.load_checkpoint(model, arch, dataset)
    model = model.to(DEVICE).eval()
    target_layers = worker.get_target_layers(model)
    scoring_loader = worker.get_scoring_dataloader(
        dataset,
        seed=seed,
        data_dir=str(ROOT / "data" / "tiny-imagenet-200"),
    )
    scores, _ = worker.compute_scores(
        model=model,
        method="GRA-CNN",
        scoring_loader=scoring_loader,
        device=DEVICE,
        target_layers=target_layers,
        num_batches=8,
        pruning_ratio=ratio,
        architecture=arch,
        hrank_batches=5,
        fpgm_metric="l2",
        gra_rho=0.5,
    )
    masks = worker.scores_to_masks(scores, ratio, target_layers)
    pruned = worker.apply_pruning(model, masks, arch, num_classes, DEVICE)
    return pruned.to(DEVICE).eval()


def _measure_latency_ms(model: torch.nn.Module, dataset: str, warmup: int, iterations: int) -> Tuple[float, float]:
    dummy = torch.randn(*_input_shape(dataset, 1), device=DEVICE)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
        torch.cuda.synchronize()
        samples: List[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy)
            end.record()
            torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return mean(samples), pstdev(samples)


def _measure_throughput(model: torch.nn.Module, dataset: str, warmup: int, iterations: int) -> float:
    batch_size = 64
    dummy = torch.randn(*_input_shape(dataset, batch_size), device=DEVICE)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            _ = model(dummy)
        end.record()
        torch.cuda.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0
    return (batch_size * iterations) / max(elapsed_s, 1e-12)


def _flops_and_params(model: torch.nn.Module, dataset: str) -> Tuple[float, float]:
    dummy = torch.randn(*_input_shape(dataset, 1), device=DEVICE)
    with torch.inference_mode():
        flops, params = profile(model, inputs=(dummy,), verbose=False)
    return float(flops) / 1e9, float(params) / 1e6


def benchmark() -> List[Dict[str, object]]:
    torch.backends.cudnn.benchmark = True
    device_name, driver_version = _device_info()
    rows: List[Dict[str, object]] = []

    for index, config in enumerate(CONFIGS, start=1):
        arch = str(config["arch"])
        dataset = str(config["dataset"])
        method = str(config["method"])
        ratio = config["ratio"]
        seed = config["seed"]
        print(f"[{index}/{len(CONFIGS)}] benchmark {arch}/{dataset}/{method} ratio={ratio}", flush=True)
        if method == "unpruned":
            model = _load_baseline_model(arch, dataset)
        else:
            model = _build_pruned_model(arch, dataset, float(ratio), int(seed))

        flops_g, params_m = _flops_and_params(model, dataset)
        latency_ms, latency_std_ms = _measure_latency_ms(model, dataset, warmup=50, iterations=200)
        throughput = _measure_throughput(model, dataset, warmup=50, iterations=100)
        rows.append(
            {
                "arch": arch,
                "dataset": dataset,
                "method": method,
                "ratio": "" if ratio is None else f"{float(ratio):.1f}",
                "seed": "" if seed is None else str(int(seed)),
                "params_M": f"{params_m:.6f}",
                "flops_G": f"{flops_g:.6f}",
                "latency_bs1_ms": f"{latency_ms:.6f}",
                "latency_bs1_std_ms": f"{latency_std_ms:.6f}",
                "throughput_bs64_img_s": f"{throughput:.6f}",
                "device_name": device_name,
                "driver_version": driver_version,
                "timestamp": datetime.now().isoformat(),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arch",
        "dataset",
        "method",
        "ratio",
        "seed",
        "params_M",
        "flops_G",
        "latency_bs1_ms",
        "latency_bs1_std_ms",
        "throughput_bs64_img_s",
        "device_name",
        "driver_version",
        "timestamp",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = benchmark()
    write_csv(rows)
    print(f"WROTE {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
