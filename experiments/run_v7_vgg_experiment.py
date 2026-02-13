"""GRA v7.0 VGG-16 experiment runner.

Phase A: smoke test (1 task) — vgg16 cifar10 GRA-v7 r=0.5 s=42
Phase B: vgg16 cifar10, GRA-v7/L1/FPGM, 3 ratios x 3 seeds (27 tasks)
Phase C: vgg16 cifar100, GRA-v7/L1, r=0.5, 3 seeds (6 tasks)

Total: 34 tasks, estimated ~7-8 hours on RTX 5090.

Usage:
  python experiments/run_v7_vgg_experiment.py
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime

RESULT_DIR = Path(__file__).resolve().parent / "unified_results"
CHECKPOINT_FILE = RESULT_DIR / "v7_vgg_experiment_checkpoint.json"
PYTHON = sys.executable

SEEDS_3 = [42, 123, 456]
RATIOS = [0.3, 0.5, 0.7]


def _tasks(arch, dataset, methods, ratios, seeds):
    return [
        {"arch": arch, "dataset": dataset, "method": m, "ratio": r, "seed": s}
        for m in methods for r in ratios for s in seeds
    ]


PHASE_A = [{"arch": "vgg16", "dataset": "cifar10", "method": "GRA-v7",
            "ratio": 0.5, "seed": 42}]

PHASE_B = _tasks("vgg16", "cifar10", ["GRA-v7", "L1", "FPGM"], RATIOS, SEEDS_3)

PHASE_C = _tasks("vgg16", "cifar100", ["GRA-v7", "L1"], [0.5], SEEDS_3)

ALL_PHASES = [
    ("A_smoke", PHASE_A),
    ("B_vgg16_cifar10", PHASE_B),
    ("C_vgg16_cifar100_core", PHASE_C),
]


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "start_time": None}


def save_checkpoint(ckpt):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


def task_key(t):
    return f"{t['arch']}_{t['dataset']}_{t['method']}_r{t['ratio']}_s{t['seed']}"


def result_exists(t):
    tag = f"{t['arch']}_{t['dataset']}_{t['method']}_r{t['ratio']}_s{t['seed']}"
    return (RESULT_DIR / f"{tag}.json").exists()


def run_task(t, task_idx, total_tasks, ckpt):
    key = task_key(t)
    if key in ckpt["completed"]:
        print(f"  [{task_idx}/{total_tasks}] SKIP (already completed): {key}")
        return True

    if result_exists(t):
        print(f"  [{task_idx}/{total_tasks}] SKIP (result file exists): {key}")
        ckpt["completed"].append(key)
        save_checkpoint(ckpt)
        return True

    cmd = [
        PYTHON, "experiments/unified_worker.py",
        "--arch", t["arch"],
        "--dataset", t["dataset"],
        "--method", t["method"],
        "--ratio", str(t["ratio"]),
        "--seed", str(t["seed"]),
    ]

    print(f"  [{task_idx}/{total_tasks}] RUN: {key}", flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        print(f"    OK: final_acc={data.get('final_acc', '?')} "
                              f"time={elapsed:.0f}s", flush=True)
                    except json.JSONDecodeError:
                        pass
            ckpt["completed"].append(key)
            save_checkpoint(ckpt)
            return True
        else:
            print(f"    FAIL (exit={result.returncode}): {result.stderr[-500:]}", flush=True)
            ckpt["failed"].append({"key": key, "error": result.stderr[-500:]})
            save_checkpoint(ckpt)
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"    TIMEOUT after {elapsed:.0f}s", flush=True)
        ckpt["failed"].append({"key": key, "error": "timeout"})
        save_checkpoint(ckpt)
        return False


def main():
    ckpt = load_checkpoint()
    if ckpt["start_time"] is None:
        ckpt["start_time"] = datetime.now().isoformat()
        save_checkpoint(ckpt)

    all_tasks = []
    for phase_name, tasks in ALL_PHASES:
        for t in tasks:
            all_tasks.append((phase_name, t))

    total = len(all_tasks)
    done_count = len(ckpt["completed"])

    # === RULES.md §5 hard constraint: print ETA basis at start ===
    print(f"={'=' * 70}")
    print(f"GRA v7.0 VGG-16 Experiment Runner")
    print(f"Total tasks: {total}")
    print(f"Already completed: {done_count}")
    print(f"Remaining: {total - done_count}")
    print(f"estimated_total_time: ~7-8 hours")
    print(f"avg_task_time_sec: vgg16_cifar10=900, vgg16_cifar100=960")
    print(f"eta_basis: VGG-16 CIFAR finetune 40ep ~12-15min/task (RTX 5090)")
    print(f"={'=' * 70}")

    task_times = []
    for idx, (phase_name, t) in enumerate(all_tasks, 1):
        key = task_key(t)
        if key in ckpt["completed"]:
            continue

        # Phase header
        if idx == 1 or all_tasks[idx - 2][0] != phase_name:
            print(f"\n{'=' * 60}")
            print(f"Phase: {phase_name}")
            print(f"{'=' * 60}")

        t0 = time.time()
        success = run_task(t, idx, total, ckpt)
        elapsed = time.time() - t0
        task_times.append(elapsed)

        # Progress + ETA
        done_now = len(ckpt["completed"])
        remaining = total - done_now
        if task_times:
            avg_time = sum(task_times) / len(task_times)
            eta_min = remaining * avg_time / 60
            print(f"  Progress: [{done_now}/{total}] "
                  f"avg={avg_time:.0f}s/task "
                  f"ETA={eta_min:.0f}min ({eta_min / 60:.1f}h)", flush=True)

        # Phase A smoke check: abort if failed
        if phase_name == "A_smoke" and not success:
            print("\n[ABORT] Smoke test failed! Fix the issue before running full experiment.")
            sys.exit(1)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Completed: {len(ckpt['completed'])}/{total}")
    print(f"Failed: {len(ckpt['failed'])}")
    if ckpt["failed"]:
        for f in ckpt["failed"]:
            print(f"  FAILED: {f['key']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
