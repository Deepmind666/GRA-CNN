"""P1 validation matrix runner for GRA v5."""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = PROJECT_ROOT / "experiments" / "single_task_v5.py"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "experiments" / "stage_mid_pruning" / "results_v5_p1"

lock = threading.Lock()


def parse_csv_list(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_run_log(run_log: Path, message: str) -> None:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {message}\n")


def load_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.exists():
        return set(), []

    with checkpoint_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    completed = set(tuple(x) for x in data.get("completed", []))
    results = data.get("results", [])
    return completed, results


def save_results_csv(results, csv_file: Path):
    if not results:
        return

    fieldnames = [
        "architecture", "dataset", "method", "ratio", "iso_flops", "seed",
        "gra_version", "baseline_acc", "pruned_acc", "final_acc",
        "params_before", "params_after", "compression_ratio",
        "pruning_scope", "timestamp"
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def save_checkpoint(completed, results, checkpoint_path: Path, csv_file: Path):
    payload = {
        "completed": [list(k) for k in sorted(completed)],
        "results": results,
    }
    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_results_csv(results, csv_file)


def parse_result(stdout_text: str):
    for line in stdout_text.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return None


def run_task(task, completed, results, args, run_log: Path, checkpoint_path: Path, csv_file: Path):
    arch, method, ratio, seed = task
    cmd = [
        args.python_exe,
        str(args.single_task_script),
        "--arch", arch,
        "--method", method,
        "--ratio", str(ratio),
        "--seed", str(seed),
        "--pruning_scope", args.pruning_scope,
        "--finetune_epochs", str(args.finetune_epochs),
    ]

    task_start = time.time()
    append_run_log(run_log, f"[START] {arch}|{method}|r={ratio}|s={seed}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
            cwd=str(PROJECT_ROOT),
        )
        elapsed = time.time() - task_start
        result = parse_result(proc.stdout)

        if result is not None and proc.returncode == 0:
            with lock:
                completed.add(task)
                results.append(result)
                save_checkpoint(completed, results, checkpoint_path, csv_file)
            append_run_log(
                run_log,
                f"[DONE] {arch}|{method}|r={ratio}|s={seed} "
                f"final={result.get('final_acc', 'NA')} elapsed_sec={elapsed:.1f}",
            )
            return {"ok": True, "task": task, "result": result, "elapsed_sec": elapsed}

        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-30:])
        append_run_log(
            run_log,
            f"[FAIL] {arch}|{method}|r={ratio}|s={seed} rc={proc.returncode} tail={tail}",
        )
        return {
            "ok": False,
            "task": task,
            "elapsed_sec": elapsed,
            "returncode": proc.returncode,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - task_start
        append_run_log(
            run_log,
            f"[FAIL] {arch}|{method}|r={ratio}|s={seed} timeout={args.timeout_sec} elapsed_sec={elapsed:.1f}",
        )
        return {"ok": False, "task": task, "elapsed_sec": elapsed, "timeout": True}


def format_eta(seconds_left: float) -> str:
    if seconds_left <= 0:
        return "0s"
    return str(timedelta(seconds=int(seconds_left)))


def build_tasks(architectures, methods, ratios, seeds, completed):
    tasks = []
    for arch in architectures:
        for method in methods:
            for ratio in ratios:
                for seed in seeds:
                    key = (arch, method, ratio, seed)
                    if key not in completed:
                        tasks.append(key)
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--single_task_script", default=str(DEFAULT_SCRIPT))
    parser.add_argument("--result_dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--timeout_sec", type=int, default=5400)
    parser.add_argument("--pruning_scope", default="stage_mid_only")
    parser.add_argument("--finetune_epochs", type=int, default=40)
    parser.add_argument("--architectures", default="resnet20,resnet56")
    parser.add_argument("--methods", default="L1,FPGM,GRA-v4,GRA-v5,Random")
    parser.add_argument("--ratios", default="0.3,0.5,0.7")
    parser.add_argument("--seeds", default="42,123,456,789,1024")
    args = parser.parse_args()

    args.single_task_script = Path(args.single_task_script).resolve()
    result_dir = Path(args.result_dir).resolve()
    checkpoint_path = result_dir / "checkpoint.json"
    run_log = result_dir / "run_log.txt"
    csv_file = result_dir / "results.csv"

    architectures = parse_csv_list(args.architectures, str)
    methods = parse_csv_list(args.methods, str)
    ratios = parse_csv_list(args.ratios, float)
    seeds = parse_csv_list(args.seeds, int)

    result_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now()
    start_ts = time.time()
    total_exp = len(architectures) * len(methods) * len(ratios) * len(seeds)

    print("=" * 72)
    print(f"P1-v5 start at {start_dt.isoformat()}")
    print(f"project_root={PROJECT_ROOT}")
    print(f"python_exe={args.python_exe}")
    print(f"single_task_script={args.single_task_script}")
    print(f"result_dir={result_dir}")
    print(f"scope={args.pruning_scope} workers={args.num_workers} total={total_exp}")
    print(f"architectures={architectures}")
    print(f"methods={methods}")
    print(f"ratios={ratios}")
    print(f"seeds={seeds}")
    print(f"timeout_sec={args.timeout_sec} finetune_epochs={args.finetune_epochs}")
    print("=" * 72)

    append_run_log(run_log, "=" * 72)
    append_run_log(run_log, f"P1-v5 start: {start_dt.isoformat()}")
    append_run_log(run_log, f"project_root={PROJECT_ROOT}")
    append_run_log(run_log, f"python_exe={args.python_exe}")
    append_run_log(run_log, f"single_task_script={args.single_task_script}")
    append_run_log(run_log, f"result_dir={result_dir}")
    append_run_log(run_log, f"scope={args.pruning_scope} workers={args.num_workers} total={total_exp}")
    append_run_log(run_log, f"timeout_sec={args.timeout_sec} finetune_epochs={args.finetune_epochs}")

    completed, results = load_checkpoint(checkpoint_path)
    tasks = build_tasks(architectures, methods, ratios, seeds, completed)

    total = len(completed) + len(tasks)
    done_before = len(completed)
    done_now = 0

    print(f"checkpoint_done={done_before} remaining={len(tasks)} total={total}")
    append_run_log(run_log, f"checkpoint_done={done_before} remaining={len(tasks)} total={total}")

    if not tasks:
        print("No pending tasks. Already complete.")
        append_run_log(run_log, "No pending tasks. Already complete.")
        return

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(run_task, t, completed, results, args, run_log, checkpoint_path, csv_file): t
            for t in tasks
        }

        for fut in as_completed(futures):
            info = fut.result()
            done_now += 1
            done_total = done_before + done_now
            remain = total - done_total

            elapsed_total = time.time() - start_ts
            avg_per_task = elapsed_total / max(done_now, 1)
            eta_sec = avg_per_task * remain
            eta_time = datetime.now() + timedelta(seconds=eta_sec)

            arch, method, ratio, seed = info["task"]
            head = f"[{done_total}/{total}] {arch}|{method}|r={ratio}|s={seed}"

            if info["ok"]:
                final_acc = info["result"].get("final_acc", "NA")
                msg = (
                    f"{head} DONE final={final_acc} task_sec={info['elapsed_sec']:.1f} "
                    f"ETA={format_eta(eta_sec)} ({eta_time.strftime('%H:%M:%S')})"
                )
            else:
                reason = "timeout" if info.get("timeout") else f"rc={info.get('returncode', 'NA')}"
                msg = (
                    f"{head} FAIL {reason} task_sec={info['elapsed_sec']:.1f} "
                    f"ETA={format_eta(eta_sec)} ({eta_time.strftime('%H:%M:%S')})"
                )

            print(msg, flush=True)
            append_run_log(run_log, msg)

    end_dt = datetime.now()
    print("=" * 72)
    print(f"P1-v5 finished at {end_dt.isoformat()} elapsed={end_dt - start_dt}")
    print("=" * 72)
    append_run_log(run_log, f"P1-v5 finished at {end_dt.isoformat()} elapsed={end_dt - start_dt}")


if __name__ == "__main__":
    main()
