"""
P1 validation matrix runner for GRA v5.
Matrix: 2 arch x 5 methods x 3 ratios x 5 seeds = 150 tasks.
Methods: L1, FPGM, GRA-v4, GRA-v5, Random
"""
import csv
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

PYTHON = r"C:\GRA-CNN\.venv\Scripts\python.exe"
SCRIPT = r"C:\GRA-CNN\experiments\single_task_v5.py"
RESULT_DIR = r"C:\GRA-CNN\experiments\stage_mid_pruning\results_v5_p1"
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint.json")
RUN_LOG = os.path.join(RESULT_DIR, "run_log.txt")
NUM_WORKERS = 1

ARCHITECTURES = ["resnet20", "resnet56"]
METHODS = ["L1", "FPGM", "GRA-v4", "GRA-v5", "Random"]
RATIOS = [0.3, 0.5, 0.7]
SEEDS = [42, 123, 456, 789, 1024]
PRUNING_SCOPE = "stage_mid_only"

lock = threading.Lock()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_run_log(message: str) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {message}\n")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return set(), []

    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        data = json.load(f)
    completed = set(tuple(x) for x in data.get("completed", []))
    results = data.get("results", [])
    return completed, results


def save_results_csv(results):
    if not results:
        return

    csv_file = os.path.join(RESULT_DIR, "results.csv")
    fieldnames = [
        "architecture", "dataset", "method", "ratio", "iso_flops", "seed",
        "gra_version", "baseline_acc", "pruned_acc", "final_acc",
        "params_before", "params_after", "compression_ratio",
        "pruning_scope", "timestamp"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def save_checkpoint(completed, results):
    payload = {
        "completed": [list(k) for k in sorted(completed)],
        "results": results,
    }
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_results_csv(results)


def parse_result(stdout_text: str):
    for line in stdout_text.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return None


def run_task(task, completed, results):
    arch, method, ratio, seed = task
    cmd = [
        PYTHON,
        SCRIPT,
        "--arch", arch,
        "--method", method,
        "--ratio", str(ratio),
        "--seed", str(seed),
        "--pruning_scope", PRUNING_SCOPE,
    ]

    task_start = time.time()
    append_run_log(f"[START] {arch}|{method}|r={ratio}|s={seed}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=r"C:\GRA-CNN",
        )
        elapsed = time.time() - task_start
        result = parse_result(proc.stdout)

        if result is not None:
            with lock:
                completed.add(task)
                results.append(result)
                save_checkpoint(completed, results)
            append_run_log(
                f"[DONE] {arch}|{method}|r={ratio}|s={seed} "
                f"final={result.get('final_acc', 'NA')} elapsed_sec={elapsed:.1f}"
            )
            return {"ok": True, "task": task, "result": result, "elapsed_sec": elapsed}

        append_run_log(f"[FAIL] {arch}|{method}|r={ratio}|s={seed} rc={proc.returncode}")
        return {"ok": False, "task": task, "elapsed_sec": elapsed, "returncode": proc.returncode}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - task_start
        append_run_log(f"[FAIL] {arch}|{method}|r={ratio}|s={seed} timeout=1800")
        return {"ok": False, "task": task, "elapsed_sec": elapsed, "timeout": True}


def format_eta(seconds_left: float) -> str:
    if seconds_left <= 0:
        return "0s"
    return str(timedelta(seconds=int(seconds_left)))


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    start_dt = datetime.now()
    start_ts = time.time()
    total_exp = len(ARCHITECTURES) * len(METHODS) * len(RATIOS) * len(SEEDS)

    print("=" * 72)
    print(f"P1-v5 start at {start_dt.isoformat()}")
    print(f"scope={PRUNING_SCOPE} workers={NUM_WORKERS} total={total_exp}")
    print(f"architectures={ARCHITECTURES}")
    print(f"methods={METHODS}")
    print(f"ratios={RATIOS}")
    print(f"seeds={SEEDS}")
    print("=" * 72)

    append_run_log("=" * 72)
    append_run_log(f"P1-v5 start: {start_dt.isoformat()}")
    append_run_log(f"scope={PRUNING_SCOPE} workers={NUM_WORKERS} total={total_exp}")

    completed, results = load_checkpoint()

    tasks = []
    for arch in ARCHITECTURES:
        for method in METHODS:
            for ratio in RATIOS:
                for seed in SEEDS:
                    key = (arch, method, ratio, seed)
                    if key not in completed:
                        tasks.append(key)

    total = len(completed) + len(tasks)
    done_before = len(completed)
    done_now = 0

    print(f"checkpoint_done={done_before} remaining={len(tasks)} total={total}")
    append_run_log(f"checkpoint_done={done_before} remaining={len(tasks)} total={total}")

    if not tasks:
        print("No pending tasks. Already complete.")
        append_run_log("No pending tasks. Already complete.")
        return

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(run_task, t, completed, results): t for t in tasks}

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

            print(msg)
            append_run_log(msg)

    end_dt = datetime.now()
    print("=" * 72)
    print(f"P1-v5 finished at {end_dt.isoformat()} elapsed={end_dt - start_dt}")
    print("=" * 72)
    append_run_log(f"P1-v5 finished at {end_dt.isoformat()} elapsed={end_dt - start_dt}")


if __name__ == "__main__":
    main()
