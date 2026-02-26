"""Batch dispatcher for equal-budget runs across seeds with checkpoint/retry.

Runs:
  python experiments/run_chip_equal_budget.py --seed <seed> [--strict-iso]

Features:
  - fixed seed list: [42,123,456,789,1024]
  - checkpoint resume
  - up to 2 retries per failed seed
  - per-seed log + batch log
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "chip_results"
RUNNER = ROOT / "experiments" / "run_chip_equal_budget.py"
PYTHON = sys.executable
SEEDS = [42, 123, 456, 789, 1024]
CKPT = RESULT_DIR / "eb_batch_checkpoint.json"
BATCH_LOG = RESULT_DIR / "eb_batch.log"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with BATCH_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_ckpt(resume: bool) -> Dict:
    if resume and CKPT.exists():
        return json.loads(CKPT.read_text(encoding="utf-8"))
    return {
        "start_time": datetime.now().isoformat(),
        "done": [],
        "failed": [],
        "attempts": {},
    }


def _save_ckpt(ckpt: Dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_seed(seed: int, strict_iso: bool, rewrite_raw: bool) -> int:
    seed_log = RESULT_DIR / f"eb_seed_{seed}.log"
    cmd = [PYTHON, str(RUNNER), "--seed", str(seed)]
    if strict_iso:
        cmd.append("--strict-iso")
    if rewrite_raw:
        cmd.append("--rewrite-raw")

    _log("=" * 70)
    _log(f"START seed={seed} cmd={' '.join(cmd)}")
    _log("=" * 70)
    t0 = time.time()
    with seed_log.open("a", encoding="utf-8") as fout:
        fout.write(f"\n\n[{_now()}] START {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=fout,
            stderr=fout,
            timeout=7200,
        )
        fout.write(f"[{_now()}] END returncode={proc.returncode}\n")
    dt = time.time() - t0
    _log(f"END seed={seed} rc={proc.returncode} elapsed={dt:.1f}s log={seed_log}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch runner for equal-budget seeds")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--strict-iso", action="store_true", help="forward to runner")
    parser.add_argument("--rewrite-raw", action="store_true",
                        help="rewrite raw on the first seed run")
    parser.add_argument("--max-retries", type=int, default=2, help="retry per seed")
    parser.add_argument("--avg-seed-sec", type=float, default=1708.0,
                        help="ETA basis per seed (4 cells x 2 methods x 427s)")
    args = parser.parse_args()

    ckpt = _load_ckpt(args.resume)
    done = set(int(s) for s in ckpt.get("done", []))

    pending = [s for s in SEEDS if s not in done]
    est = len(pending) * float(args.avg_seed_sec)
    _log(f"profile=equal-budget-batch seeds={SEEDS}")
    _log(f"completed={len(done)} pending={len(pending)}")
    _log(f"estimated_total_time={est:.1f}s ({est/3600:.2f}h)")
    _log(f"avg_task_time_sec={float(args.avg_seed_sec):.1f}")
    _log("eta_basis=4cells x 2methods x ~427s/method-cell")

    if not pending:
        _log("No pending seeds. Exit.")
        return 0

    first_seed = True
    for seed in pending:
        attempts = int(ckpt.get("attempts", {}).get(str(seed), 0))
        ok = False
        while attempts <= args.max_retries:
            rewrite_flag = bool(args.rewrite_raw and first_seed and attempts == 0)
            rc = _run_seed(
                seed=seed,
                strict_iso=args.strict_iso,
                rewrite_raw=rewrite_flag,
            )
            attempts += 1
            ckpt.setdefault("attempts", {})[str(seed)] = attempts
            _save_ckpt(ckpt)
            if rc == 0:
                ok = True
                break
            _log(f"RETRY seed={seed} attempt={attempts}/{args.max_retries + 1}")

        first_seed = False
        if ok:
            ckpt.setdefault("done", []).append(seed)
            _save_ckpt(ckpt)
            _log(f"PASS seed={seed}")
        else:
            ckpt.setdefault("failed", []).append(
                {
                    "seed": seed,
                    "attempts": attempts,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            _save_ckpt(ckpt)
            _log(f"FAIL seed={seed} after attempts={attempts}")

    done_n = len(ckpt.get("done", []))
    fail_n = len(ckpt.get("failed", []))
    _log(f"FINISH done={done_n} failed={fail_n} total={len(SEEDS)}")
    return 0 if fail_n == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

