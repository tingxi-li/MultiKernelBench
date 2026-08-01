"""Guarded kickoff for the same-seed GPU stress.

The process waits for a functioning NVIDIA driver, verifies the selected GPU
is not occupied, then launches the frozen runner exactly once and analyzes its
append-only output. It exits after completion or a runner failure.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "measurements.jsonl"
SUMMARY = HERE / "results" / "summary.json"
GPU = "2"
POLL_SECONDS = 30


def stamp(message: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def driver_ready() -> bool:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return result.returncode == 0 and "GPU" in result.stdout


def gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", f"--id={GPU}", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def main() -> int:
    if OUT.exists() or SUMMARY.exists():
        stamp("existing output detected; refusing a second launch")
        return 2
    stamp(f"waiting for NVIDIA driver and idle physical GPU {GPU}")
    while not driver_ready():
        time.sleep(POLL_SECONDS)
    while not gpu_idle():
        stamp(f"GPU {GPU} is occupied; waiting")
        time.sleep(POLL_SECONDS)
    stamp(f"launching same-seed stress on physical GPU {GPU}")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run = subprocess.run(
        ["python", "-m", "ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v1.run"],
        cwd=HERE.parents[4], env=env,
    )
    if run.returncode != 0:
        stamp(f"runner failed with exit code {run.returncode}")
        return run.returncode
    stamp("runner complete; analyzing")
    analysis = subprocess.run(
        ["python", "-m", "ako_runs.controlled_followup.robust_gate.audits.fused_same_seed_stress_v1.analyze"],
        cwd=HERE.parents[4], env=env,
    )
    stamp(f"analysis exit code {analysis.returncode}")
    return analysis.returncode


if __name__ == "__main__":
    raise SystemExit(main())
