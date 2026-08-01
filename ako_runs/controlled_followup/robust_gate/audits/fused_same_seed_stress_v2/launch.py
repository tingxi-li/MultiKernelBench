"""One-shot launch-readiness preflight; never waits or silently queues work."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.robust_gate.schema import file_sha256, load_json

from .runner import FREEZE_PATH, HERE, LAUNCH_PATH, MANIFEST_PATH, SEED_PLAN_PATH, gpu_snapshot, validate_gpu, verify_campaign


PREFLIGHT_PATH = HERE / "receipts" / "launch_preflight_receipt.json"


def _exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite of {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def live_preflight() -> dict[str, Any]:
    manifest, _gate, _lock, _seeds = verify_campaign(require_freeze=True)
    if not LAUNCH_PATH.is_file():
        raise FileNotFoundError("missing frozen launch receipt")
    launch = load_json(LAUNCH_PATH)
    if launch.get("freeze_receipt_sha256") != file_sha256(FREEZE_PATH) or launch.get("seed_plan_sha256") != file_sha256(SEED_PLAN_PATH):
        raise ValueError("launch receipt binding mismatch")
    physical = manifest["hardware"]["physical_gpu"]
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
    receipt: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "fused_same_seed_stress_launch_preflight_receipt",
        "campaign_id": manifest["campaign_id"],
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "freeze_receipt_sha256": file_sha256(FREEZE_PATH),
        "launch_receipt_sha256": file_sha256(LAUNCH_PATH),
        "physical_gpu": physical,
        "required_gpu_uuid": manifest["hardware"]["required_uuid"],
        "driver_command_returncode": result.returncode,
        "driver_stdout": result.stdout.strip(),
        "driver_stderr": result.stderr.strip(),
    }
    if result.returncode != 0:
        receipt.update({"status": "blocked", "reason": "nvidia_driver_unavailable", "launched": False})
        return receipt
    snapshot = gpu_snapshot(physical)
    validate_gpu(snapshot, manifest)
    occupancy = subprocess.run(
        ["nvidia-smi", f"--id={physical}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    occupants = [line.strip() for line in occupancy.stdout.splitlines() if line.strip()]
    receipt["gpu_snapshot"] = snapshot
    receipt["occupants"] = occupants
    if occupancy.returncode != 0:
        receipt.update({"status": "blocked", "reason": "occupancy_query_failed", "launched": False})
    elif occupants:
        receipt.update({"status": "blocked", "reason": "gpu_not_idle", "launched": False})
    else:
        receipt.update({"status": "ready", "reason": None, "launched": False, "command": launch["command"]})
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", required=True)
    args = parser.parse_args()
    receipt = live_preflight()
    if PREFLIGHT_PATH.exists():
        prior = load_json(PREFLIGHT_PATH)
        print(json.dumps({"current": receipt, "preserved_first_preflight": prior}, sort_keys=True))
    else:
        _exclusive(PREFLIGHT_PATH, receipt)
        print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())

