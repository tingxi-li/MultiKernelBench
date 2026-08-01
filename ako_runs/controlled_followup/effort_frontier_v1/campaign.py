"""Frozen constants and deterministic jobs for the RQ5 effort frontier."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_ID = "fused-effort-frontier-v1-20260731"
MODEL = {"provider": "openai", "requested_alias": "gpt-5.6-sol"}
PROGRAMMABLE_LANES = ("cublaslt", "triton", "tilelang", "cuda_unlimited")
CONTROL_LANE = "torch_contract_fp32"
SEARCH_REPLICATES = 5
CHECKPOINTS_S = (1800, 7200, 28800)
CHECKPOINT_LABELS = ("0p5h", "2h", "8h")
SEARCH_NAMESPACE = "MKB-effort-frontier-v1-search-seed-20260731"
TIMING_DISTRIBUTIONS = ("positive_rand_seed0", "signed_mixed_withheld")
CONFIRM_BLOCKS = 15
CONFIRM_WARMUP = 25
CONFIRM_TRIALS = 100
GPU_UUIDS = (
    "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae",
    "GPU-91b61ae6-d21e-485e-43b3-7505d62149b1",
    "GPU-3ed448f2-f23a-09fc-e18b-f580523c4a3f",
    "GPU-eafdd6ce-8857-40fd-f494-47a7240bf6b5",
)
GATE_SPEC = REPO_ROOT / (
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json"
)
GATE_RECEIPT = REPO_ROOT / (
    "ako_runs/controlled_followup/robust_gate/validation/fused_gate_acceptance_v2.json"
)
MODEL_LOCK = HERE / "locks/model_resolution_lock.json"
EXECUTOR_REGISTRY = HERE / "locks/executor_registry.json"
PROVENANCE_LOCK = HERE / "locks/prelaunch_provenance.json"
MANIFEST = HERE / "manifest.json"
TREATMENT_ARTIFACT_NOTE = HERE / "EXECUTOR_TREATMENT_ARTIFACT.md"

# Files whose exact bytes must be named by the prelaunch provenance lock.  The
# result directory, mutable external locks, Python caches, and generated
# analysis are deliberately outside this set.
PROVENANCE_FILES = (
    "README.md",
    "EXECUTOR_TREATMENT_ARTIFACT.md",
    "__init__.py",
    "analyze.py",
    "campaign.py",
    "capture_evidence.py",
    "confirmation.py",
    "controller.py",
    "events.py",
    "launch.py",
    "make_manifest.py",
    "manifest.json",
    "prompts/system.md",
    "prompts/task.md",
    "validate.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    """Atomically publish one file and fsync both the file and its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def campaign_provenance_hashes() -> dict[str, str]:
    return {
        f"ako_runs/controlled_followup/effort_frontier_v1/{relative}": file_sha256(
            HERE / relative
        )
        for relative in PROVENANCE_FILES
    }


def search_seed(lane: str, replicate: int) -> int:
    if lane not in PROGRAMMABLE_LANES or replicate not in range(SEARCH_REPLICATES):
        raise ValueError("unregistered effort-frontier seed coordinates")
    payload = f"{SEARCH_NAMESPACE}|{lane}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def trajectories() -> list[dict[str, Any]]:
    rows = []
    ordinal = 0
    for lane in PROGRAMMABLE_LANES:
        for replicate in range(SEARCH_REPLICATES):
            gpu = ordinal % 4
            rows.append(
                {
                    "ordinal": ordinal,
                    "trajectory_id": f"effort_v1.{lane}.r{replicate}",
                    "lane": lane,
                    "replicate": replicate,
                    "search_seed": search_seed(lane, replicate),
                    "physical_gpu": gpu,
                    "required_gpu_uuid": GPU_UUIDS[gpu],
                    "logical_device": "cuda:0",
                    "checkpoints": [
                        {"label": label, "active_effort_s": seconds}
                        for label, seconds in zip(CHECKPOINT_LABELS, CHECKPOINTS_S)
                    ],
                }
            )
            ordinal += 1
    return rows
