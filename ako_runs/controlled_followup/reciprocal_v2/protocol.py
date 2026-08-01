"""Deterministic protocol helpers for reciprocal transfer v2.

This module is deliberately CPU-only.  Importing it must never import a GPU DSL,
load CUDA, or inspect performance results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_ID = "standard-matmul-reciprocal-transfer-v2-20260731"
ORIGINS = (
    "tilelang_phase1_confirmed",
    "triton_grouped_autotuned",
    "cuda_unlimited_native_d",
)
DESTINATIONS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
TRANSFER_MODES = ("literal", "retuned")
TRANSLATORS = ("translator_a", "translator_b")
KINDS = ("audit", "primary")
STAGES = ("audit", "screen", "primary")
KC_LADDER = (8192, 4096, 2048, 1024, 512)
RETUNE_ATTEMPTS = 19
SCREEN_REPS = 2
CONFIRM_REPS = 15
BLOCK_ORDER_SEED = 2026073103
TIMING_PHYSICAL_GPU = 0
TIMING_GPU_UUID = "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae"
TIMING_GPU_NAME = "NVIDIA RTX 6000 Ada Generation"

GATE_SPEC = REPO_ROOT / (
    "ako_runs/controlled_followup/robust_gate/calibration/"
    "gate_spec_matmul_v4.json"
)
GATE_SUMMARY = REPO_ROOT / (
    "ako_runs/controlled_followup/robust_gate/validation/"
    "matmul_holdout_summary_v4.json"
)
GATE_RECEIPT = REPO_ROOT / (
    "ako_runs/controlled_followup/robust_gate/validation/"
    "v4_acceptance_receipt.json"
)
GATE_LOCK = HERE / "dependencies/gate_lock.json"
RESOLUTION_LOCK = HERE / "dependencies/recipe_resolution_lock.json"
IMPLEMENTATION_REGISTRY = HERE / "dependencies/implementation_registry.json"
TRANSLATOR_ISOLATION_LOCK = HERE / "dependencies/translator_isolation.json"
PROVENANCE_LOCK = HERE / "dependencies/prelaunch_provenance.json"
SOURCE_FREEZE = HERE / "receipts/source_freeze.json"
PRIMARY_RAW = HERE / "results/primary/measurements.jsonl"
PRIMARY_ANALYSIS = HERE / "results/primary/analysis.json"
PRIMARY_COMPLETION = HERE / "results/primary/completion_receipt.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def expected_cells() -> list[tuple[str, str, str, str]]:
    return [
        (origin, destination, mode, translator)
        for origin in ORIGINS
        for destination in DESTINATIONS
        for mode in TRANSFER_MODES
        for translator in TRANSLATORS
    ]


def cell_id(origin: str, destination: str, mode: str, translator: str) -> str:
    return f"{origin}__to__{destination}__{mode}__{translator}"


def expected_resolution_cells() -> list[tuple[str, str, str]]:
    """The 24 independently translated recipe/destination implementations."""
    return [
        (origin, destination, translator)
        for origin in ORIGINS
        for destination in DESTINATIONS
        for translator in TRANSLATORS
    ]


def resolution_cell_id(origin: str, destination: str, translator: str) -> str:
    return f"{origin}__to__{destination}__{translator}"


def block_orders() -> list[list[str]]:
    """Return 15 deterministic complete-block permutations without RNG drift."""
    cells = [cell_id(*axes) for axes in expected_cells()]
    return [
        sorted(
            cells,
            key=lambda cid: hashlib.sha256(
                f"{BLOCK_ORDER_SEED}:{block}:{cid}".encode("utf-8")
            ).digest(),
        )
        for block in range(CONFIRM_REPS)
    ]


def block_order_sha256() -> str:
    return canonical_sha256(block_orders())
