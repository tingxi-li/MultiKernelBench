#!/usr/bin/env python3
"""Pure protocol, hashing, manifest, planning, and statistics helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_PATH = HERE / "campaign.json"
CELLS_PATH = HERE / "cells.json"
LOCK_PATH = HERE / "launch_lock.json"
RESULTS_ROOT = HERE / "results"
BASE_JOBS_PATH = REPO_ROOT / "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json"
ROBUST_ADAPTER_PATH = REPO_ROOT / "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json"

LANES = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
STRATEGIES = ("register_fused", "smem_staged", "global_intermediate")
GRID_IDS = tuple(f"g{index:02d}" for index in range(19))
TERMINAL_AUDIT_OUTCOMES = (
    "UNSUPPORTED",
    "BUILD_FAILED",
    "LAUNCH_FAILED",
    "GATE_FAILED",
    "GATE_PASSED",
)

# Only builders whose implementation path can be inspected in checked-in code
# are advertised. A requested-but-unsupported cell is still emitted and remains
# in every feasibility denominator.
SUPPORT: dict[tuple[str, str], tuple[bool, str]] = {
    ("register_fused", "tilelang"): (True, "phase2 TileLang accumulator-fragment epilogue"),
    ("register_fused", "triton"): (True, "phase2 Triton accumulator epilogue"),
    ("register_fused", "cuda_noptx"): (
        False,
        "nvcuda::wmma fragment element layout is opaque; no safe register epilogue exists without inline PTX or a strategy substitution",
    ),
    ("register_fused", "cuda_unlimited"): (True, "phase2 explicit-MMA register epilogue"),
    ("smem_staged", "tilelang"): (True, "crossed-v1 explicit TileLang shared accumulator staging"),
    ("smem_staged", "triton"): (
        False,
        "the pinned Triton API has no explicit user-managed shared-memory accumulator allocation; global scratch would be the global_intermediate strategy",
    ),
    ("smem_staged", "cuda_noptx"): (True, "phase2 WMMA shared accumulator staging"),
    ("smem_staged", "cuda_unlimited"): (True, "phase2 explicit-MMA shared accumulator staging"),
    ("global_intermediate", "tilelang"): (True, "phase1 TileLang GEMM plus common combined postprocess"),
    ("global_intermediate", "triton"): (True, "phase1 Triton GEMM plus common combined postprocess"),
    ("global_intermediate", "cuda_noptx"): (True, "phase1 WMMA GEMM plus common combined postprocess"),
    ("global_intermediate", "cuda_unlimited"): (True, "phase1 explicit-MMA GEMM plus common combined postprocess"),
}

SOURCE_PATHS = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/campaign.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/cells.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/core.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/make_manifest.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/freeze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/validate.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/candidates.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/tilelang_smem.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/postprocess.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/audit.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/run_one.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/launch.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/analyze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/capture_evidence.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/README.md",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/tests/test_protocol.py",
)

DEPENDENCY_PATHS = (
    "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json",
    "ako_runs/controlled_followup/fused_grid/robust_adapter.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json",
    "ako_runs/controlled_followup/robust_gate/manifest.json",
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
    "ako_runs/phase1_matmul/common.py",
    "ako_runs/phase1_matmul/variants/__init__.py",
    "ako_runs/phase1_matmul/variants/tilelang_gemm.py",
    "ako_runs/phase1_matmul/variants/triton_gemm.py",
    "ako_runs/phase1_matmul/variants/cuda_noptx_gemm.py",
    "ako_runs/phase1_matmul/variants/cuda_unlimited_gemm.py",
    "ako_runs/phase2_fused_sdpa/common2.py",
    "ako_runs/phase2_fused_sdpa/runner2.py",
    "ako_runs/phase2_fused_sdpa/variants2/__init__.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_tilelang.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_triton.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_cuda_noptx.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_cuda_unlimited.py",
)


class ProtocolError(RuntimeError):
    """A frozen campaign or evidence contract was violated."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc


def stable_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def result_root(tag: str) -> Path:
    if not tag or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in tag):
        raise ProtocolError(f"unsafe result tag: {tag!r}")
    root = (RESULTS_ROOT / tag).resolve()
    root.relative_to(RESULTS_ROOT.resolve())
    return root


def cell_filename(cell_id: str) -> str:
    return cell_id.replace(".", "__") + ".json"


def timing_filename(cell_id: str, distribution: str, rep: int) -> str:
    return f"{cell_id.replace('.', '__')}__{distribution}__rep{rep:02d}.json"


def parse_set(set_string: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in set_string.split(","):
        key, value = item.split("=", 1)
        if key in {"arith", "cast"}:
            result[key] = value
        elif key.startswith("x_"):
            result.setdefault("extra", {})[key[2:]] = value
        else:
            result[key] = int(value)
    return result


def make_cells(base_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(job["dsl"], job["grid_id"]): job for job in base_jobs}
    expected = {(lane, grid_id) for lane in LANES for grid_id in GRID_IDS}
    if set(by_key) != expected or len(base_jobs) != 76:
        raise ProtocolError("base fused grid is not the frozen 4 x 19 design")
    cells = []
    index = 0
    for strategy in STRATEGIES:
        for lane in LANES:
            supported, detail = SUPPORT[(strategy, lane)]
            for grid_id in GRID_IDS:
                origin = by_key[(lane, grid_id)]
                cells.append(
                    {
                        "cell_id": f"{strategy}.{lane}.{grid_id}",
                        "cell_index": index,
                        "grid_id": grid_id,
                        "grid_index": int(grid_id[1:]),
                        "lane": lane,
                        "origin_job": origin,
                        "origin_job_sha256": canonical_sha256(origin),
                        "requested": True,
                        "strategy": strategy,
                        "support_declared": supported,
                        "support_detail": detail,
                    }
                )
                index += 1
    return cells


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("schema_version") != 1 or campaign.get("campaign_id") != "fused-epilogue-crossed-v1":
        raise ProtocolError("unexpected campaign identity/schema")
    factors = campaign.get("factors", {})
    if factors.get("strategies") != list(STRATEGIES):
        raise ProtocolError("strategy order changed")
    if factors.get("lanes") != list(LANES):
        raise ProtocolError("lane order changed")
    if factors.get("grid_ids") != list(GRID_IDS):
        raise ProtocolError("grid order changed")
    performance = campaign.get("performance", {})
    if performance.get("screen") != {
        "dist": "rand", "order_seed": 2026073102, "processes": 2, "seed": 0
    }:
        raise ProtocolError("screen protocol changed")
    confirmation = performance.get("confirmation", {})
    if confirmation.get("blocks_per_distribution") != 15 or confirmation.get("order_seed") != 2026073103:
        raise ProtocolError("confirmation protocol changed")
    if confirmation.get("distributions") != [
        {"dist": "rand", "label": "positive", "seed": 0},
        {"dist": "randn", "label": "withheld_signed", "seed": 2026073101},
    ]:
        raise ProtocolError("confirmation distributions changed")
    if performance.get("trials") != 100 or performance.get("warmup_s") != 2.0 or performance.get("flush_l2") is not True:
        raise ProtocolError("timing contract changed")


def validate_cells(cells: list[dict[str, Any]], base_jobs: list[dict[str, Any]] | None = None) -> None:
    if len(cells) != 228 or len({cell.get("cell_id") for cell in cells}) != 228:
        raise ProtocolError("cell manifest must contain 228 unique cells")
    if [cell.get("cell_index") for cell in cells] != list(range(228)):
        raise ProtocolError("cell indices are not canonical")
    expected = make_cells(base_jobs if base_jobs is not None else read_json(BASE_JOBS_PATH))
    if cells != expected:
        raise ProtocolError("cell manifest differs from deterministic 3 x 4 x 19 expansion")


def load_contract() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    campaign = read_json(CAMPAIGN_PATH)
    cells = read_json(CELLS_PATH)
    lock = read_json(LOCK_PATH)
    validate_campaign(campaign)
    validate_cells(cells)
    if lock.get("schema_version") != 1 or lock.get("campaign_id") != campaign["campaign_id"]:
        raise ProtocolError("unexpected launch-lock identity/schema")
    if lock.get("campaign_sha256") != file_sha256(CAMPAIGN_PATH):
        raise ProtocolError("launch lock campaign hash mismatch")
    if lock.get("cells_sha256") != file_sha256(CELLS_PATH):
        raise ProtocolError("launch lock cell-manifest hash mismatch")
    for relative, expected in lock.get("source_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"frozen source changed: {relative}")
    for relative, expected in lock.get("dependency_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"frozen dependency changed: {relative}")
    if canonical_sha256(lock.get("source_sha256")) != lock.get("source_bundle_sha256"):
        raise ProtocolError("source bundle digest mismatch")
    if canonical_sha256(lock.get("dependency_sha256")) != lock.get("dependency_bundle_sha256"):
        raise ProtocolError("dependency bundle digest mismatch")
    adapter = read_json(ROBUST_ADAPTER_PATH)
    expected_gate_binding = {
        "adapter_manifest_sha256": file_sha256(ROBUST_ADAPTER_PATH),
        "gate_spec_sha256": adapter["robust_gate"]["gate_spec_sha256"],
        "manifest_sha256": adapter["robust_gate"]["manifest_sha256"],
    }
    if lock.get("frozen_gate") != expected_gate_binding:
        raise ProtocolError("launch lock does not bind the current frozen fused-v2 gate")
    return campaign, cells, lock


def gpu_snapshot(index: int) -> dict[str, str]:
    fields = ("index", "uuid", "name", "driver_version", "compute_cap", "pstate", "memory.total", "memory.used", "utilization.gpu", "clocks.sm", "clocks.mem", "power.limit", "temperature.gpu")
    completed = subprocess.run(
        ["nvidia-smi", f"--id={index}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if completed.returncode != 0:
        raise ProtocolError(completed.stderr.strip() or "nvidia-smi failed")
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != len(fields):
        raise ProtocolError("unexpected nvidia-smi snapshot")
    return dict(zip(fields, values))


def validate_gpu(snapshot: dict[str, str], campaign: dict[str, Any]) -> None:
    hardware = campaign["hardware"]
    if snapshot.get("uuid") not in hardware["allowed_gpu_uuids"]:
        raise ProtocolError(f"GPU UUID not frozen: {snapshot.get('uuid')}")
    if snapshot.get("name") != hardware["required_name"] or snapshot.get("compute_cap") != hardware["required_compute_capability"]:
        raise ProtocolError("GPU model/compute capability mismatch")


def nvcc_fingerprint() -> dict[str, Any]:
    candidates = [
        Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.1")) / "bin/nvcc",
        Path("/usr/local/cuda-13.1/bin/nvcc"),
        Path(shutil.which("nvcc")) if shutil.which("nvcc") else None,
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        completed = subprocess.run(
            [str(candidate), "--version"], capture_output=True, text=True, timeout=20
        )
        return {
            "path": str(candidate),
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "stdout": completed.stdout.strip(),
        }
    return {"path": None, "returncode": None, "stderr": "nvcc not found", "stdout": ""}


def screen_plan(cells: list[dict[str, Any]], legal_ids: set[str], seed: int = 2026073102) -> list[dict[str, Any]]:
    result = []
    randomizer = random.Random(seed)
    legal = [cell["cell_id"] for cell in cells if cell["cell_id"] in legal_ids]
    for rep in range(2):
        block = list(legal)
        randomizer.shuffle(block)
        result.extend({"rep": rep, "cell_id": cell_id, "distribution": "positive"} for cell_id in block)
    return result


def confirmation_plan(selected_ids: set[str], seed: int = 2026073103) -> list[dict[str, Any]]:
    result = []
    randomizer = random.Random(seed)
    for rep in range(15):
        # Jointly randomize both distributions inside every process block so
        # signed-vs-positive timing cannot be identified with campaign order.
        block = [
            (cell_id, distribution)
            for cell_id in sorted(selected_ids)
            for distribution in ("positive", "withheld_signed")
        ]
        randomizer.shuffle(block)
        result.extend(
            {"rep": rep, "cell_id": cell_id, "distribution": distribution}
            for cell_id, distribution in block
        )
    return result


def exact_median_interval(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value <= 0 for value in ordered):
        raise ProtocolError("median interval requires positive finite values")
    n = len(ordered)
    median = statistics.median(ordered)
    choices = []
    for k in range(1, (n + 1) // 2 + 1):
        coverage = 1.0 - 2.0 * sum(math.comb(n, index) for index in range(k)) / (2**n)
        if coverage >= 0.95:
            choices.append((k, coverage))
    if choices:
        k, coverage = max(choices)
        lo, hi, meets = ordered[k - 1], ordered[n - k], True
    else:
        k, coverage = 1, 1.0 - 2.0 / (2**n)
        lo, hi, meets = ordered[0], ordered[-1], False
    return {"n": n, "median": median, "ci_lo": lo, "ci_hi": hi, "order_k": k, "coverage": coverage, "meets_95": meets}


def spearman_rank(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    def ranks(values: list[float]) -> list[float]:
        result = [0.0] * len(values)
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        position = 0
        while position < len(ordered):
            stop = position + 1
            while stop < len(ordered) and values[ordered[stop]] == values[ordered[position]]:
                stop += 1
            rank = (position + stop - 1) / 2.0 + 1.0
            for index in ordered[position:stop]:
                result[index] = rank
            position = stop
        return result
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return numerator / denominator if denominator else None
