#!/usr/bin/env python3
"""Deterministic v2 protocol built on the sealed v1's generic helpers."""
from __future__ import annotations

import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ako_runs.controlled_followup.fused_epilogue_crossed_v1 import core as v1  # noqa: E402


CAMPAIGN_ID = "fused-epilogue-crossed-v2"
CAMPAIGN_PATH = HERE / "campaign.json"
SUPPORT_RESOLUTION_PATH = HERE / "support_resolution.json"
LOCK_PATH = HERE / "launch_lock.json"
PROBE_LOCK_PATH = HERE / "probe_lock.json"
RESULTS_ROOT = HERE / "results"
BASE_JOBS_PATH = v1.BASE_JOBS_PATH
ROBUST_ADAPTER_PATH = v1.ROBUST_ADAPTER_PATH
LANES = v1.LANES
GRID_IDS = v1.GRID_IDS
STRATEGIES = (
    "register_fused",
    "smem_staged",
    "global_intermediate",
    "register_common_postprocess",
)
TERMINAL_AUDIT_OUTCOMES = v1.TERMINAL_AUDIT_OUTCOMES
SHAM_BASE_CELL = "register_common_postprocess.tilelang.g01"
SHAM_LABELS = ("sham_a", "sham_b")
PROBE_KEYS = ("cuda_noptx_register", "triton_smem")
PROBE_STATUSES = ("preregistered_not_executed", "supported", "unsupported")

ProtocolError = v1.ProtocolError
canonical_bytes = v1.canonical_bytes
canonical_sha256 = v1.canonical_sha256
file_sha256 = v1.file_sha256
read_json = v1.read_json
stable_write = v1.stable_write
parse_set = v1.parse_set
exact_median_interval = v1.exact_median_interval
spearman_rank = v1.spearman_rank
gpu_snapshot = v1.gpu_snapshot
validate_gpu = v1.validate_gpu
nvcc_fingerprint = v1.nvcc_fingerprint


def ptxas_kernel_resources(log: str, kernel: str) -> dict[str, Any]:
    entries = list(re.finditer(r"Compiling entry function '([^']+)'", log))
    matches = [index for index, entry in enumerate(entries) if kernel in entry.group(1)]
    if len(matches) != 1:
        raise ProtocolError(f"ptxas log contains {len(matches)} entries for {kernel}")
    index = matches[0]
    block = log[entries[index].start() : entries[index + 1].start() if index + 1 < len(entries) else None]

    def metric(pattern: str) -> int | None:
        match = re.search(pattern, block)
        return int(match.group(1)) if match else None

    registers = metric(r"Used (\d+) registers")
    if registers is None:
        raise ProtocolError(f"ptxas register census missing for {kernel}")
    return {
        "entry": entries[index].group(1),
        "kernel": kernel,
        "registers": registers,
        "spill_store_bytes": metric(r"(\d+) bytes spill stores"),
        "spill_load_bytes": metric(r"(\d+) bytes spill loads"),
        "stack_frame_bytes": metric(r"(\d+) bytes stack frame"),
        "static_shared_bytes": metric(r"(\d+) bytes smem"),
    }


SOURCE_PATHS = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/support_resolution.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/core.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/make_manifest.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/freeze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/validate.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/candidates.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/cuda_unlimited.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/cuda_noptx_register.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/triton_smem.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/postprocess.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/support_probes.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/resolve_support.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/campaign_runner.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/analyze.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/capture_evidence.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/README.md",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/tests/__init__.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/tests/test_protocol.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/tests/test_builders.py",
)

DEPENDENCY_PATHS = (
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1r1/final_summary.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/results/crossed_v1r1/reanalysis_tail_v1.json",
    "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json",
    "ako_runs/controlled_followup/fused_grid/robust_adapter.py",
    "ako_runs/controlled_followup/fused_grid/robust_adapter_manifest.json",
    "ako_runs/controlled_followup/fused_grid/manifest.json",
    "ako_runs/controlled_followup/robust_gate/manifest.json",
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
    "ako_runs/controlled_followup/legacy_cuda_harness_fix/checked_cuda_launch.h",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/INCIDENT_CROSSED_V2_20260801.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/INCIDENT_CROSSED_V2R1_20260801.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2/audit/receipts/shard00.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2/audit/receipts/shard01.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2/audit/receipts/shard02.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v2/results/crossed_v2/audit/receipts/shard03.json",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/core.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/recovery_v1r1/audit.py",
    "ako_runs/controlled_followup/fused_epilogue_crossed_v1/tilelang_smem.py",
    "ako_runs/phase1_matmul/common.py",
    "ako_runs/phase1_matmul/variants/__init__.py",
    "ako_runs/phase1_matmul/variants/tilelang_gemm.py",
    "ako_runs/phase1_matmul/variants/triton_gemm.py",
    "ako_runs/phase1_matmul/variants/cuda_noptx_gemm.py",
    "ako_runs/phase1_matmul/variants/cuda_unlimited_gemm.py",
    "ako_runs/phase2_fused_sdpa/common2.py",
    "ako_runs/phase2_fused_sdpa/runner2.py",
    "ako_runs/phase2_fused_sdpa/variants2/__init__.py",
    "ako_runs/phase2_fused_sdpa/variants2/cuda_fused_common.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_tilelang.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_triton.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_cuda_noptx.py",
)


def result_root(tag: str) -> Path:
    if not tag or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in tag):
        raise ProtocolError(f"unsafe result tag: {tag!r}")
    root = (RESULTS_ROOT / tag).resolve()
    root.relative_to(RESULTS_ROOT.resolve())
    return root


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("schema_version") != 2 or campaign.get("campaign_id") != CAMPAIGN_ID:
        raise ProtocolError("unexpected campaign identity/schema")
    factors = campaign.get("factors", {})
    if tuple(factors.get("strategies", ())) != STRATEGIES:
        raise ProtocolError("strategy order changed")
    if tuple(factors.get("lanes", ())) != LANES or tuple(factors.get("grid_ids", ())) != GRID_IDS:
        raise ProtocolError("lane/grid order changed")
    if campaign.get("requested_cells") != 304:
        raise ProtocolError("requested-cell denominator changed")
    primary = campaign.get("inference", {}).get("primary_trials")
    if primary != {"start_inclusive": 60, "stop_exclusive": 100}:
        raise ProtocolError("settled-tail primary window changed")
    if campaign.get("inference", {}).get("sham_base_cell") != SHAM_BASE_CELL:
        raise ProtocolError("sham base changed")
    parent = campaign.get("parent", {})
    for key in ("controlling_result", "tail_overlay"):
        path = REPO_ROOT / parent.get(key, "")
        expected = parent.get(f"{key}_sha256")
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"parent binding changed: {key}")


def validate_support_resolution(value: dict[str, Any], *, require_resolved: bool = False) -> None:
    if value.get("schema_version") != 1 or value.get("campaign_id") != CAMPAIGN_ID:
        raise ProtocolError("unexpected support-resolution identity/schema")
    probes = value.get("probes")
    if not isinstance(probes, dict) or tuple(probes) != PROBE_KEYS:
        raise ProtocolError("support probe keys/order changed")
    for key in PROBE_KEYS:
        row = probes[key]
        if row.get("status") not in PROBE_STATUSES:
            raise ProtocolError(f"invalid support status: {key}")
        resolved = row["status"] in {"supported", "unsupported"}
        if resolved:
            relative = row.get("result_index_path")
            digest = row.get("result_index_sha256")
            path = REPO_ROOT / relative if isinstance(relative, str) else None
            if path is None or not path.is_file() or file_sha256(path) != digest:
                raise ProtocolError(f"resolved support probe is not hash-bound: {key}")
            index = read_json(path)
            if index.get("probe_key") != key or index.get("complete") is not True:
                raise ProtocolError(f"invalid support result index: {key}")
            attempts = index.get("attempts")
            if not isinstance(attempts, list) or len(attempts) != 19:
                raise ProtocolError(f"support probe does not retain 19 attempts: {key}")
            from ako_runs.controlled_followup.fused_epilogue_crossed_v2 import support_probes

            try:
                validated = support_probes.load_result_index(path)
            except Exception as exc:
                raise ProtocolError(f"support probe receipts do not validate: {key}: {exc}") from exc
            if validated.get("campaign_id") != CAMPAIGN_ID:
                raise ProtocolError(f"foreign support probe campaign: {key}")
            if validated.get("resolution", {}).get("status") != row["status"]:
                raise ProtocolError(f"support status is not receipt-derived: {key}")
        elif row.get("result_index_path") is not None or row.get("result_index_sha256") is not None:
            raise ProtocolError(f"unresolved support probe claims evidence: {key}")
    resolved_all = all(probes[key]["status"] in {"supported", "unsupported"} for key in PROBE_KEYS)
    if value.get("status") != ("resolved" if resolved_all else "unresolved"):
        raise ProtocolError("aggregate support status mismatch")
    if require_resolved and not resolved_all:
        pending = [key for key in PROBE_KEYS if probes[key]["status"] not in {"supported", "unsupported"}]
        raise ProtocolError(f"support probes unresolved: {pending}")


def support_evidence_paths(resolution: dict[str, Any]) -> tuple[str, ...]:
    """Return the complete, repository-local closure of resolved probe evidence."""
    validate_support_resolution(resolution, require_resolved=True)
    paths: set[str] = set()

    def add(relative: Any) -> Path:
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ProtocolError(f"probe evidence path is not repository-relative: {relative!r}")
        path = (REPO_ROOT / relative).resolve()
        try:
            normalized = str(path.relative_to(REPO_ROOT.resolve()))
        except ValueError as exc:
            raise ProtocolError(f"probe evidence escapes repository: {relative}") from exc
        if not path.is_file():
            raise ProtocolError(f"probe evidence is missing: {normalized}")
        paths.add(normalized)
        return path

    add(str(PROBE_LOCK_PATH.relative_to(REPO_ROOT)))
    for probe in resolution["probes"].values():
        index_path = add(probe["result_index_path"])
        index = read_json(index_path)
        for attempt in index["attempts"]:
            add(attempt["source_path"])
            add(attempt["diagnostics_path"])
            receipt = read_json(add(attempt["receipt_path"]))
            if receipt.get("gate_attempted") is True:
                add(receipt["gate_path"])
    return tuple(sorted(paths))


def _support(strategy: str, lane: str, resolution: dict[str, Any]) -> tuple[bool | None, str, str | None]:
    if strategy in {"register_fused", "register_common_postprocess"} and lane == "cuda_noptx":
        key = "cuda_noptx_register"
    elif strategy == "smem_staged" and lane == "triton":
        key = "triton_smem"
    else:
        key = None
    if key is None:
        return True, "supported by an inspected checked builder", None
    status = resolution["probes"][key]["status"]
    if status == "supported":
        return True, f"measured supported by {key} 19-grid probe", key
    if status == "unsupported":
        return False, f"measured unsupported by {key} 19-grid probe", key
    return None, f"support pending {key} 19-grid probe", key


def make_cells(base_jobs: list[dict[str, Any]], resolution: dict[str, Any]) -> list[dict[str, Any]]:
    validate_support_resolution(resolution)
    by_key = {(job["dsl"], job["grid_id"]): job for job in base_jobs}
    if set(by_key) != {(lane, grid) for lane in LANES for grid in GRID_IDS}:
        raise ProtocolError("base grid differs from frozen 4 x 19 design")
    cells = []
    for strategy in STRATEGIES:
        for lane in LANES:
            supported, detail, probe = _support(strategy, lane, resolution)
            for grid in GRID_IDS:
                origin = by_key[(lane, grid)]
                cells.append(
                    {
                        "cell_id": f"{strategy}.{lane}.{grid}",
                        "cell_index": len(cells),
                        "grid_id": grid,
                        "grid_index": int(grid[1:]),
                        "lane": lane,
                        "origin_job": origin,
                        "origin_job_sha256": canonical_sha256(origin),
                        "requested": True,
                        "strategy": strategy,
                        "support_declared": supported,
                        "support_detail": detail,
                        "support_probe_key": probe,
                    }
                )
    return cells


def load_cells(*, require_resolved: bool = False) -> list[dict[str, Any]]:
    resolution = read_json(SUPPORT_RESOLUTION_PATH)
    validate_support_resolution(resolution, require_resolved=require_resolved)
    return make_cells(read_json(BASE_JOBS_PATH), resolution)


def validate_cells(cells: list[dict[str, Any]], *, require_resolved: bool = False) -> None:
    expected = load_cells(require_resolved=require_resolved)
    if cells != expected or len(cells) != 304:
        raise ProtocolError("cell manifest differs from deterministic 4 x 4 x 19 expansion")
    if [cell["cell_index"] for cell in cells] != list(range(304)):
        raise ProtocolError("cell indices are not canonical")


def screen_plan(cells: list[dict[str, Any]], legal_ids: set[str]) -> list[dict[str, Any]]:
    return v1.screen_plan(cells, legal_ids, seed=2026073102)


def confirmation_plan(selected_ids: set[str]) -> list[dict[str, Any]]:
    import random

    if SHAM_BASE_CELL not in selected_ids:
        raise ProtocolError("sham base must be gate-legal and selected")
    randomizer = random.Random(2026073103)
    result = []
    for rep in range(15):
        block = [
            {"cell_id": cell_id, "distribution": distribution, "label": cell_id, "record_kind": "cell", "rep": rep}
            for cell_id in sorted(selected_ids)
            for distribution in ("positive", "withheld_signed")
        ]
        block.extend(
            {"cell_id": SHAM_BASE_CELL, "distribution": distribution, "label": label, "record_kind": "sham", "rep": rep}
            for label in SHAM_LABELS
            for distribution in ("positive", "withheld_signed")
        )
        randomizer.shuffle(block)
        result.extend(block)
    return result


def timing_filename(label: str, distribution: str, rep: int) -> str:
    return f"{label.replace('.', '__')}__{distribution}__rep{rep:02d}.json"


def summarize_times(times: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in times]
    if len(values) != 100 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ProtocolError("timing record must contain 100 positive finite trials")
    first = statistics.median(values[:10])
    last = statistics.median(values[90:])
    return {
        "full_median_ms": statistics.median(values),
        "primary_tail_median_ms": statistics.median(values[60:100]),
        "first_decile_median_ms": first,
        "last_decile_median_ms": last,
        "first_to_last_decile_ratio": last / first,
    }


def resolution_floor(sham_intervals: Iterable[dict[str, Any]]) -> float:
    endpoints = []
    for interval in sham_intervals:
        endpoints.extend((float(interval["ci_lo"]), float(interval["ci_hi"])))
    if not endpoints or any(value <= 0 or not math.isfinite(value) for value in endpoints):
        raise ProtocolError("resolution floor requires positive finite sham intervals")
    return max(abs(math.log(value)) for value in endpoints)


def effect_is_reportable(interval: dict[str, Any], floor_log_ratio: float) -> bool:
    lo, hi = math.log(float(interval["ci_lo"])), math.log(float(interval["ci_hi"]))
    return lo > floor_log_ratio or hi < -floor_log_ratio


def make_unfrozen_contract() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    campaign = read_json(CAMPAIGN_PATH)
    validate_campaign(campaign)
    resolution = read_json(SUPPORT_RESOLUTION_PATH)
    validate_support_resolution(resolution)
    return campaign, make_cells(read_json(BASE_JOBS_PATH), resolution), resolution


def validate_frozen_hashes(lock: dict[str, Any]) -> None:
    for field, bundle_field, paths in (
        ("source_sha256", "source_bundle_sha256", SOURCE_PATHS),
        ("dependency_sha256", "dependency_bundle_sha256", DEPENDENCY_PATHS),
    ):
        observed = lock.get(field, {})
        if set(observed) != set(paths):
            raise ProtocolError(f"launch lock {field} path set mismatch")
        for relative, expected in observed.items():
            path = REPO_ROOT / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise ProtocolError(f"frozen file changed: {relative}")
        if lock.get(bundle_field) != canonical_sha256(observed):
            raise ProtocolError(f"launch lock {bundle_field} mismatch")
    adapter = read_json(ROBUST_ADAPTER_PATH)
    expected_gate = {
        "adapter_manifest_sha256": file_sha256(ROBUST_ADAPTER_PATH),
        "gate_spec_sha256": adapter["robust_gate"]["gate_spec_sha256"],
        "manifest_sha256": adapter["robust_gate"]["manifest_sha256"],
    }
    if lock.get("frozen_gate") != expected_gate:
        raise ProtocolError("launch lock frozen-gate binding mismatch")


def load_contract() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    campaign, cells, _resolution = make_unfrozen_contract()
    validate_cells(cells, require_resolved=True)
    if not LOCK_PATH.is_file():
        raise ProtocolError("launch lock is missing; support probes must resolve before freeze")
    lock = read_json(LOCK_PATH)
    if (
        lock.get("schema_version") != 2
        or lock.get("campaign_id") != CAMPAIGN_ID
        or lock.get("lock_stage") != "campaign"
    ):
        raise ProtocolError("unexpected launch-lock identity/schema")
    if lock.get("campaign_sha256") != file_sha256(CAMPAIGN_PATH):
        raise ProtocolError("launch lock campaign hash mismatch")
    if lock.get("support_resolution_sha256") != file_sha256(SUPPORT_RESOLUTION_PATH):
        raise ProtocolError("launch lock support-resolution hash mismatch")
    if lock.get("cells_sha256") != canonical_sha256(cells):
        raise ProtocolError("launch lock generated-cell hash mismatch")
    validate_frozen_hashes(lock)
    evidence_paths = support_evidence_paths(read_json(SUPPORT_RESOLUTION_PATH))
    evidence = lock.get("support_evidence_sha256", {})
    if set(evidence) != set(evidence_paths):
        raise ProtocolError("launch lock support-evidence path set mismatch")
    for relative, expected in evidence.items():
        path = REPO_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"frozen support evidence changed: {relative}")
    if lock.get("support_evidence_bundle_sha256") != canonical_sha256(evidence):
        raise ProtocolError("launch lock support-evidence bundle mismatch")
    return campaign, cells, lock
