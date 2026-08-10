#!/usr/bin/env python3
"""Derive admission eligibility and local paired effects from retained v7 evidence."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from ako_runs.controlled_followup.fused_epilogue_crossed_v1.core import exact_median_interval
from ako_runs.controlled_followup.tilelang_abstraction_v2.protocol import classify_interval

from . import artifacts
from .protocol import (
    CAMPAIGN_ID,
    canonical_sha256,
    file_sha256,
    load_lock,
    read_json,
    repo_path,
    stable_write,
    validate_artifact_receipt,
    validate_timing_manifest,
)


def _pair(campaign: dict[str, Any], pair_id: str) -> dict[str, Any]:
    return next(pair for pair in campaign["pairs"] if pair["pair_id"] == pair_id)


def _require_fields(value: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    changed = [key for key, item in expected.items() if value.get(key) != item]
    if changed:
        raise ValueError(f"{label} coordinates differ: {','.join(changed)}")


def _validate_remote_receipt(value: Any, lock_sha: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("remote authorization receipt is missing")
    if (
        value.get("head_commit") != value.get("upstream_commit")
        or not isinstance(value.get("head_commit"), str)
        or not (40 <= len(value["head_commit"]) <= 64)
        or any(char not in "0123456789abcdef" for char in value["head_commit"])
        or value.get("lock_sha256") != lock_sha
        or value.get("live_upstream_commit") != value.get("head_commit")
        or not isinstance(value.get("upstream_remote"), str)
        or not value.get("upstream_remote")
        or not isinstance(value.get("upstream_ref"), str)
        or not value.get("upstream_ref", "").startswith("refs/")
        or value.get("head_equals_configured_upstream") is not True
        or not isinstance(value.get("lock_git_blob"), str)
        or not (40 <= len(value["lock_git_blob"]) <= 64)
        or any(char not in "0123456789abcdef" for char in value["lock_git_blob"])
    ):
        raise ValueError("remote authorization receipt is inconsistent")


def _validate_idle_snapshot(value: Any, phase: str, process_pid: int | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"phase", "checked_at_unix", "self_pid", "compute_processes", "stderr"}
        or value.get("phase") != phase
        or isinstance(value.get("checked_at_unix"), bool)
        or not isinstance(value.get("checked_at_unix"), (int, float))
        or not math.isfinite(float(value["checked_at_unix"]))
        or not isinstance(value.get("self_pid"), int)
        or value["self_pid"] <= 0
        or (process_pid is not None and value["self_pid"] != process_pid)
        or not isinstance(value.get("compute_processes"), list)
        or not isinstance(value.get("stderr"), str)
    ):
        raise ValueError(f"malformed GPU idle receipt at {phase}")
    for row in value["compute_processes"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"pid", "used_memory_mib"}
            or row.get("pid") != value["self_pid"]
            or not isinstance(row.get("used_memory_mib"), str)
        ):
            raise ValueError(f"foreign GPU process at {phase}")
    return value


def _validate_child_window(value: dict[str, Any], prefix: str) -> None:
    pid, parent = value.get("process_pid"), value.get("parent_pid")
    start, end = value.get("t_start"), value.get("t_end")
    if (
        not isinstance(pid, int) or pid <= 0
        or not isinstance(parent, int) or parent <= 0
        or isinstance(start, bool) or not isinstance(start, (int, float))
        or isinstance(end, bool) or not isinstance(end, (int, float))
        or not math.isfinite(float(start)) or not math.isfinite(float(end))
        or not start < end
        or value.get("gpu_idle_postflight_error") is not None
    ):
        raise ValueError(f"{prefix} child execution receipt is malformed")
    pre = _validate_idle_snapshot(value.get("gpu_idle_preflight"), f"{prefix}_child_pre", pid)
    post = _validate_idle_snapshot(value.get("gpu_idle_postflight"), f"{prefix}_child_post", pid)
    if not pre["checked_at_unix"] <= start <= post["checked_at_unix"] <= end:
        raise ValueError(f"{prefix} child idle/timing order differs")


def _arm_coordinates(
    pair: dict[str, Any],
    side: str,
    lock_sha: str,
    lock: dict[str, Any],
    materials: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha,
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "gate_manifest_sha256": materials["entries"][pair["gate"]["manifest_material_id"]]["sha256"],
        "gate_spec_sha256": materials["entries"][pair["gate"]["spec_material_id"]]["sha256"],
        "pair_id": pair["pair_id"],
        "family": pair["family"],
        "side": side,
        "variant": pair[side]["variant"],
        "set": pair[side]["set"],
        "physical_gpu": lock["gpu"]["physical_index"],
        "gpu": lock["gpu"],
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid gate JSONL line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"gate JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def _threshold_failures(gate: dict[str, Any], metrics: Any) -> list[str]:
    if not isinstance(metrics, dict):
        return ["metrics_missing"]
    failures = []
    for name, spec in gate.get("thresholds", {}).items():
        value = metrics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) > float(spec["value"])
        ):
            failures.append(name)
    return failures


def _validate_gate_rows(
    pair: dict[str, Any],
    side: str,
    receipt: dict[str, Any],
    rows: list[dict[str, Any]],
    expected: dict[str, Any],
    materials: dict[str, Any],
) -> dict[str, Any]:
    binding = {
        **expected,
        "created_utc": receipt.get("created_utc"),
        "implementation_sha256": receipt["implementation_sha256"],
        "remote_authorization": receipt.get("remote_authorization"),
    }
    if pair["family"] == "matmul":
        from ako_runs.controlled_followup.robust_gate.seeds import tensor_seeds

        manifest = read_json(repo_path(materials["entries"][pair["gate"]["manifest_material_id"]]["path"]))
        spec = read_json(repo_path(materials["entries"][pair["gate"]["spec_material_id"]]["path"]))
        cases = manifest["operations"]["matmul"]["cases"]
        count = manifest["split_counts"]["validation"]
        expected_coordinates = {
            (case["id"], seed, gate_id)
            for case in cases for seed in range(count) for gate_id in pair["gate"]["gate_ids"]
        }
        observed = set()
        maxima: dict[str, float] = {}
        failed = 0
        for row in rows:
            _require_fields(row, binding, "matmul gate row")
            coordinate = (row.get("case_id"), row.get("seed_index"), row.get("gate_id"))
            if coordinate in observed or coordinate not in expected_coordinates:
                raise ValueError(f"matmul gate coordinate is duplicate/foreign: {coordinate}")
            observed.add(coordinate)
            case_id, seed_index, gate_id = coordinate
            gate = spec["gates"][f"matmul/{gate_id}"]
            failures = _threshold_failures(gate, row.get("metrics"))
            _require_fields(row, {
                "record_type": "tilelang_abstraction_v7_matmul_gate",
                "op": "matmul",
                "tensor_seeds": tensor_seeds(manifest, "matmul", case_id, "validation", seed_index),
                "ok": True,
                "gate_pass": not failures,
                "threshold_failures": failures,
            }, "matmul gate row")
            failed += bool(failures)
            for name, value in row["metrics"].items():
                numeric = float(value)
                maxima[f"{gate_id}/{name}"] = max(maxima.get(f"{gate_id}/{name}", float("-inf")), numeric)
        complete = observed == expected_coordinates and len(rows) == len(expected_coordinates)
        return {
            "complete": complete,
            "expected_records": len(expected_coordinates),
            "observed_records": len(rows),
            "failed_records": failed,
            "full_gate_pass": complete and failed == 0,
            "maxima": maxima,
        }

    adapter = artifacts.exact_robust_adapter()
    recovery = artifacts.exact_recovery_audit()

    context = adapter.load_repository()
    expected_coordinates = {
        (case_id, seed, gate_id)
        for case_id in context.cases for seed in range(64) for gate_id in pair["gate"]["gate_ids"]
    }
    observed = set()
    candidate = f"{CAMPAIGN_ID}:{pair['pair_id']}:{side}"
    for row in rows:
        _require_fields(row, binding, "fused gate row")
        coordinate = (row.get("case_id"), row.get("seed_index"), row.get("gate_id"))
        if coordinate in observed or coordinate not in expected_coordinates:
            raise ValueError(f"fused gate coordinate is duplicate/foreign: {coordinate}")
        observed.add(coordinate)
        case_id, seed_index, gate_id = coordinate
        gate = context.gate_spec["gates"][f"fused_softmax/{gate_id}"]
        failures = adapter.threshold_failures(gate, row.get("metrics", {}))
        _require_fields(row, {
            "record_type": "robust_gate_measurement",
            "op": "fused_softmax",
            "candidate": candidate,
            "split": "validation",
            "device": "cuda:0",
            "shape": pair["shape"],
            "tensor_seeds": adapter.tensor_seeds(
                context.robust_manifest, "fused_softmax", case_id, "validation", seed_index
            ),
            "gate_spec_sha256": expected["gate_spec_sha256"],
            "manifest_sha256": context.manifest_sha256,
            "phase2_config": receipt["metadata"]["config"],
            "ok": True,
            "gate_pass": not failures,
            "threshold_failures": failures,
        }, "fused gate row")
    if observed != expected_coordinates or len(rows) != len(expected_coordinates):
        raise ValueError("fused gate coverage differs")
    return recovery.fixed_gate_summary(context, rows)


def _validate_gate_receipt(
    pair: dict[str, Any],
    side: str,
    receipt_path: Path,
    lock_sha: str,
    lock: dict[str, Any],
    materials: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    receipt = read_json(receipt_path)
    expected = _arm_coordinates(pair, side, lock_sha, lock, materials)
    _require_fields(
        receipt, {**expected, "toolchain": lock["toolchain"]},
        "admission receipt",
    )
    _validate_remote_receipt(receipt.get("remote_authorization"), lock_sha)
    _validate_child_window(receipt, "admission")
    artifact_root = repo_path(receipt.get("artifact_root", ""))
    expected_artifact_root = receipt_path.parent / (receipt_path.stem + "_artifact")
    if artifact_root.resolve() != expected_artifact_root.resolve():
        raise ValueError("admission artifact root differs from canonical position")
    identity = validate_artifact_receipt(
        receipt.get("admitted_cache", {}), expected_root=artifact_root
    )
    source_id = {
        "matmul": "tilelang_matmul_abstraction",
        "fused_softmax": "tilelang_fused_abstraction",
    }[pair["family"]]
    if (
        receipt.get("implementation_sha256") != identity
        or receipt.get("implementation_source_sha256") != materials["entries"][source_id]["sha256"]
    ):
        raise ValueError("admission implementation binding differs")
    gate_path = receipt_path.with_suffix(".gate.jsonl")
    expected_gate_path = str(gate_path.resolve().relative_to(repo_path(".")))
    if (
        receipt.get("gate_path") != expected_gate_path
        or not gate_path.is_file()
        or receipt.get("gate_sha256") != file_sha256(gate_path)
    ):
        raise ValueError("gate JSONL path/hash differs")
    rows = _jsonl(gate_path)
    derived = _validate_gate_rows(pair, side, receipt, rows, expected, materials)
    if (
        receipt.get("gate_summary") != derived
        or receipt.get("terminal_outcome") != "GATE_PASSED"
        or derived.get("full_gate_pass") is not True
    ):
        raise ValueError("gate outcome was not independently re-derived")
    return receipt, identity


def _profile_projection(
    pair: dict[str, Any],
    side: str,
    receipt: dict[str, Any],
    receipt_path: Path,
    lock_sha: str,
    lock: dict[str, Any],
    lock_path: Path,
    expected_identity: str,
) -> dict[str, Any]:
    from .campaign_runner import _ncu_module, _profile_command

    expected = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha,
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "toolchain": lock["toolchain"],
        "pair_id": pair["pair_id"],
        "family": pair["family"],
        "side": side,
        "variant": pair[side]["variant"],
        "set": pair[side]["set"],
        "physical_gpu": lock["gpu"]["physical_index"],
        "gpu": lock["gpu"],
        "returncode": 0,
        "ok": True,
        "expected_artifact_identity_sha256": expected_identity,
        "implementation_sha256": expected_identity,
        "artifact_identity_match": True,
        "artifact_error": None,
    }
    _require_fields(receipt, expected, "profile receipt")
    _validate_remote_receipt(receipt.get("remote_authorization"), lock_sha)
    _validate_child_window(receipt, "profile")
    admission_path = receipt_path.with_name(receipt_path.name.replace(".profile.json", ".json"))
    admission = read_json(admission_path)
    artifact_root = repo_path(admission.get("artifact_root", ""))
    identity = validate_artifact_receipt(
        receipt.get("admitted_cache", {}), expected_root=artifact_root
    )
    if identity != expected_identity:
        raise ValueError("profile executable differs from gated executable")
    raw_path = receipt_path.with_suffix(".raw.json")
    if (
        receipt.get("raw_path") != str(raw_path.resolve().relative_to(repo_path(".")))
        or not raw_path.is_file()
        or file_sha256(raw_path) != receipt.get("raw_sha256")
    ):
        raise ValueError("profile raw path/hash mismatch")
    load_path = receipt_path.with_suffix(".load.json")
    if (
        receipt.get("artifact_load_path") != str(load_path.resolve().relative_to(repo_path(".")))
        or not load_path.is_file()
        or receipt.get("artifact_load_sha256") != file_sha256(load_path)
    ):
        raise ValueError("profile load receipt path/hash differs")
    load_value = read_json(load_path)
    _require_fields(load_value, {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha,
        "pair_id": pair["pair_id"],
        "side": side,
        "admitted_cache_sha256": expected_identity,
    }, "profile load receipt")
    artifacts.validate_load_evidence(load_value.get("load_evidence"), receipt["admitted_cache"])
    if load_value.get("cache_before") != receipt["admitted_cache"] or load_value.get("cache_after") != receipt["admitted_cache"]:
        raise ValueError("profile load receipt cache differs")
    if receipt.get("command") != _profile_command(
        pair, side, lock["gpu"]["physical_index"],
        lock_path,
        admission_path, load_path,
    ):
        raise ValueError("profile command differs")
    raw = read_json(raw_path)
    records = raw.get("records")
    if (
        raw.get("metrics_requested") != _ncu_module().METRICS
        or not isinstance(records, list)
        or len(records) != 1
        or records[0].get("ok") is not True
    ):
        raise ValueError("profile must contain exactly one successful record")
    record = records[0]
    if pair["family"] == "matmul":
        _require_fields(record, {
            "dsl": "tilelang_abs", "variant": pair[side]["variant"],
            "geom": "primary", "set": pair[side]["set"],
        }, "profile raw record")
    else:
        _require_fields(record, {
            "op": "fused", "dsl": "tilelang_abs", "variant": pair[side]["variant"],
            "set": pair[side]["set"],
        }, "profile raw record")
    if pair["family"] == "matmul":
        source = record.get("metrics", {})
        available = {
            "hmma_inst": source.get("hmma_inst"),
            "grid": source.get("grid"),
            "block": source.get("block"),
        }
        resources = {
            "registers": source.get("regs"),
            "shared_bytes": (source.get("smem_static_B") or 0) + (source.get("smem_dyn_B") or 0),
            "dram_read_bytes": source.get("dram_rd_B"),
            "dram_write_bytes": source.get("dram_wr_B"),
        }
    else:
        kernels = [kernel for kernel in record.get("kernels", []) if not kernel.get("is_setup")]
        available = {
            "n_algo_kernels": record.get("n_algo_kernels"),
            "hmma_inst": sum(kernel.get("hmma_inst", 0) or 0 for kernel in kernels),
            "grid": [kernel.get("grid") for kernel in kernels],
            "block": [kernel.get("block") for kernel in kernels],
        }
        resources = {
            "registers": [kernel.get("regs") for kernel in kernels],
            "shared_bytes": [
                (kernel.get("smem_static_B") or 0) + (kernel.get("smem_dyn_B") or 0)
                for kernel in kernels
            ],
            "dram_read_bytes": sum(kernel.get("dram_read_B", 0) or 0 for kernel in kernels),
            "dram_write_bytes": sum(kernel.get("dram_write_B", 0) or 0 for kernel in kernels),
        }
    projection = {field: available.get(field) for field in pair["match"]["profile_equal_fields"]}
    if any(value is None or value == [] for value in projection.values()):
        raise ValueError(f"profile lacks required equality fields: {projection}")
    return {"dynamic_work": projection, "resources": resources, "raw_sha256": receipt["raw_sha256"]}


def _measured_pair_audit(
    pair: dict[str, Any],
    gates: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Audit only observed equality; keep source-review claims as assumptions."""
    mismatches = []
    high, low = profiles["high"]["dynamic_work"], profiles["low"]["dynamic_work"]
    if high != low:
        mismatches.append("dynamic_work")
    if gates["high"]["implementation_sha256"] == gates["low"]["implementation_sha256"]:
        mismatches.append("distinct_treatments")

    threads = pair["match"]["threads"]
    if pair["family"] == "matmul":
        expected_grid = (pair["shape"]["M"] // pair["match"]["tile"][0]) * (
            pair["shape"]["N"] // pair["match"]["tile"][1]
        )
        if high.get("block") != threads or high.get("grid") != expected_grid:
            mismatches.append("registered_launch_geometry")
        if not isinstance(high.get("hmma_inst"), (int, float)) or high["hmma_inst"] <= 0:
            mismatches.append("tensor_core_work")
    else:
        expected_grid = pair["shape"]["M"]
        if (
            high.get("n_algo_kernels") != 1
            or high.get("block") != [threads]
            or high.get("grid") != [expected_grid]
        ):
            mismatches.append("registered_launch_geometry")

    design = {
        key: pair["match"][key]
        for key in ("algorithm", "dtype", "tile", "pipeline_depth", "instruction_family", "logical_work")
    }
    return {
        "pair_id": pair["pair_id"],
        "classification": "runtime_estimand" if not mismatches else "excluded_fail_closed",
        "included_in_runtime_estimand": not mismatches,
        "mismatch_fields": mismatches,
        "verified_fields": [
            "full_current_gate", "same_arm_exact_admitted_executable_load",
            "distinct_treatment_artifacts", "dynamic_work", "launch_geometry",
        ],
        "observed_dynamic_work": {"high": high, "low": low},
        "preregistered_design_assumptions_not_empirically_verified": design,
    }


def admission_artifact_hashes(root: Path, campaign: dict[str, Any]) -> dict[str, str]:
    """Return the exact pre-summary admission census, rejecting foreign files."""
    root = root.resolve()
    allowed = {root / "launch_receipt.json"}
    required = set(allowed)
    for pair in campaign["pairs"]:
        pair_id = pair["pair_id"]
        if not pair["gate"]["available"]:
            path = root / f"{pair_id}__unavailable.json"
            allowed.add(path)
            required.add(path)
            continue
        for side in ("high", "low"):
            receipt_path = root / f"{pair_id}__{side}.json"
            gate_path = root / f"{pair_id}__{side}.gate.jsonl"
            profile_path = root / f"{pair_id}__{side}.profile.json"
            raw_path = root / f"{pair_id}__{side}.profile.raw.json"
            load_path = root / f"{pair_id}__{side}.profile.load.json"
            artifact_root = root / f"{pair_id}__{side}_artifact"
            allowed.update({receipt_path, gate_path, profile_path, raw_path, load_path})
            required.add(receipt_path)
            if receipt_path.is_file():
                receipt = read_json(receipt_path)
                if receipt.get("gate_path") is not None:
                    required.add(gate_path)
                if receipt.get("terminal_outcome") == "GATE_PASSED":
                    required.add(profile_path)
                retained = {path.resolve() for path in artifact_root.rglob("*") if path.is_file()}
                allowed.update(retained)
                cache = receipt.get("admitted_cache")
                if isinstance(cache, dict):
                    cache_root = repo_path(cache.get("root", ""))
                    expected_cache = {
                        (cache_root / row["path"]).resolve()
                        for row in cache.get("files", []) if isinstance(row, dict) and isinstance(row.get("path"), str)
                    }
                    if retained != expected_cache:
                        raise ValueError(f"admitted cache/temp census differs: {artifact_root}")
            if profile_path.is_file():
                profile = read_json(profile_path)
                if profile.get("raw_path") is not None:
                    required.add(raw_path)
                if profile.get("artifact_load_path") is not None:
                    required.add(load_path)
    actual = {
        path.resolve() for path in root.rglob("*") if path.is_file()
        and path.resolve() not in {root / "run_status.json", root / "summary.json"}
    }
    if not required <= actual:
        raise ValueError("required admission artifact is missing")
    if actual - allowed:
        raise ValueError(f"unexpected admission artifacts: {sorted(str(path) for path in actual - allowed)}")
    return dict(sorted(
        (str(path.relative_to(repo_path("."))), file_sha256(path)) for path in actual
    ))


def _validate_admission_stage(
    root: Path,
    campaign: dict[str, Any],
    lock: dict[str, Any],
    lock_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    from .campaign_runner import GPU_LOCK_ID

    launch_path, status_path = root / "launch_receipt.json", root / "run_status.json"
    launch = read_json(launch_path) if launch_path.is_file() else {}
    expected_launch = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha,
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "toolchain": lock["toolchain"],
        "expected_pair_ids": [pair["pair_id"] for pair in campaign["pairs"]],
        "physical_gpu": lock["gpu"]["physical_index"],
        "gpu": lock["gpu"],
        "gpu_lock_id": GPU_LOCK_ID,
    }
    _require_fields(launch, expected_launch, "admission launch receipt")
    _validate_remote_receipt(launch.get("remote_authorization"), lock_sha)
    pre = _validate_idle_snapshot(launch.get("gpu_idle_preflight"), "admission_stage_pre")
    hashes = admission_artifact_hashes(root, campaign)
    status = read_json(status_path) if status_path.is_file() else {}
    expected_status = {
        "schema_version": 1,
        "record_type": "tilelang_abstraction_v7_admission_run_status",
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha,
        "complete": True,
        "artifact_sha256": hashes,
        "artifact_bundle_sha256": canonical_sha256(hashes),
        "launch_receipt_sha256": file_sha256(launch_path),
    }
    _require_fields(status, expected_status, "admission run status")
    post = _validate_idle_snapshot(status.get("gpu_idle_postflight"), "admission_stage_post")
    if pre["self_pid"] != post["self_pid"] or pre["checked_at_unix"] > post["checked_at_unix"]:
        raise ValueError("admission parent execution receipt differs")
    return launch, status, hashes


def admission_summary(root: Path, lock_path: Path) -> dict[str, Any]:
    campaign, materials, lock = load_lock(lock_path)
    lock_sha = file_sha256(lock_path)
    launch_path = root / "launch_receipt.json"
    launch, status, evidence = _validate_admission_stage(root, campaign, lock, lock_sha)
    evidence = dict(evidence)
    status_path = root / "run_status.json"
    evidence[str(status_path.resolve().relative_to(repo_path(".")))] = file_sha256(status_path)
    rows = []
    for pair in campaign["pairs"]:
        pair_id = pair["pair_id"]
        reasons: list[str] = []
        if not pair["gate"]["available"]:
            unavailable = root / f"{pair_id}__unavailable.json"
            value = read_json(unavailable) if unavailable.is_file() else {}
            if unavailable.is_file():
                evidence[str(unavailable.resolve().relative_to(repo_path(".")))] = file_sha256(unavailable)
            if any((
                value.get("campaign_id") != CAMPAIGN_ID,
                value.get("campaign_lock_sha256") != lock_sha,
                value.get("pair_id") != pair_id,
                value.get("terminal_outcome") != "CURRENT_GATE_UNAVAILABLE",
                value.get("reason") != pair["gate"]["reason"],
                value.get("build_attempted") is not False,
                value.get("timing_authorized") is not False,
            )):
                reasons.append("missing_or_changed_unavailable-gate receipt")
            rows.append({
                "pair_id": pair_id, "family": pair["family"], "timing_eligible": False,
                "classification": "blocked_no_current_gate", "reasons": reasons or [pair["gate"]["reason"]],
                "gate_available": False,
            })
            continue
        gates, profiles = {}, {}
        for side in ("high", "low"):
            gate_path = root / f"{pair_id}__{side}.json"
            profile_path = root / f"{pair_id}__{side}.profile.json"
            if not gate_path.is_file():
                reasons.append(f"{side}: admission receipt missing")
                continue
            evidence[str(gate_path.resolve().relative_to(repo_path(".")))] = file_sha256(gate_path)
            if profile_path.is_file():
                evidence[str(profile_path.resolve().relative_to(repo_path(".")))] = file_sha256(profile_path)
            try:
                gate, identity = _validate_gate_receipt(
                    pair, side, gate_path, lock_sha, lock, materials
                )
                profile_receipt = read_json(profile_path)
                projection = _profile_projection(
                    pair, side, profile_receipt, profile_path, lock_sha, lock, lock_path, identity
                )
                retained = [
                    gate_path,
                    gate_path.with_suffix(".gate.jsonl"),
                    profile_path,
                    profile_path.with_suffix(".raw.json"),
                    profile_path.with_suffix(".load.json"),
                ]
                cache_root = repo_path(gate["admitted_cache"]["root"])
                retained.extend(cache_root / value["path"] for value in gate["admitted_cache"]["files"])
                for retained_path in retained:
                    evidence[str(retained_path.resolve().relative_to(repo_path(".")))] = file_sha256(retained_path)
                gates[side], profiles[side] = gate, projection
            except Exception as exc:
                reasons.append(f"{side}: {exc}")
        audit = None
        if set(gates) == {"high", "low"} and set(profiles) == {"high", "low"}:
            try:
                audit = _measured_pair_audit(pair, gates, profiles)
                if audit["included_in_runtime_estimand"] is not True:
                    reasons.append("measured match audit excluded: " + ",".join(audit["mismatch_fields"]))
            except Exception as exc:
                reasons.append(f"measured pair audit failed: {exc}")
        eligible = not reasons and audit is not None and audit["included_in_runtime_estimand"] is True
        rows.append({
            "pair_id": pair_id, "family": pair["family"], "gate_available": True,
            "timing_eligible": eligible,
            "classification": "runtime_estimand" if eligible else "excluded_fail_closed",
            "reasons": reasons, "measured_match_audit": audit,
            "artifact_identity_sha256": {
                side: gates[side]["admitted_cache"]["files_sha256"] for side in gates
            },
            "arm_receipts": {
                side: {
                    "admission_path": str((root / f"{pair_id}__{side}.json").resolve().relative_to(repo_path("."))),
                    "admission_sha256": file_sha256(root / f"{pair_id}__{side}.json"),
                    "profile_path": str((root / f"{pair_id}__{side}.profile.json").resolve().relative_to(repo_path("."))),
                    "profile_sha256": file_sha256(root / f"{pair_id}__{side}.profile.json"),
                }
                for side in gates
            },
        })
    eligible = [row["pair_id"] for row in rows if row["timing_eligible"]]
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "record_type": "tilelang_abstraction_v7_admission_summary",
        "campaign_lock_sha256": lock_sha,
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "launch_receipt_path": str(launch_path.resolve().relative_to(repo_path("."))),
        "launch_receipt_sha256": file_sha256(launch_path),
        "run_status_path": str(status_path.resolve().relative_to(repo_path("."))),
        "run_status_sha256": file_sha256(status_path),
        "complete": len(rows) == len(campaign["pairs"]),
        "material_receipts_verified": True,
        "pairs": rows,
        "timing_eligible_pair_ids": eligible,
        "census": {
            "requested_pairs": len(rows),
            "current_gate_unavailable": sum(not row["gate_available"] for row in rows),
            "timing_eligible": len(eligible),
            "excluded": len(rows) - len(eligible),
        },
        "claim_scope": "local_pair_only",
        "generalization_design_ready": False,
        "evidence_sha256": dict(sorted(evidence.items())),
        "evidence_bundle_sha256": canonical_sha256(dict(sorted(evidence.items()))),
    }


def load_verified_admission(path: Path, lock_path: Path) -> dict[str, Any]:
    """Load only a summary that exactly re-derives from its retained evidence."""
    if path.name != "summary.json" or not path.is_file():
        raise ValueError("timing requires the retained admission/summary.json")
    supplied = read_json(path)
    derived = admission_summary(path.parent, lock_path)
    campaign, _materials, _lock = load_lock(lock_path)
    expected = set(admission_artifact_hashes(path.parent, campaign)) | {
        str((path.parent / "run_status.json").resolve().relative_to(repo_path("."))),
        str(path.resolve().relative_to(repo_path("."))),
    }
    observed = {
        str(item.resolve().relative_to(repo_path(".")))
        for item in path.parent.rglob("*") if item.is_file()
    }
    if observed != expected:
        raise ValueError("admission artifact census differs after summary creation")
    if supplied != derived:
        raise ValueError("admission summary differs from independently re-derived evidence")
    return derived


def timing_artifact_hashes(stage_root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Return the exact pre-summary timing census, including ordered receipts."""
    stage_root = stage_root.resolve()
    raw_root, position_root = stage_root / "raw", stage_root / "position_receipts"
    expected = {stage_root / "launch_receipt.json"}
    for index, row in enumerate(manifest["rows"], 1):
        stem = f"{index:04d}__{row['row_id']}"
        expected.add(raw_root / f"{stem}.json")
        expected.add(position_root / f"{stem}.json")
    actual = {
        path.resolve() for path in stage_root.rglob("*") if path.is_file()
        and path.resolve() not in {stage_root / "run_status.json", stage_root / "summary.json"}
    }
    if actual != expected:
        raise ValueError("timing artifact census differs from the exact manifest")
    return dict(sorted(
        (str(path.relative_to(repo_path("."))), file_sha256(path)) for path in actual
    ))


def _validate_position_receipt(
    position: dict[str, Any],
    record: dict[str, Any],
    row: dict[str, Any],
    path: Path,
    raw_path: Path,
    manifest_sha: str,
    global_position: int,
    predecessor_sha: str | None,
    previous_completed: int,
    parent_pid: int,
) -> int:
    launched = position.get("child_launched_unix_ns")
    completed = position.get("child_completed_unix_ns")
    expected = {
        "schema_version": 1,
        "record_type": "tilelang_abstraction_v7_position_receipt",
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": manifest_sha,
        "row_id": row["row_id"],
        "global_position": global_position,
        "predecessor_position_receipt_sha256": predecessor_sha,
        "child_pid": record["process_pid"],
        "returncode": 0,
        "raw_path": str(raw_path.resolve().relative_to(repo_path("."))),
        "raw_sha256": file_sha256(raw_path),
        "gpu_postflight_error": None,
    }
    _require_fields(position, expected, "timing position receipt")
    idle = _validate_idle_snapshot(
        position.get("gpu_idle_after_child"), "timing_parent_after_child", parent_pid
    )
    if (
        not isinstance(launched, int) or not isinstance(completed, int)
        or launched < previous_completed or launched >= completed
        or position.get("child_pid") != record.get("process_pid")
        or not launched <= int(record["t_start"] * 1_000_000_000)
        <= int(record["t_end"] * 1_000_000_000) <= completed
        or idle["checked_at_unix"] < record["t_end"]
        or path.name != f"{global_position:04d}__{row['row_id']}.json"
    ):
        raise ValueError(f"timing position/order receipt differs at {global_position}")
    return completed


def _verify_records(
    raw_root: Path,
    manifest: dict[str, Any],
    admission: dict[str, Any],
    campaign: dict[str, Any],
    lock: dict[str, Any],
    lock_sha: str,
    lock_path: Path,
    manifest_path: Path,
    admission_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    from .campaign_runner import _arm_time_command

    paths = sorted(raw_root.glob("*.json"))
    if len(paths) != len(manifest["rows"]):
        raise ValueError(f"timing record census {len(paths)} != {len(manifest['rows'])}")
    expected_by_name = {
        f"{index:04d}__{row['row_id']}.json": row
        for index, row in enumerate(manifest["rows"], 1)
    }
    if {path.name for path in paths} != set(expected_by_name):
        raise ValueError("timing record filenames differ from the exact manifest order")
    records, hashes = {}, {}
    for path in paths:
        record = read_json(path)
        row = record.get("manifest_row")
        row_id = row.get("row_id") if isinstance(row, dict) else None
        if (
            record.get("ok") is not True
            or record.get("campaign_lock_sha256") != lock_sha
            or record.get("manifest_sha256") != canonical_sha256(manifest)
            or record.get("manifest_row_sha256") != canonical_sha256(row)
            or row not in manifest["rows"]
            or row != expected_by_name[path.name]
            or row_id in records
        ):
            raise ValueError(f"foreign, failed, or duplicate timing record: {path}")
        times = record.get("times_ms", [])
        if (
            len(times) != row["trials"]
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
                for value in times
            )
        ):
            raise ValueError(f"timing trials are malformed: {path}")
        start = campaign["inference"]["primary_trial_start"]
        stop = campaign["inference"]["primary_trial_stop"]
        expected_summaries = {
            "primary_tail_median_ms": statistics.median(times[start:stop]),
            "full_median_ms": statistics.median(times),
            "first_decile_median_ms": statistics.median(times[:10]),
            "last_decile_median_ms": statistics.median(times[-10:]),
        }
        pair = _pair(campaign, row["pair_id"])
        admission_pair = next(value for value in admission["pairs"] if value["pair_id"] == row["pair_id"])
        expected_identity = admission_pair["artifact_identity_sha256"][row["implementation_side"]]
        expected_fields = {
            **expected_summaries,
            "campaign_id": CAMPAIGN_ID,
            "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
            "toolchain": lock["toolchain"],
            "physical_gpu": lock["gpu"]["physical_index"],
            "gpu": lock["gpu"],
            "returncode": 0,
            "command": _arm_time_command(
                row, lock["gpu"]["physical_index"], lock_path,
                manifest_path, admission_path, path,
            ),
            "cwd": str(repo_path(".").resolve()),
            "expected_artifact_identity_sha256": expected_identity,
            "implementation_sha256": expected_identity,
            "artifact_identity_match": True,
            "artifact_error": None,
        }
        _require_fields(record, expected_fields, "timing record")
        _validate_remote_receipt(record.get("remote_authorization"), lock_sha)
        _validate_child_window(record, "timing")
        gate_path = admission_path.parent / f"{row['pair_id']}__{row['implementation_side']}.json"
        gate = read_json(gate_path)
        artifact_root = repo_path(gate.get("artifact_root", ""))
        if record.get("admitted_cache") != gate.get("admitted_cache"):
            raise ValueError(f"timing cache receipt differs from admission: {path}")
        identity = validate_artifact_receipt(
            record.get("admitted_cache", {}), expected_root=artifact_root
        )
        artifacts.validate_load_evidence(
            record.get("artifact_load"), record["admitted_cache"]
        )
        if identity != expected_identity:
            raise ValueError(f"timed executable differs from admitted executable: {path}")
        if not isinstance(record.get("t_start"), (int, float)) or not isinstance(record.get("t_end"), (int, float)) or record["t_end"] < record["t_start"]:
            raise ValueError(f"timing clock receipt is malformed: {path}")
        if record.get("primary_tail_median_ms") != statistics.median(times[start:stop]):
            raise ValueError(f"timing trials/summary changed: {path}")
        records[row_id] = record
        hashes[str(path.resolve().relative_to(repo_path(".")))] = file_sha256(path)
    if set(records) != {row["row_id"] for row in manifest["rows"]}:
        raise ValueError("timing rows do not exactly cover the manifest")
    return records, hashes


def timing_summary(raw_root: Path, manifest_path: Path, admission_path: Path, lock_path: Path) -> dict[str, Any]:
    from .campaign_runner import GPU_LOCK_ID

    campaign, _materials, lock = load_lock(lock_path)
    manifest = read_json(manifest_path)
    admission = load_verified_admission(admission_path, lock_path)
    lock_sha = file_sha256(lock_path)
    validate_timing_manifest(manifest, campaign, lock_sha, admission)
    launch_path = raw_root.parent / "launch_receipt.json"
    launch = read_json(launch_path) if launch_path.is_file() else {}
    if (
        launch.get("campaign_id") != CAMPAIGN_ID
        or launch.get("campaign_lock_sha256") != lock_sha
        or launch.get("dependency_bundle_sha256") != lock["dependency_bundle_sha256"]
        or launch.get("toolchain") != lock["toolchain"]
        or launch.get("manifest_sha256") != canonical_sha256(manifest)
        or launch.get("manifest_file_sha256") != file_sha256(manifest_path)
        or launch.get("admission_summary_sha256") != canonical_sha256(admission)
        or launch.get("admission_summary_file_sha256") != file_sha256(admission_path)
        or launch.get("expected_records") != len(manifest["rows"])
        or launch.get("physical_gpu") != lock["gpu"]["physical_index"]
        or launch.get("gpu") != lock["gpu"]
        or launch.get("gpu_lock_id") != GPU_LOCK_ID
    ):
        raise ValueError("timing launch receipt is missing or foreign")
    _validate_remote_receipt(launch.get("remote_authorization"), lock_sha)
    stage_pre = _validate_idle_snapshot(launch.get("gpu_idle_preflight"), "timing_stage_pre")
    records, _record_hashes = _verify_records(
        raw_root, manifest, admission, campaign, lock, lock_sha,
        lock_path, manifest_path, admission_path,
    )
    stage_root = raw_root.parent
    hashes = timing_artifact_hashes(stage_root, manifest)
    status_path = stage_root / "run_status.json"
    status = read_json(status_path) if status_path.is_file() else {}
    expected_status = {
        "schema_version": 1,
        "record_type": "tilelang_abstraction_v7_timing_run_status",
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": lock_sha,
        "complete": True,
        "expected_records": len(manifest["rows"]),
        "observed_records": len(manifest["rows"]),
        "artifact_sha256": hashes,
        "artifact_bundle_sha256": canonical_sha256(hashes),
        "launch_receipt_sha256": file_sha256(launch_path),
    }
    _require_fields(status, expected_status, "timing run status")
    stage_post = _validate_idle_snapshot(status.get("gpu_idle_postflight"), "timing_stage_post")
    if stage_pre["self_pid"] != stage_post["self_pid"] or stage_pre["checked_at_unix"] > stage_post["checked_at_unix"]:
        raise ValueError("timing parent execution receipt differs")
    predecessor_sha, previous_completed = None, 0
    child_pids: set[int] = set()
    for index, row in enumerate(manifest["rows"], 1):
        raw_path = raw_root / f"{index:04d}__{row['row_id']}.json"
        position_path = stage_root / "position_receipts" / f"{index:04d}__{row['row_id']}.json"
        record = records[row["row_id"]]
        if record.get("parent_pid") != stage_pre["self_pid"]:
            raise ValueError("timing child parent PID differs from the stage launcher")
        if record["process_pid"] in child_pids:
            raise ValueError("timing records reused a child process")
        child_pids.add(record["process_pid"])
        previous_completed = _validate_position_receipt(
            read_json(position_path), record, row, position_path, raw_path,
            canonical_sha256(manifest), index, predecessor_sha, previous_completed,
            stage_pre["self_pid"],
        )
        predecessor_sha = file_sha256(position_path)
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in manifest["rows"]:
        grouped.setdefault((row["pair_id"], row["distribution"], row["block"]), {})[row["role"]] = records[row["row_id"]]
    effects = []
    for pair_id in manifest["eligible_pair_ids"]:
        pair = _pair(campaign, pair_id)
        for distribution in campaign["inference"]["distributions"]:
            blocks = [grouped[(pair_id, distribution, index)] for index in range(campaign["inference"]["confirmation_blocks"])]
            if any(set(block) != {"high", "low", "sham_a", "sham_b"} for block in blocks):
                raise ValueError("incomplete four-record timing block")
            if any(len({
                block["sham_a"]["implementation_sha256"],
                block["sham_b"]["implementation_sha256"],
                block["high"]["implementation_sha256"],
            }) != 1 for block in blocks):
                raise ValueError("sham labels do not bind the byte-identical high implementation")
            ratios = [block["low"]["primary_tail_median_ms"] / block["high"]["primary_tail_median_ms"] for block in blocks]
            shams = [block["sham_b"]["primary_tail_median_ms"] / block["sham_a"]["primary_tail_median_ms"] for block in blocks]
            interval, sham_interval = exact_median_interval(ratios), exact_median_interval(shams)
            floor = max(abs(math.log(sham_interval["ci_lo"])), abs(math.log(sham_interval["ci_hi"])))
            classification = classify_interval(
                math.log(interval["ci_lo"]), math.log(interval["ci_hi"]),
                delta_hw=campaign["inference"]["delta_hw_log_ratio"],
                epsilon=campaign["inference"]["equivalence_log_ratio"],
            )
            clears_floor = math.log(interval["ci_lo"]) > floor or math.log(interval["ci_hi"]) < -floor
            if not clears_floor:
                classification["direction"] = "unresolved_below_sham_resolution"
            if floor > campaign["inference"]["equivalence_log_ratio"]:
                classification["equivalence"] = "not_testable_sham_floor_exceeds_equivalence_bound"
            full = exact_median_interval(
                block["low"]["full_median_ms"] / block["high"]["full_median_ms"] for block in blocks
            )
            drift = exact_median_interval(
                (block["low"]["last_decile_median_ms"] / block["low"]["first_decile_median_ms"])
                / (block["high"]["last_decile_median_ms"] / block["high"]["first_decile_median_ms"])
                for block in blocks
            )
            effects.append({
                "pair_id": pair_id, "family": pair["family"], "distribution": distribution,
                "estimand": "T_low/T_high", "primary_settled_tail_interval": interval,
                "sham_interval": sham_interval, "sham_resolution_floor_log": floor,
                "clears_sham_resolution_floor": clears_floor, "classification": classification,
                "reportable_direction": clears_floor and classification["direction"] in {"lower_level_faster", "higher_level_faster"},
                "full_window_diagnostic_interval": full,
                "relative_drift_diagnostic_interval": drift,
            })
    return {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "record_type": "tilelang_abstraction_v7_timing_summary",
        "campaign_lock_sha256": lock_sha, "manifest_sha256": canonical_sha256(manifest),
        "admission_summary_sha256": canonical_sha256(admission),
        "complete": True, "records": len(records), "effects": effects,
        "claim_scope": "local_pair_only", "generalization_design_ready": False,
        "legacy_results_controlling": False,
        "launch_receipt_sha256": file_sha256(launch_path),
        "run_status_sha256": file_sha256(status_path),
        "evidence_sha256": hashes, "evidence_bundle_sha256": canonical_sha256(hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admission", "timing"))
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = (
        admission_summary(args.root, args.lock)
        if args.action == "admission"
        else timing_summary(args.root, args.manifest, args.admission, args.lock)
    )
    stable_write(args.output, result)
    print(json.dumps({"complete": result["complete"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
