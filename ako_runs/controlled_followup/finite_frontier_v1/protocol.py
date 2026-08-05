#!/usr/bin/env python3
"""CPU-only, fail-closed contracts for finite-frontier experiments F0/F1."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STRATEGIES = (
    "register_fused",
    "smem_staged",
    "global_intermediate",
    "register_common_postprocess",
)
LANES = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
GRID_IDS = tuple(f"g{index:02d}" for index in range(19))
F0_STAGES = ("support_probe", "build", "setup", "launch", "gate")
TERMINAL_STATUS = (
    "UNSUPPORTED",
    "BUILD_FAILED",
    "LAUNCH_FAILED",
    "GATE_FAILED",
    "GATE_PASSED",
)
REQUIRED_CLOSURE_ROLES = (
    "hardware_identity_capture",
    "toolchain_capture",
    "support_probe",
    "gate_lock",
    "runner",
    "dependency_inventory",
)
REQUIRED_TOOLCHAIN_KEYS = {"python", "torch", "triton", "tilelang", "nvcc"}


class ProtocolError(RuntimeError):
    pass


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"hardware binding requires non-empty {field}")
    return value.strip()


def _compute_capability(value: Any) -> tuple[int, int]:
    capability: tuple[int, int] | None = None
    if isinstance(value, str):
        match = re.fullmatch(r"(?:sm_)?(\d+)(?:\.(\d+))?", value.strip().lower())
        if match:
            major, minor = match.groups()
            if minor is not None:
                capability = int(major), int(minor)
            elif len(major) == 2:
                capability = int(major[0]), int(major[1])
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        capability = value[0], value[1]
    if capability is None or capability[0] <= 0 or not 0 <= capability[1] <= 9:
        raise ProtocolError("gpu_identity.compute_capability must be explicit, e.g. '9.0' or 'sm_90'")
    return capability


def _load_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must contain a JSON object")
    return value


def _verified_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ProtocolError(f"closure path must not be a symlink: {relative}")
    path = candidate.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"closure path escapes source_root: {relative}") from exc
    if not path.is_file():
        raise ProtocolError(f"closure path is not a regular file: {relative}")
    return path


def validate_hardware_binding(binding: Any, source_root: Path) -> dict[str, Any]:
    if not isinstance(binding, dict) or binding.get("schema_version") != 1:
        raise ProtocolError("hardware binding schema_version must be 1")
    gpu = binding.get("gpu_identity")
    if not isinstance(gpu, dict):
        raise ProtocolError("hardware binding requires gpu_identity")
    for field in ("uuid", "name", "driver_version"):
        _nonempty_text(gpu.get(field), f"gpu_identity.{field}")
    capability = _compute_capability(gpu.get("compute_capability"))
    if capability == (8, 9):
        raise ProtocolError("F0 requires an explicitly non-sm_89 GPU")

    toolchain = binding.get("toolchain")
    if not isinstance(toolchain, dict) or not REQUIRED_TOOLCHAIN_KEYS <= set(toolchain):
        raise ProtocolError("hardware binding requires non-empty toolchain metadata")
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(version, str)
        or not version.strip()
        for name, version in toolchain.items()
    ):
        raise ProtocolError("toolchain must map names to non-empty version strings")
    sources = binding.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ProtocolError("hardware binding requires source_sha256 entries")
    if any(
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for path, digest in sources.items()
    ):
        raise ProtocolError("source_sha256 must map paths to lowercase SHA-256 digests")
    roles = binding.get("closure_roles")
    if not isinstance(roles, dict) or set(roles) != set(REQUIRED_CLOSURE_ROLES):
        raise ProtocolError("closure_roles must bind identity, toolchain, probe, gate, runner, and inventory")
    if len(set(roles.values())) != len(roles):
        raise ProtocolError("closure roles must bind distinct files")
    for role, relative in roles.items():
        if relative not in sources:
            raise ProtocolError(f"closure role is absent from source_sha256: {role}")
    for relative, expected in sources.items():
        actual = hashlib.sha256(_verified_path(source_root, relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ProtocolError(f"source hash mismatch: {relative}")
    identity = _load_object(_verified_path(source_root, roles["hardware_identity_capture"]), "identity capture")
    captured_gpu = identity.get("gpu_identity")
    if captured_gpu != gpu:
        raise ProtocolError("hardware identity does not match its capture")
    captured_toolchain = _load_object(_verified_path(source_root, roles["toolchain_capture"]), "toolchain capture")
    if captured_toolchain.get("toolchain") != toolchain:
        raise ProtocolError("toolchain metadata does not match its capture")
    inventory = _load_object(_verified_path(source_root, roles["dependency_inventory"]), "dependency inventory")
    expected_inventory = sorted(path for path in sources if path != roles["dependency_inventory"])
    if inventory.get("source_paths") != expected_inventory:
        raise ProtocolError("dependency inventory does not match source_sha256")
    return binding


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_f0_manifest(binding: Any, source_root: Path) -> dict[str, Any]:
    binding = validate_hardware_binding(binding, source_root)
    rows = [
        {
            "cell_id": f"{strategy}.{lane}.{grid_id}",
            "strategy": strategy,
            "lane": lane,
            "grid_id": grid_id,
            "timing_allowed": False,
        }
        for strategy in STRATEGIES
        for lane in LANES
        for grid_id in GRID_IDS
    ]
    if (
        len(rows) != 304
        or len({row["cell_id"] for row in rows}) != 304
        or Counter(row["strategy"] for row in rows) != {item: 76 for item in STRATEGIES}
        or Counter(row["lane"] for row in rows) != {item: 76 for item in LANES}
        or Counter(row["grid_id"] for row in rows) != {item: 16 for item in GRID_IDS}
    ):
        raise ProtocolError("F0 factorial census changed")
    return {
        "schema_version": 1,
        "experiment": "F0",
        "purpose": "cross_architecture_feasibility_only",
        "status": "design_only_requested_cells",
        "hardware_binding_sha256": _canonical_sha256(binding),
        "closure_roles": binding["closure_roles"],
        "allowed_stages": list(F0_STAGES),
        "timing_allowed": False,
        "requested_cells": 304,
        "rows": rows,
    }


def validate_f0_results(
    binding: dict[str, Any],
    source_root: Path,
    manifest: dict[str, Any],
    results: Any,
    evidence_root: Path,
) -> dict[str, int]:
    expected_manifest = make_f0_manifest(binding, source_root)
    if manifest != expected_manifest:
        raise ProtocolError("F0 manifest differs from the verified hardware-bound factorial")
    if not isinstance(results, list) or len(results) != 304:
        raise ProtocolError("F0 requires exactly 304 terminal results")
    expected = {row["cell_id"] for row in manifest["rows"]}
    observed: set[str] = set()
    evidence_paths: set[str] = set()
    evidence_hashes: set[str] = set()
    counts = Counter()
    allowed = {"cell_id", "terminal_status", "evidence_path", "evidence_sha256"}
    for row in results:
        if not isinstance(row, dict) or set(row) != allowed:
            raise ProtocolError("F0 result contains missing, extra, or timing fields")
        if row["cell_id"] not in expected or row["cell_id"] in observed:
            raise ProtocolError("F0 result has an unknown or duplicate cell")
        if row["terminal_status"] not in TERMINAL_STATUS:
            raise ProtocolError("F0 result has an unknown terminal status")
        if not isinstance(row["evidence_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"]) is None:
            raise ProtocolError("F0 result evidence_sha256 is invalid")
        if not isinstance(row["evidence_path"], str) or not row["evidence_path"]:
            raise ProtocolError("F0 result evidence_path is invalid")
        if row["evidence_path"] in evidence_paths or row["evidence_sha256"] in evidence_hashes:
            raise ProtocolError("F0 terminal evidence must be unique per cell")
        receipt_path = _verified_path(evidence_root, row["evidence_path"])
        actual_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        if actual_hash != row["evidence_sha256"]:
            raise ProtocolError("F0 terminal evidence hash mismatch")
        receipt = _load_object(receipt_path, "F0 terminal receipt")
        expected_receipt = {
            "schema_version": 1,
            "record_type": "finite_frontier_f0_terminal_receipt",
            "cell_id": row["cell_id"],
            "terminal_status": row["terminal_status"],
            "hardware_binding_sha256": manifest["hardware_binding_sha256"],
            "timing_allowed": False,
        }
        if receipt != expected_receipt:
            raise ProtocolError("F0 terminal receipt content mismatch")
        observed.add(row["cell_id"])
        evidence_paths.add(row["evidence_path"])
        evidence_hashes.add(row["evidence_sha256"])
        counts[row["terminal_status"]] += 1
    if observed != expected:
        raise ProtocolError("F0 result census differs from the manifest")
    return {status: counts[status] for status in TERMINAL_STATUS}


def f1_design_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "F1",
        "status": "design_only_blocked",
        "required_authorization": "new explicit post-paper authorization",
        "required_material_inputs": [
            "frozen_task_corpus_with_task_units",
            "candidate_count_by_task_unit",
            "hardware_bindings",
            "candidate_independent_gate_lock",
            "toolchain_and_source_lock",
            "selection_confirm_blocks",
            "terminal_confirm_blocks",
            "sham_blocks_and_distributions",
        ],
        "stage_formulas": {
            "audit": "lane_count * sum(candidate_count[u] * architecture_count[u] for u in task_units)",
            "screen": "2 * gate_legal_candidate_lane_architecture_rows",
            "selection_confirm": "selection_confirm_blocks * sum(min(3, gate_legal_candidates[stratum]))",
            "terminal_confirm": "terminal_confirm_blocks * locked_winner_strata",
            "sham": "sham_blocks * architecture_count * sham_distribution_count * 2_byte_identical_labels",
            "total": "audit + screen + selection_confirm + terminal_confirm + sham",
        },
    }


def authorize_launch(experiment: str) -> None:
    if experiment == "F1":
        raise ProtocolError("F1 is design-only until new authorization and all frozen materials exist")
    raise ProtocolError("this CPU-only protocol generates contracts and never launches GPU work")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    f0 = subparsers.add_parser("f0-manifest")
    f0.add_argument("hardware_binding", type=Path)
    f0.add_argument("--source-root", type=Path, required=True)
    subparsers.add_parser("f1-contract")
    args = parser.parse_args(argv)
    try:
        result = (
            make_f0_manifest(_read_json(args.hardware_binding), args.source_root)
            if args.command == "f0-manifest"
            else f1_design_contract()
        )
    except ProtocolError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
