#!/usr/bin/env python3
"""CPU-only contract for a four-device Ada feasibility replication."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "ada_device_replication_v1"
INSTRUMENT_CAMPAIGN_ID = "fused-epilogue-crossed-v2"
REFERENCE_RESULT_TAG = "crossed_v2r3"
# Compatibility for the outer execution scaffold: this is an instrument ID, not a result tag.
UNDERLYING_CAMPAIGN_ID = INSTRUMENT_CAMPAIGN_ID
STATE = "design_only_not_authorized"
CLAIM_SCOPE = "noncontrolling_same_host_same_sku_device_feasibility_reproducibility"
GPU_NAME = "NVIDIA RTX 6000 Ada Generation"
COMPUTE_CAPABILITY = "8.9"
GPU_UUIDS = (
    "GPU-45af34ad-0c74-74d0-ef3a-652090d837ae",
    "GPU-91b61ae6-d21e-485e-43b3-7505d62149b1",
    "GPU-3ed448f2-f23a-09fc-e18b-f580523c4a3f",
    "GPU-eafdd6ce-8857-40fd-f494-47a7240bf6b5",
)
TAGS = tuple(f"{CAMPAIGN_ID}_gpu{index}" for index in range(len(GPU_UUIDS)))
STRATEGIES = (
    "register_fused",
    "smem_staged",
    "global_intermediate",
    "register_common_postprocess",
)
LANES = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
GRID_IDS = tuple(f"g{index:02d}" for index in range(19))
ALLOWED_STAGES = ("support_probe", "build", "setup", "launch", "gate")
TERMINAL_STATUSES = (
    "UNSUPPORTED",
    "BUILD_FAILED",
    "LAUNCH_FAILED",
    "GATE_FAILED",
    "GATE_PASSED",
)
DEPENDENCY_ROLES = ("source_lock", "gate_lock", "analyzer", "runner", "instrument_launch_lock")
CELLS_PER_DEVICE = len(STRATEGIES) * len(LANES) * len(GRID_IDS)
TOTAL_RECORDS = len(GPU_UUIDS) * CELLS_PER_DEVICE


class ProtocolError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _verified_file(root: Path, relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ProtocolError("dependency path must be a safe relative path")
    candidate = root / relative
    if candidate.is_symlink():
        raise ProtocolError(f"dependency must not be a symlink: {relative}")
    path = candidate.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"dependency escapes its root: {relative}") from exc
    if not path.is_file():
        raise ProtocolError(f"dependency is not a regular file: {relative}")
    return path


def validate_contract(contract: Any, dependency_root: Path) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "campaign_id",
        "instrument_campaign_id",
        "reference_result_tag",
        "state",
        "claim_scope",
        "timing_allowed",
        "devices",
        "dependency_roles",
        "dependency_sha256",
    }
    if not isinstance(contract, dict) or set(contract) != expected_fields:
        raise ProtocolError("contract fields differ from the Ada replication schema")
    expected_scalars = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "instrument_campaign_id": INSTRUMENT_CAMPAIGN_ID,
        "reference_result_tag": REFERENCE_RESULT_TAG,
        "state": STATE,
        "claim_scope": CLAIM_SCOPE,
        "timing_allowed": False,
    }
    if any(contract[field] != value for field, value in expected_scalars.items()):
        raise ProtocolError("contract identity, scope, state, or timing policy changed")

    devices = contract["devices"]
    if not isinstance(devices, list) or len(devices) != len(GPU_UUIDS):
        raise ProtocolError("contract requires exactly the four bound Ada devices")
    expected_device_fields = {"uuid", "name", "compute_capability"}
    by_uuid: dict[str, dict[str, Any]] = {}
    for device in devices:
        if not isinstance(device, dict) or set(device) != expected_device_fields:
            raise ProtocolError("device fields must be uuid, name, and compute_capability")
        uuid = device["uuid"]
        if uuid in by_uuid:
            raise ProtocolError("device UUIDs must be unique")
        by_uuid[uuid] = device
    if set(by_uuid) != set(GPU_UUIDS):
        raise ProtocolError("device UUID set differs from the four kickoff bindings")
    if any(
        device["name"] != GPU_NAME
        or device["compute_capability"] != COMPUTE_CAPABILITY
        for device in by_uuid.values()
    ):
        raise ProtocolError("all bound devices must be the same sm_89 RTX 6000 Ada SKU")

    roles = contract["dependency_roles"]
    hashes = contract["dependency_sha256"]
    if not isinstance(roles, dict) or set(roles) != set(DEPENDENCY_ROLES):
        raise ProtocolError("dependency roles must bind source, gate, analyzer, runner, and instrument lock")
    if len(set(roles.values())) != len(roles):
        raise ProtocolError("dependency roles must bind distinct files")
    if not isinstance(hashes, dict) or set(hashes) != set(roles.values()):
        raise ProtocolError("dependency hashes must cover exactly the role-bound files")
    for relative, expected in hashes.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ProtocolError("dependency hashes must be lowercase SHA-256 values")
        actual = hashlib.sha256(_verified_file(dependency_root, relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ProtocolError(f"dependency hash mismatch: {relative}")
    return contract


def make_manifest(contract: Any, dependency_root: Path) -> dict[str, Any]:
    contract = validate_contract(contract, dependency_root)
    devices = {device["uuid"]: device for device in contract["devices"]}
    rows = []
    for device_index, uuid in enumerate(GPU_UUIDS):
        for strategy in STRATEGIES:
            for lane in LANES:
                for grid_id in GRID_IDS:
                    cell_id = f"{strategy}.{lane}.{grid_id}"
                    rows.append(
                        {
                            "request_id": f"{TAGS[device_index]}/{cell_id}",
                            "campaign_tag": TAGS[device_index],
                            "device_uuid": uuid,
                            "device_name": devices[uuid]["name"],
                            "compute_capability": devices[uuid]["compute_capability"],
                            "cell_id": cell_id,
                            "strategy": strategy,
                            "lane": lane,
                            "grid_id": grid_id,
                            "timing_allowed": False,
                        }
                    )
    if (
        len(rows) != TOTAL_RECORDS
        or len({row["request_id"] for row in rows}) != TOTAL_RECORDS
        or Counter(row["device_uuid"] for row in rows) != {uuid: CELLS_PER_DEVICE for uuid in GPU_UUIDS}
        or Counter(row["strategy"] for row in rows) != {item: 304 for item in STRATEGIES}
        or Counter(row["lane"] for row in rows) != {item: 304 for item in LANES}
        or Counter(row["grid_id"] for row in rows) != {item: 64 for item in GRID_IDS}
    ):
        raise ProtocolError("four-device factorial census changed")
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "instrument_campaign_id": INSTRUMENT_CAMPAIGN_ID,
        "reference_result_tag": REFERENCE_RESULT_TAG,
        "state": STATE,
        "claim_scope": CLAIM_SCOPE,
        "architecture_factor_varied": False,
        "contract_sha256": _canonical_sha256(contract),
        "dependency_roles": contract["dependency_roles"],
        "allowed_stages": list(ALLOWED_STAGES),
        "timing_allowed": False,
        "devices": [devices[uuid] for uuid in GPU_UUIDS],
        "cells_per_device": CELLS_PER_DEVICE,
        "requested_records": TOTAL_RECORDS,
        "rows": rows,
    }


def _derived_instrument_status(
    record: dict[str, Any], row: dict[str, Any], launch_lock_sha256: str,
    source_bundle_sha256: str, instrument_record_path: Path,
) -> str:
    cell = record.get("cell")
    if (
        record.get("schema_version") != 2
        or record.get("campaign_id") != INSTRUMENT_CAMPAIGN_ID
        or not isinstance(cell, dict)
        or any(cell.get(field) != row[field] for field in ("cell_id", "strategy", "lane", "grid_id"))
        or cell.get("requested") is not True
        or record.get("cell_sha256") != _canonical_sha256(cell)
        or record.get("launch_lock_sha256") != launch_lock_sha256
        or record.get("source_bundle_sha256") != source_bundle_sha256
    ):
        raise ProtocolError("instrument record lost its campaign, cell, source, or lock binding")
    outcome = record.get("terminal_outcome")
    if outcome not in TERMINAL_STATUSES:
        raise ProtocolError("instrument record has an unknown terminal outcome")
    if outcome == "UNSUPPORTED":
        if cell.get("support_declared") is not False or record.get("build_attempted") is not False or record.get("gate_attempted") is not False:
            raise ProtocolError("instrument unsupported outcome is not source-derived")
        return outcome
    if cell.get("support_declared") is not True:
        raise ProtocolError("instrument executable outcome lacks declared support")
    if outcome == "BUILD_FAILED":
        if record.get("build_attempted") is not True or record.get("gate_attempted") is not False:
            raise ProtocolError("instrument build failure has inconsistent stages")
        return outcome
    if record.get("build_attempted") is not True or record.get("gate_attempted") is not True:
        raise ProtocolError("instrument launch/gate outcome has inconsistent stages")
    gate_path = instrument_record_path.parent.parent / "gate" / (
        row["cell_id"].replace(".", "__") + ".jsonl"
    )
    if Path(record.get("gate_jsonl_path", "")).name != gate_path.name or not gate_path.is_file():
        raise ProtocolError("instrument gate evidence path is missing or mismatched")
    if hashlib.sha256(gate_path.read_bytes()).hexdigest() != record.get("gate_jsonl_sha256"):
        raise ProtocolError("instrument gate evidence hash mismatch")
    gate_rows = _read_jsonl(gate_path)
    coverage = {(item.get("case_id"), item.get("seed_index"), item.get("gate_id")) for item in gate_rows}
    complete = len(gate_rows) == 512 and len(coverage) == 512
    execution_error = any(item.get("ok") is not True for item in gate_rows)
    failed = sum(item.get("ok") is not True or item.get("gate_pass") is not True for item in gate_rows)
    derived = "LAUNCH_FAILED" if not complete or execution_error else "GATE_FAILED" if failed else "GATE_PASSED"
    gate_summary = record.get("gate_summary", {})
    if (
        outcome != derived
        or gate_summary.get("observed_records") != len(gate_rows)
        or gate_summary.get("failed_records") != failed
        or gate_summary.get("complete") is not complete
        or gate_summary.get("full_gate_pass") is not (complete and failed == 0)
    ):
        raise ProtocolError("instrument terminal outcome is not rederived from gate evidence")
    return derived


def validate_results(
    contract: dict[str, Any],
    dependency_root: Path,
    manifest: dict[str, Any],
    results: Any,
    evidence_root: Path,
    instrument_root: Path,
) -> dict[str, Any]:
    expected_manifest = make_manifest(contract, dependency_root)
    if manifest != expected_manifest:
        raise ProtocolError("manifest differs from the verified four-device factorial")
    if not isinstance(results, list) or len(results) != TOTAL_RECORDS:
        raise ProtocolError(f"results require exactly {TOTAL_RECORDS} terminal records")

    manifest_sha256 = _canonical_sha256(manifest)
    launch_lock_path = _verified_file(
        dependency_root, contract["dependency_roles"]["instrument_launch_lock"]
    )
    launch_lock_sha256 = hashlib.sha256(launch_lock_path.read_bytes()).hexdigest()
    launch_lock = _read_json(launch_lock_path)
    source_bundle_sha256 = launch_lock.get("source_bundle_sha256")
    if (
        launch_lock.get("schema_version") != 2
        or launch_lock.get("campaign_id") != INSTRUMENT_CAMPAIGN_ID
        or launch_lock.get("lock_stage") != "campaign"
        or not isinstance(source_bundle_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_bundle_sha256) is None
    ):
        raise ProtocolError("instrument launch lock is not the frozen campaign lock")
    expected = {row["request_id"]: row for row in manifest["rows"]}
    observed: set[str] = set()
    evidence_paths: set[str] = set()
    evidence_hashes: set[str] = set()
    instrument_record_paths: set[str] = set()
    instrument_record_hashes: set[str] = set()
    by_cell: dict[str, dict[str, str]] = {}
    total_counts = Counter()
    device_counts = {uuid: Counter() for uuid in GPU_UUIDS}
    result_fields = {
        "request_id", "terminal_status", "evidence_path", "evidence_sha256",
        "instrument_record_path", "instrument_record_sha256",
        "instrument_audit_receipt_path", "instrument_audit_receipt_sha256",
        "instrument_audit_summary_path", "instrument_audit_summary_sha256",
        "instrument_launch_lock_sha256",
    }
    receipt_cache: dict[tuple[str, str], dict[str, Any]] = {}
    summary_cache: dict[tuple[str, str], dict[str, Any]] = {}
    cells_by_audit_receipt: dict[str, set[str]] = {}
    digest_by_audit_receipt: dict[str, str] = {}
    device_by_summary: dict[str, str] = {}
    counts_by_summary: dict[str, Counter] = {}
    legal_by_summary: dict[str, set[str]] = {}
    receipts_by_summary: dict[str, set[str]] = {}

    for result in results:
        if not isinstance(result, dict) or set(result) != result_fields:
            raise ProtocolError("result contains missing, extra, or timing fields")
        request_id = result["request_id"]
        if request_id not in expected or request_id in observed:
            raise ProtocolError("result has an unknown or duplicate request")
        status = result["terminal_status"]
        if status not in TERMINAL_STATUSES:
            raise ProtocolError("result has an unknown terminal status")
        relative = result["evidence_path"]
        digest = result["evidence_sha256"]
        if not isinstance(relative, str) or not relative:
            raise ProtocolError("result evidence_path is invalid")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ProtocolError("result evidence_sha256 is invalid")
        if relative in evidence_paths or digest in evidence_hashes:
            raise ProtocolError("terminal evidence must be unique per request")
        provenance_digests = (
            result["instrument_record_sha256"], result["instrument_audit_receipt_sha256"],
            result["instrument_audit_summary_sha256"], result["instrument_launch_lock_sha256"],
        )
        if any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in provenance_digests):
            raise ProtocolError("instrument provenance hashes must be lowercase SHA-256 values")
        if result["instrument_launch_lock_sha256"] != launch_lock_sha256:
            raise ProtocolError("result cites another instrument launch lock")
        receipt_path = _verified_file(evidence_root, relative)
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != digest:
            raise ProtocolError("terminal evidence hash mismatch")
        row = expected[request_id]
        expected_record_path = f"{row['campaign_tag']}/audit/records/{row['cell_id'].replace('.', '__')}.json"
        expected_summary_path = f"{row['campaign_tag']}/audit_summary.json"
        if result["instrument_record_path"] != expected_record_path or result["instrument_audit_summary_path"] != expected_summary_path:
            raise ProtocolError("instrument record or summary path differs from the device-bound tag")
        instrument_record_path = _verified_file(instrument_root, expected_record_path)
        if hashlib.sha256(instrument_record_path.read_bytes()).hexdigest() != result["instrument_record_sha256"]:
            raise ProtocolError("instrument record hash mismatch")
        if expected_record_path in instrument_record_paths or result["instrument_record_sha256"] in instrument_record_hashes:
            raise ProtocolError("instrument records must be unique per request")
        instrument_record = _read_json(instrument_record_path)
        cell_index = instrument_record.get("cell", {}).get("cell_index")
        expected_cell_index = (
            STRATEGIES.index(row["strategy"]) * len(LANES) * len(GRID_IDS)
            + LANES.index(row["lane"]) * len(GRID_IDS)
            + GRID_IDS.index(row["grid_id"])
        )
        if (
            not isinstance(cell_index, int)
            or isinstance(cell_index, bool)
            or cell_index != expected_cell_index
        ):
            raise ProtocolError("instrument record cell_index is invalid")
        expected_audit_receipt_path = f"{row['campaign_tag']}/audit/receipts/shard{cell_index % 4:02d}.json"
        if result["instrument_audit_receipt_path"] != expected_audit_receipt_path:
            raise ProtocolError("instrument audit receipt path differs from the cell shard")
        audit_key = (expected_audit_receipt_path, result["instrument_audit_receipt_sha256"])
        if audit_key not in receipt_cache:
            path = _verified_file(instrument_root, expected_audit_receipt_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != audit_key[1]:
                raise ProtocolError("instrument audit receipt hash mismatch")
            receipt_cache[audit_key] = _read_json(path)
        audit_receipt = receipt_cache[audit_key]
        device_index = GPU_UUIDS.index(row["device_uuid"])
        audit_contract = audit_receipt.get("contract", {})
        gpu = audit_receipt.get("gpu", {})
        if (
            audit_receipt.get("schema_version") != 2
            or audit_receipt.get("record_type") != "fused_crossed_v2_audit_receipt"
            or audit_contract.get("campaign_id") != INSTRUMENT_CAMPAIGN_ID
            or audit_contract.get("physical_gpu") != device_index
            or audit_contract.get("shard_count") != 4
            or audit_contract.get("shard_index") != cell_index % 4
            or audit_contract.get("launch_lock_sha256") != launch_lock_sha256
            or audit_contract.get("source_bundle_sha256") != source_bundle_sha256
            or gpu.get("uuid") != row["device_uuid"]
            or gpu.get("name") != GPU_NAME
            or gpu.get("compute_cap") != COMPUTE_CAPABILITY
            or str(gpu.get("index")) != str(device_index)
            or instrument_record.get("physical_gpu") != device_index
        ):
            raise ProtocolError("instrument audit receipt lost its device, shard, source, or lock binding")
        derived_status = _derived_instrument_status(
            instrument_record, row, launch_lock_sha256, source_bundle_sha256,
            instrument_record_path,
        )
        if status != derived_status:
            raise ProtocolError("outer terminal status differs from the instrument record")

        summary_key = (expected_summary_path, result["instrument_audit_summary_sha256"])
        if summary_key not in summary_cache:
            path = _verified_file(instrument_root, expected_summary_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != summary_key[1]:
                raise ProtocolError("instrument audit summary hash mismatch")
            summary_cache[summary_key] = _read_json(path)
        instrument_summary = summary_cache[summary_key]
        if (
            instrument_summary.get("schema_version") != 2
            or instrument_summary.get("record_type") != "fused_crossed_v2_audit_summary"
            or instrument_summary.get("campaign_id") != INSTRUMENT_CAMPAIGN_ID
            or instrument_summary.get("complete") is not True
            or instrument_summary.get("requested_cells") != CELLS_PER_DEVICE
            or instrument_summary.get("launch_lock_sha256") != launch_lock_sha256
            or instrument_summary.get("source_bundle_sha256") != source_bundle_sha256
        ):
            raise ProtocolError("instrument audit summary lost its census, source, or lock binding")

        receipt = _read_json(receipt_path)
        expected_receipt = {
            "schema_version": 1,
            "record_type": "ada_device_replication_v1_terminal_receipt",
            "campaign_id": CAMPAIGN_ID,
            "claim_scope": CLAIM_SCOPE,
            "contract_sha256": manifest["contract_sha256"],
            "manifest_sha256": manifest_sha256,
            "request_id": request_id,
            "campaign_tag": row["campaign_tag"],
            "device_uuid": row["device_uuid"],
            "cell_id": row["cell_id"],
            "terminal_status": status,
            "timing_allowed": False,
            "instrument_record_path": result["instrument_record_path"],
            "instrument_record_sha256": result["instrument_record_sha256"],
            "instrument_audit_receipt_path": result["instrument_audit_receipt_path"],
            "instrument_audit_receipt_sha256": result["instrument_audit_receipt_sha256"],
            "instrument_audit_summary_path": result["instrument_audit_summary_path"],
            "instrument_audit_summary_sha256": result["instrument_audit_summary_sha256"],
            "instrument_launch_lock_sha256": result["instrument_launch_lock_sha256"],
        }
        if receipt != expected_receipt:
            raise ProtocolError("terminal receipt content mismatch")
        observed.add(request_id)
        evidence_paths.add(relative)
        evidence_hashes.add(digest)
        instrument_record_paths.add(expected_record_path)
        instrument_record_hashes.add(result["instrument_record_sha256"])
        by_cell.setdefault(row["cell_id"], {})[row["device_uuid"]] = status
        total_counts[status] += 1
        device_counts[row["device_uuid"]][status] += 1
        cells_by_audit_receipt.setdefault(expected_audit_receipt_path, set()).add(row["cell_id"])
        digest_by_audit_receipt[expected_audit_receipt_path] = result["instrument_audit_receipt_sha256"]
        prior_device = device_by_summary.setdefault(expected_summary_path, row["device_uuid"])
        if prior_device != row["device_uuid"]:
            raise ProtocolError("instrument audit summary is shared across devices")
        counts_by_summary.setdefault(expected_summary_path, Counter())[status] += 1
        legal_by_summary.setdefault(expected_summary_path, set())
        if status == "GATE_PASSED":
            legal_by_summary[expected_summary_path].add(row["cell_id"])
        receipts_by_summary.setdefault(expected_summary_path, set()).add(result["instrument_audit_receipt_sha256"])

    if observed != set(expected):
        raise ProtocolError("result census differs from the manifest")
    for audit_path, cells in cells_by_audit_receipt.items():
        audit = receipt_cache[(audit_path, digest_by_audit_receipt[audit_path])]
        if set(audit.get("contract", {}).get("assigned_cell_ids", [])) != cells:
            raise ProtocolError("instrument audit receipt assignment differs from retained records")
    if len(device_by_summary) != len(GPU_UUIDS):
        raise ProtocolError("instrument results require one audit summary per device")
    for summary_path, uuid in device_by_summary.items():
        digest = next(key[1] for key in summary_cache if key[0] == summary_path)
        summary = summary_cache[(summary_path, digest)]
        expected_counts = {status: counts_by_summary[summary_path][status] for status in TERMINAL_STATUSES}
        listed_hashes = {
            item.get("sha256") for item in summary.get("receipt_hashes", []) if isinstance(item, dict)
        }
        if (
            summary.get("outcome_counts") != expected_counts
            or summary.get("timing_eligible_cell_ids") != sorted(legal_by_summary[summary_path])
            or not receipts_by_summary[summary_path] <= listed_hashes
            or sum(expected_counts.values()) != CELLS_PER_DEVICE
            or uuid not in GPU_UUIDS
        ):
            raise ProtocolError("instrument audit summary is not rederived from its retained records")
    discordant = sorted(
        cell_id for cell_id, statuses in by_cell.items() if len(set(statuses.values())) != 1
    )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "noncontrolling_complete_census",
        "claim_scope": CLAIM_SCOPE,
        "architecture_factor_varied": False,
        "timing_allowed": False,
        "complete_records": TOTAL_RECORDS,
        "concordant_cells": CELLS_PER_DEVICE - len(discordant),
        "discordant_cells": discordant,
        "terminal_counts": {status: total_counts[status] for status in TERMINAL_STATUSES},
        "terminal_counts_by_device": {
            uuid: {status: device_counts[uuid][status] for status in TERMINAL_STATUSES}
            for uuid in GPU_UUIDS
        },
    }


def refuse_launch() -> None:
    raise ProtocolError(
        "launch forbidden: freeze and remotely verify a separately authorized successor lock"
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"JSONL row is not an object: {path}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSONL {path}: {exc}") from exc
    return rows
