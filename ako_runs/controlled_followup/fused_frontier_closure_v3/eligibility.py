#!/usr/bin/env python3
"""Verify and summarize imported original fused-v2 gate eligibility."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_frontier_closure_v3 import core
else:  # pragma: no cover
    from . import core


CLOSURE = core.REPO_ROOT / "ako_runs/controlled_followup/fused_closure_v2"
REACH = core.REPO_ROOT / "ako_runs/controlled_followup/fused_reachability_v2"
REACH_INDEX = REACH / "evidence/complete_v1.index.json"


def _verify_reach_evidence() -> dict[str, Any]:
    index = core.read_json(REACH_INDEX)
    if index.get("record_type") != "fused_reachability_v2_evidence_index":
        raise core.ClosureError("foreign reachability evidence index")
    bundle = core.REPO_ROOT / index["bundle_path"]
    if core.sha256_file(bundle) != index["bundle_sha256"]:
        raise core.ClosureError("reachability evidence bundle hash differs")
    manifest = index["manifest"]
    if core.canonical_sha256(manifest) != index["manifest_sha256"]:
        raise core.ClosureError("reachability evidence manifest hash differs")
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    for relative, entry in entries.items():
        path = core.REPO_ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != entry["size"]
            or core.sha256_file(path) != entry["sha256"]
        ):
            raise core.ClosureError(f"reachability evidence entry differs: {relative}")
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = {item.name: item for item in archive.getmembers() if item.isfile()}
        embedded = members.pop("EVIDENCE_MANIFEST.json", None)
        if embedded is None:
            raise core.ClosureError("reachability evidence has no embedded manifest")
        embedded_bytes = archive.extractfile(embedded).read()
        if embedded_bytes != core.canonical_json_bytes(manifest) + b"\n":
            raise core.ClosureError("reachability embedded manifest differs")
        if set(members) != set(entries):
            raise core.ClosureError("reachability archive membership differs")
        for relative, entry in entries.items():
            data = archive.extractfile(members[relative]).read()
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry[
                "sha256"
            ]:
                raise core.ClosureError(
                    f"reachability archive payload differs: {relative}"
                )
    return {
        "index_path": str(REACH_INDEX.relative_to(core.REPO_ROOT)),
        "index_sha256": core.sha256_file(REACH_INDEX),
        "bundle_path": index["bundle_path"],
        "bundle_sha256": index["bundle_sha256"],
        "manifest_sha256": index["manifest_sha256"],
        "entry_count": index["entry_count"],
        "launch_lock_sha256": manifest["launch_lock_sha256"],
        "source_bundle_sha256": manifest["source_bundle_sha256"],
    }


def _verify_closure() -> tuple[dict[str, Any], dict[str, Any]]:
    from ako_runs.controlled_followup.fused_closure_v2 import (
        core as closure_core,
        launch as closure_launch,
        provenance as closure_provenance,
    )

    campaign = closure_core.load_campaign()
    receipt = closure_provenance.verify_receipt()
    gate_path = CLOSURE / "results/gate_v1/gate_summary.json"
    gate, gate_sha = closure_launch._load_gate_summary(campaign, receipt, gate_path)
    analysis_path = CLOSURE / "results/performance_v1/analysis_summary.json"
    analysis = closure_core.read_json(analysis_path)
    if (
        analysis.get("record_type")
        != "fused_closure_v2_performance_analysis"
        or analysis.get("status") != "COMPLETE"
        or analysis.get("source_receipt_sha256")
        != closure_core.sha256_file(closure_core.SOURCE_RECEIPT_PATH)
    ):
        raise core.ClosureError("fused_closure_v2 analysis is not complete/bound")
    for relative, expected in analysis.get("raw_record_sha256", {}).items():
        path = core.REPO_ROOT / relative
        if not path.is_file() or core.sha256_file(path) != expected:
            raise core.ClosureError(f"closure performance raw hash differs: {relative}")
    if len(analysis.get("raw_record_sha256", {})) != 135:
        raise core.ClosureError("closure performance evidence is incomplete")
    adjudications = {row["candidate_id"]: row for row in gate["adjudications"]}
    wanted = {
        "torch_contract_fp32": "torch_contract_fp32",
        "tilelang_full_g08": "tilelang_full_g08",
        "triton_full_g05": "triton_full_g05",
        "cuda_noptx_old_g04": "cuda_noptx_common_g04",
        "cuda_unlimited_old_g02": "cuda_unlimited_common_g02",
    }
    result = {}
    for target, source in wanted.items():
        row = adjudications.get(source)
        if not row or not row.get("same_contract_eligible"):
            raise core.ClosureError(f"closure candidate is gate-ineligible: {source}")
        gates = row.get("gates", {})
        if set(gates) != {"semantic_mixed", "conformance_mixed"} or any(
            group.get("n_records") != 256
            or group.get("n_failed_records") != 0
            or not group.get("coverage_complete")
            or not group.get("success")
            for group in gates.values()
        ):
            raise core.ClosureError(f"closure gate coverage differs: {source}")
        result[target] = {
            "eligible": True,
            "source_candidate_id": source,
            "gate_records": 512,
            "gate_failures": 0,
            "gate_ids": ["semantic_mixed", "conformance_mixed"],
            "case_count": 4,
            "validation_seeds_per_case": 64,
        }
    binding = {
        "source_receipt_path": str(
            closure_core.SOURCE_RECEIPT_PATH.relative_to(core.REPO_ROOT)
        ),
        "source_receipt_sha256": closure_core.sha256_file(
            closure_core.SOURCE_RECEIPT_PATH
        ),
        "gate_summary_path": str(gate_path.relative_to(core.REPO_ROOT)),
        "gate_summary_sha256": gate_sha,
        "gate_raw_records_canonical_sha256": gate[
            "raw_records_canonical_sha256"
        ],
        "analysis_path": str(analysis_path.relative_to(core.REPO_ROOT)),
        "analysis_sha256": core.sha256_file(analysis_path),
        "performance_launch_receipt_sha256": analysis["launch_receipt_sha256"],
        "performance_raw_records": len(analysis["raw_record_sha256"]),
    }
    return binding, result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                import json

                rows.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                raise core.ClosureError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def _verify_reachability() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _verify_reach_evidence()
    from ako_runs.controlled_followup.fused_reachability_v2 import protocol

    lock = protocol.verify_lock()
    if core.sha256_file(protocol.LOCK) != evidence["launch_lock_sha256"]:
        raise core.ClosureError("reachability launch lock differs from evidence")
    selection_path = REACH / "results/screen_analysis_v1/confirmation_selection.json"
    confirmation_path = REACH / "results/confirmation_analysis_v1/confirmation_summary.json"
    selection = core.read_json(selection_path)
    confirmation = core.read_json(confirmation_path)
    if not selection.get("all_selected_pass") or selection.get("status") != (
        "COMPLETE_ALL_PASS"
    ):
        raise core.ClosureError("reachability robust selection is not complete/pass")
    if confirmation.get("phase") != "confirmation" or confirmation.get(
        "expected_reps"
    ) != 15:
        raise core.ClosureError("reachability confirmation summary differs")
    selected = {row["job_id"]: row for row in selection["selected"]}
    targets = {
        "cuda_noptx_streamed_g05": ("cuda_noptx_streamed.g05", "cuda_noptx"),
        "cuda_noptx_streamed_g09": ("cuda_noptx_streamed.g09", "cuda_noptx"),
        "cuda_unlimited_streamed_g07": (
            "cuda_unlimited_streamed.g07",
            "cuda_unlimited",
        ),
    }
    result = {}
    robust_bindings = {}
    for lane, tag in (
        ("cuda_noptx", "robust_noptx_v1"),
        ("cuda_unlimited", "robust_unlimited_v1"),
    ):
        root = REACH / "results" / tag
        summary_path = root / "summary.json"
        records_path = root / "records.jsonl"
        summary = core.read_json(summary_path)
        if (
            not summary.get("success")
            or summary.get("status") != "PASS"
            or not summary.get("complete_frozen_validation_split")
            or not summary.get("launch_coverage", {}).get("complete")
        ):
            raise core.ClosureError(f"reachability robust summary failed: {lane}")
        groups = {
            (group["grid_job_id"], group["gate_id"]): group
            for group in summary["groups"]
        }
        records = _load_jsonl(records_path)
        for target, (job_id, job_lane) in targets.items():
            if job_lane != lane:
                continue
            selection_row = selected.get(job_id)
            if not selection_row or not selection_row.get("robust_eligible"):
                raise core.ClosureError(f"reachability selection ineligible: {job_id}")
            for gate_id in ("semantic_mixed", "conformance_mixed"):
                group = groups.get((job_id, gate_id))
                if (
                    not group
                    or group.get("n_records") != 256
                    or group.get("n_failed_records") != 0
                    or not group.get("coverage_complete")
                    or not group.get("success")
                ):
                    raise core.ClosureError(f"reachability gate group differs: {job_id}")
            job_rows = [row for row in records if row.get("grid_job_id") == job_id]
            keys = {
                (row.get("gate_id"), row.get("case_id"), row.get("seed_index"))
                for row in job_rows
            }
            expected = {
                (gate_id, case_id, seed)
                for gate_id in ("semantic_mixed", "conformance_mixed")
                for case_id in (
                    "legacy_u01_gain1",
                    "signed_normal_gain1",
                    "rademacher_gain4",
                    "signed_normal_gain16",
                )
                for seed in range(64)
            }
            if (
                keys != expected
                or len(job_rows) != 512
                or any(not row.get("ok") or not row.get("gate_pass") for row in job_rows)
            ):
                raise core.ClosureError(f"reachability raw gate coverage differs: {job_id}")
            result[target] = {
                "eligible": True,
                "source_job_id": job_id,
                "source_job_sha256": selection_row["job_sha256"],
                "gate_records": 512,
                "gate_failures": 0,
                "gate_ids": ["semantic_mixed", "conformance_mixed"],
                "case_count": 4,
                "validation_seeds_per_case": 64,
            }
        robust_bindings[lane] = {
            "summary_path": str(summary_path.relative_to(core.REPO_ROOT)),
            "summary_sha256": core.sha256_file(summary_path),
            "records_path": str(records_path.relative_to(core.REPO_ROOT)),
            "records_sha256": core.sha256_file(records_path),
            "record_count": len(records),
        }
    lanes = {row["lane"]: row for row in confirmation["inference"]["lanes"]}
    if (
        lanes["cuda_noptx"]["status"] != "UNRESOLVED"
        or lanes["cuda_noptx"]["point_winner_job_id"]
        != "cuda_noptx_streamed.g05"
        or lanes["cuda_unlimited"]["status"] != "RESOLVED"
        or lanes["cuda_unlimited"]["point_winner_job_id"]
        != "cuda_unlimited_streamed.g07"
    ):
        raise core.ClosureError("reachability selection status differs")
    binding = {
        **evidence,
        "launch_lock_path": str(protocol.LOCK.relative_to(core.REPO_ROOT)),
        "selection_path": str(selection_path.relative_to(core.REPO_ROOT)),
        "selection_sha256": core.sha256_file(selection_path),
        "confirmation_path": str(confirmation_path.relative_to(core.REPO_ROOT)),
        "confirmation_sha256": core.sha256_file(confirmation_path),
        "robust": robust_bindings,
    }
    return binding, result


def expected_receipt() -> dict[str, Any]:
    campaign = core.load_campaign()
    closure_binding, closure_candidates = _verify_closure()
    reach_binding, reach_candidates = _verify_reachability()
    candidates = {**closure_candidates, **reach_candidates}
    if list(candidates) != campaign["candidate_order"]:
        # Dict concatenation follows source groups, not campaign order; normalize.
        candidates = {candidate_id: candidates[candidate_id] for candidate_id in campaign["candidate_order"]}
    if set(candidates) != set(campaign["candidate_order"]):
        raise core.ClosureError("eligibility receipt candidate set differs")
    return {
        "schema_version": 1,
        "record_type": "fused_frontier_closure_v3_original_eligibility",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "scope": campaign["eligibility"]["scope"],
        "claim_limit": campaign["eligibility"]["claim_limit"],
        "all_candidates_original_gate_eligible": all(
            row["eligible"] for row in candidates.values()
        ),
        "candidates": candidates,
        "fused_closure_v2": closure_binding,
        "fused_reachability_v2": reach_binding,
    }


def verify_receipt(path: Path = core.ELIGIBILITY_RECEIPT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise core.ClosureError(f"missing eligibility receipt: {path}")
    observed = core.read_json(path)
    expected = expected_receipt()
    if observed != expected or path.read_bytes() != core.stable_json_bytes(observed):
        raise core.ClosureError("eligibility receipt differs from original evidence")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.emit:
        print(core.stable_json_bytes(expected_receipt()).decode("utf-8"), end="")
    else:
        receipt = verify_receipt()
        print(
            f"OK original gate eligibility for {len(receipt['candidates'])} candidates; "
            "scope=frozen 4x64 mixed validation only"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
