#!/usr/bin/env python3
"""Build or verify the deterministic pre-launch source receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_closure_v2 import core
else:  # pragma: no cover - module invocation follows this branch
    from . import core


LOCAL_SOURCE_NAMES = (
    "README.md",
    "PREREGISTRATION.md",
    "__init__.py",
    "analyze.py",
    "campaign.json",
    "candidates.py",
    "core.py",
    "gate_adjudicate.py",
    "launch.py",
    "measure.py",
    "provenance.py",
    "test_closure.py",
)

EXTERNAL_SOURCE_PATHS = (
    "ako_runs/phase1_matmul/common.py",
    "ako_runs/phase1_matmul/variants/__init__.py",
    "ako_runs/phase1_matmul/variants/cuda_noptx_gemm.py",
    "ako_runs/phase1_matmul/variants/cuda_unlimited_gemm.py",
    "ako_runs/phase2_fused_sdpa/common2.py",
    "ako_runs/phase2_fused_sdpa/variants2/SPEC2.md",
    "ako_runs/phase2_fused_sdpa/variants2/__init__.py",
    "ako_runs/phase2_fused_sdpa/variants2/cuda_fused_common.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_cuda_noptx.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_cuda_unlimited.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_tilelang.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_torch.py",
    "ako_runs/phase2_fused_sdpa/variants2/fused_triton.py",
    "ako_runs/controlled_followup/robust_gate/__init__.py",
    "ako_runs/controlled_followup/robust_gate/distributions.py",
    "ako_runs/controlled_followup/robust_gate/metrics.py",
    "ako_runs/controlled_followup/robust_gate/oracles.py",
    "ako_runs/controlled_followup/robust_gate/schema.py",
    "ako_runs/controlled_followup/robust_gate/seeds.py",
    "ako_runs/controlled_followup/robust_gate/validate.py",
)

FROZEN_EVIDENCE_PATHS = (
    "ako_runs/controlled_followup/fused_grid/manifest.json",
    "ako_runs/controlled_followup/fused_grid/jobs/fused_gbgs_grid.json",
    "ako_runs/controlled_followup/fused_grid/results/fused_gbgs_grid_rank/screen_summary.json",
    "ako_runs/controlled_followup/fused_grid/results/fused_gbgs_confirm_robust_v1/summary.json",
    "ako_runs/controlled_followup/robust_gate/manifest.json",
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json",
    "ako_runs/controlled_followup/robust_gate/validation/fused_gate_acceptance_v2.json",
    "ako_runs/controlled_followup/robust_gate/validation/fused_holdout_summary_v2.json",
)


def _hash_map(paths: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for relative in paths:
        path = core.REPO_ROOT / relative
        if not path.is_file():
            raise core.ClosureError(f"missing receipt input {relative}")
        result[relative] = core.sha256_file(path)
    return result


def _local_hash_map() -> dict[str, str]:
    result = {}
    for name in LOCAL_SOURCE_NAMES:
        path = core.HERE / name
        if not path.is_file():
            raise core.ClosureError(f"missing local source {name}")
        result[str(path.relative_to(core.REPO_ROOT))] = core.sha256_file(path)
    return result


def _gate_binding(evidence: dict[str, str]) -> dict[str, Any]:
    manifest_path = core.REPO_ROOT / "ako_runs/controlled_followup/robust_gate/manifest.json"
    gate_path = core.REPO_ROOT / (
        "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json"
    )
    acceptance_path = core.REPO_ROOT / (
        "ako_runs/controlled_followup/robust_gate/validation/fused_gate_acceptance_v2.json"
    )
    manifest = core.read_json(manifest_path)
    gate = core.read_json(gate_path)
    acceptance = core.read_json(acceptance_path)
    manifest_canonical = core.canonical_sha256(manifest)
    gate_canonical = core.canonical_sha256(gate)
    if gate.get("manifest_sha256") != manifest_canonical:
        raise core.ClosureError("frozen fused-v2 gate no longer binds robust manifest")
    expected_keys = {
        "fused_softmax/semantic_mixed",
        "fused_softmax/conformance_mixed",
    }
    if set(gate.get("gates", {})) != expected_keys:
        raise core.ClosureError("frozen gate key set changed")
    if not acceptance.get("accepted_for_fused_grid_screening"):
        raise core.ClosureError("fused-v2 acceptance receipt is not accepted")
    frozen = acceptance.get("frozen_gate", {})
    if frozen.get("file_sha256") != evidence[str(gate_path.relative_to(core.REPO_ROOT))]:
        raise core.ClosureError("acceptance receipt gate file hash differs")
    if frozen.get("manifest_canonical_sha256") != manifest_canonical:
        raise core.ClosureError("acceptance receipt manifest binding differs")
    return {
        "accepted": True,
        "acceptance_receipt_sha256": evidence[
            str(acceptance_path.relative_to(core.REPO_ROOT))
        ],
        "gate_spec_canonical_sha256": gate_canonical,
        "gate_spec_file_sha256": evidence[str(gate_path.relative_to(core.REPO_ROOT))],
        "robust_manifest_canonical_sha256": manifest_canonical,
        "robust_manifest_file_sha256": evidence[
            str(manifest_path.relative_to(core.REPO_ROOT))
        ],
    }


def expected_receipt() -> dict[str, Any]:
    campaign = core.load_campaign()
    evidence = _hash_map(FROZEN_EVIDENCE_PATHS)
    return {
        "schema_version": 1,
        "record_type": "fused_closure_v2_source_receipt",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "candidate_sha256": {
            candidate["candidate_id"]: core.candidate_sha256(candidate)
            for candidate in campaign["candidates"]
        },
        "performance_protocol_sha256": core.protocol_sha256(campaign),
        "local_source_sha256": _local_hash_map(),
        "external_source_sha256": _hash_map(EXTERNAL_SOURCE_PATHS),
        "frozen_evidence_sha256": evidence,
        "frozen_gate": _gate_binding(evidence),
        "freeze_policy": (
            "deterministic pre-launch receipt; any byte change requires a new receipt "
            "and a new result tag"
        ),
    }


def verify_receipt(path: Path = core.SOURCE_RECEIPT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise core.ClosureError(f"missing source receipt: {path}")
    observed = core.read_json(path)
    expected = expected_receipt()
    if observed != expected:
        raise core.ClosureError("source receipt differs from current frozen inputs")
    if path.read_bytes() != core.stable_json_bytes(observed):
        raise core.ClosureError("source receipt serialization is not stable JSON")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()
    if args.emit:
        print(core.stable_json_bytes(expected_receipt()).decode("utf-8"), end="")
        return 0
    if not args.check:
        parser.error("choose --check or --emit")
    receipt = verify_receipt()
    print(
        f"OK {receipt['campaign_id']} source receipt "
        f"sha256={core.sha256_file(core.SOURCE_RECEIPT_PATH)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

