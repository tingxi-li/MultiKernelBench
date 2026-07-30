#!/usr/bin/env python3
"""Generate the content-addressed fused-grid/robust-gate adapter manifest.

This generator only reads the frozen robust-gate campaign and the existing
fused-grid job list.  It does not rewrite either input.  The generated adapter
manifest binds every local source used to construct Phase-2 candidates and to
score their outputs, so a resumed result cannot silently cross a code change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTROLLED = HERE.parent
REPO_ROOT = HERE.parents[2]
ROBUST = CONTROLLED / "robust_gate"
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"

GRID_MANIFEST = HERE / "manifest.json"
ROBUST_MANIFEST = ROBUST / "manifest.json"
GATE_SPEC = ROBUST / "calibration/gate_spec_fused_v2.json"
OUT = HERE / "robust_adapter_manifest.json"

GATE_KEYS = (
    "fused_softmax/semantic_mixed",
    "fused_softmax/conformance_mixed",
)
SPLITS = {"tuning": 8, "validation": 64}

# Runtime code only.  The generated manifest is deliberately absent to avoid a
# self-hash cycle; tests and documentation are evidence, not execution inputs.
SOURCE_PATHS = (
    HERE / "make_robust_manifest.py",
    HERE / "robust_adapter.py",
    HERE / "robust_launch.py",
    ROBUST / "__init__.py",
    ROBUST / "schema.py",
    ROBUST / "distributions.py",
    ROBUST / "oracles.py",
    ROBUST / "metrics.py",
    ROBUST / "seeds.py",
    ROBUST / "validate.py",
    PHASE1 / "common.py",
    PHASE1 / "variants/__init__.py",
    PHASE1 / "variants/cuda_noptx_gemm.py",
    PHASE1 / "variants/cuda_unlimited_gemm.py",
    PHASE2 / "common2.py",
    PHASE2 / "variants2/__init__.py",
    PHASE2 / "variants2/cuda_fused_common.py",
    PHASE2 / "variants2/fused_tilelang.py",
    PHASE2 / "variants2/fused_triton.py",
    PHASE2 / "variants2/fused_cuda_noptx.py",
    PHASE2 / "variants2/fused_cuda_unlimited.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def stable_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def repo_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_document() -> dict[str, Any]:
    # Import the frozen validators from their owning package rather than
    # reimplementing their manifest rules here.
    sys.path.insert(0, str(CONTROLLED))
    from robust_gate.schema import validate_gate_spec, validate_manifest

    grid_manifest = load(GRID_MANIFEST)
    jobs_path = HERE / grid_manifest["jobs_file"]
    jobs = load(jobs_path)
    robust_manifest = load(ROBUST_MANIFEST)
    gate_spec = load(GATE_SPEC)

    validate_manifest(robust_manifest)
    validate_gate_spec(gate_spec)
    robust_canonical = canonical_sha256(robust_manifest)
    if gate_spec["manifest_sha256"] != robust_canonical:
        raise ValueError("frozen gate spec does not bind the robust manifest")
    if gate_spec["campaign_id"] != robust_manifest["campaign_id"]:
        raise ValueError("gate-spec and robust-manifest campaign IDs differ")
    if set(gate_spec["gates"]) != set(GATE_KEYS):
        raise ValueError(
            f"expected exactly fused mixed gates {GATE_KEYS}, got "
            f"{sorted(gate_spec['gates'])}"
        )

    op_spec = robust_manifest["operations"]["fused_softmax"]
    case_ids = [case["id"] for case in op_spec["cases"]]
    for gate_key in GATE_KEYS:
        gate_id = gate_key.split("/", 1)[1]
        frozen = gate_spec["gates"][gate_key]
        declared = op_spec["gates"][gate_id]
        if frozen["required_cases"] != case_ids:
            raise ValueError(f"{gate_key} changed its required cases/order")
        if frozen["required_validation_seeds_per_case"] != SPLITS["validation"]:
            raise ValueError(f"{gate_key} changed its locked validation count")
        if frozen["contract"] != declared["contract"]:
            raise ValueError(f"{gate_key} contract differs from robust manifest")
        if frozen["reference"] != declared["reference"]:
            raise ValueError(f"{gate_key} reference differs from robust manifest")
    if robust_manifest["split_counts"]["tuning"] != SPLITS["tuning"]:
        raise ValueError("robust manifest tuning split is no longer eight seeds")

    jobs_raw_hash = file_sha256(jobs_path)
    if jobs_raw_hash != grid_manifest["jobs_sha256"]:
        raise ValueError("grid jobs do not match the existing grid manifest")
    if len(jobs) != grid_manifest["job_count"]:
        raise ValueError("grid manifest job count is stale")
    job_hashes: dict[str, str] = {}
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"invalid grid job ID: {job!r}")
        if job_id in job_hashes:
            raise ValueError(f"duplicate grid job ID {job_id!r}")
        if job.get("geom") != "fused" or job.get("variant") != "GBGS":
            raise ValueError(f"job left the fused GBGS contract: {job!r}")
        job_hashes[job_id] = canonical_sha256(job)

    fixed = grid_manifest["fixed_factors"]
    expected_shape = op_spec["shape"]
    if {name: fixed[name] for name in ("M", "K", "N")} != expected_shape:
        raise ValueError("grid and robust-gate fused shapes differ")
    expected_fixed = {
        "variant": "GBGS",
        "arith": "fp16",
        "cast": "precast",
        "wcache": "cached",
        "epilogue": "smem",
    }
    for name, value in expected_fixed.items():
        if fixed.get(name) != value:
            raise ValueError(f"grid fixed factor {name}={fixed.get(name)!r}, expected {value!r}")

    source_hashes = {repo_path(path): file_sha256(path) for path in SOURCE_PATHS}
    return {
        "schema_version": 1,
        "campaign_id": "fused-gbgs-robust-gate-v1",
        "operation": "fused_softmax",
        "grid": {
            "campaign_id": grid_manifest["campaign_id"],
            "manifest_path": repo_path(GRID_MANIFEST),
            "manifest_sha256": file_sha256(GRID_MANIFEST),
            "manifest_canonical_sha256": canonical_sha256(grid_manifest),
            "jobs_path": repo_path(jobs_path),
            "jobs_sha256": jobs_raw_hash,
            "job_count": len(jobs),
            "job_sha256": job_hashes,
        },
        "robust_gate": {
            "campaign_id": robust_manifest["campaign_id"],
            "manifest_path": repo_path(ROBUST_MANIFEST),
            "manifest_sha256": file_sha256(ROBUST_MANIFEST),
            "manifest_canonical_sha256": robust_canonical,
            "gate_spec_path": repo_path(GATE_SPEC),
            "gate_spec_sha256": file_sha256(GATE_SPEC),
            "gate_spec_canonical_sha256": canonical_sha256(gate_spec),
            "gate_keys": list(GATE_KEYS),
            "case_ids": case_ids,
            "split_counts": SPLITS,
            "shape": expected_shape,
        },
        "execution_contract": {
            "candidate_output_reuse": "one output per grid job/case/seed scores both gates",
            "input_reference_reuse": "one input and one reference per kind per case/seed across selected jobs",
            "failure_policy": "retain build, setup, execution, and metric failures",
            "validation_policy": "locked frozen thresholds; complete 4-case x 64-seed coverage required",
        },
        "source_sha256": source_hashes,
        "source_bundle_sha256": canonical_sha256(source_hashes),
    }


def expected_bytes() -> bytes:
    return stable_bytes(build_document())


def check() -> None:
    expected = expected_bytes()
    if not OUT.exists():
        raise SystemExit(f"missing generated file: {OUT}")
    if OUT.read_bytes() != expected:
        raise SystemExit(
            f"stale generated file: {OUT}\n"
            "run make_robust_manifest.py to regenerate it"
        )
    document = json.loads(expected)
    print(
        f"OK {document['campaign_id']}: {document['grid']['job_count']} jobs, "
        f"gates={','.join(document['robust_gate']['gate_keys'])}, "
        f"source_bundle_sha256={document['source_bundle_sha256']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return 0
    data = expected_bytes()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(f".{OUT.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, OUT)
    print(f"wrote {repo_path(OUT)} ({hashlib.sha256(data).hexdigest()})")
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
