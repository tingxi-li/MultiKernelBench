#!/usr/bin/env python3
"""Build the controlled fused-GBGS equal-grid campaign manifests.

The configuration grid is intentionally *read* from Phase 1's checked-in
``native_tuned.json`` rather than copied here.  Phase 1 contains 76 jobs: the
same 19 configurations for each of four DSLs.  This generator verifies that
invariant, extracts the 19-point grid once, and applies it to Phase 2's fused
``GBGS`` arm.

Generated files are deterministic: there are no timestamps, host names, or
absolute paths in them.  Runtime provenance belongs in launcher result records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE_JOBS = REPO_ROOT / "ako_runs/phase1_matmul/jobs/native_tuned.json"
JOBS_PATH = HERE / "jobs/fused_gbgs_grid.json"
MANIFEST_PATH = HERE / "manifest.json"

CAMPAIGN_ID = "fused-gbgs-shared-grid-v1"
DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
EXPECTED_GRID_POINTS = 19
FIXED = {
    "op": "fused",
    "variant": "GBGS",
    "M": 1024,
    "K": 8192,
    "N": 8192,
    "threads": 256,
    "kc": 2048,
    "arith": "fp16",
    "cast": "precast",
    "wcache": "cached",
    "epilogue": "smem",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_set(setstr: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for part in setstr.split(","):
        key, raw = part.split("=", 1)
        parsed[key.strip()] = int(raw.strip())
    return parsed


def source_grid() -> tuple[list[dict[str, int]], str]:
    """Return Phase 1's ordered 19-point grid and its source-file hash."""
    raw = SOURCE_JOBS.read_bytes()
    source = json.loads(raw)
    if not isinstance(source, list):
        raise ValueError(f"{SOURCE_JOBS} must contain a JSON list")

    by_dsl: dict[str, list[dict[str, int]]] = {dsl: [] for dsl in DSLS}
    for job in source:
        dsl = job.get("dsl")
        if dsl not in by_dsl:
            raise ValueError(f"unexpected Phase-1 DSL {dsl!r}")
        if job.get("variant") != "D" or job.get("geom") != "primary":
            raise ValueError(f"unexpected Phase-1 grid job: {job!r}")
        point = parse_set(job.get("set", ""))
        if set(point) != {"BM", "BN", "BK", "stages", "kc"}:
            raise ValueError(f"unexpected Phase-1 grid axes: {point!r}")
        if point["kc"] != FIXED["kc"]:
            raise ValueError(f"Phase-1 point changed KC: {point!r}")
        by_dsl[dsl].append(point)

    reference = by_dsl[DSLS[0]]
    if len(reference) != EXPECTED_GRID_POINTS:
        raise ValueError(
            f"expected {EXPECTED_GRID_POINTS} Phase-1 points, found {len(reference)}"
        )
    if len({tuple(sorted(point.items())) for point in reference}) != len(reference):
        raise ValueError("Phase-1 source grid contains a duplicate point")
    for dsl in DSLS[1:]:
        if by_dsl[dsl] != reference:
            raise ValueError(f"Phase-1 grid/order differs for {dsl}")
    return reference, sha256_bytes(raw)


def fused_set(point: dict[str, int]) -> str:
    """Serialize in runner2.parse_set's accepted, filename-stable order."""
    return ",".join(
        (
            f"BM={point['BM']}",
            f"BN={point['BN']}",
            f"BK={point['BK']}",
            f"threads={FIXED['threads']}",
            f"stages={point['stages']}",
            f"kc={point['kc']}",
            f"arith={FIXED['arith']}",
            f"cast={FIXED['cast']}",
            f"x_wcache={FIXED['wcache']}",
            f"x_epilogue={FIXED['epilogue']}",
        )
    )


def build_documents() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grid, source_hash = source_grid()
    jobs: list[dict[str, Any]] = []
    # Preserve Phase 1's DSL-major order.  The launcher independently shuffles
    # (job, repetition) pairs with a fixed seed before execution.
    for dsl in DSLS:
        for grid_index, point in enumerate(grid):
            grid_id = f"g{grid_index:02d}"
            jobs.append(
                {
                    "dsl": dsl,
                    "geom": "fused",
                    "grid_id": grid_id,
                    "grid_index": grid_index,
                    "job_id": f"{dsl}.{grid_id}",
                    "set": fused_set(point),
                    "variant": FIXED["variant"],
                }
            )

    jobs_bytes = stable_json_bytes(jobs)
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "dsl_order": list(DSLS),
        "fixed_factors": FIXED,
        "grid": grid,
        "grid_point_count": len(grid),
        "job_count": len(jobs),
        "jobs_file": str(JOBS_PATH.relative_to(HERE)),
        "jobs_sha256": sha256_bytes(jobs_bytes),
        "phase1_grid_source": str(SOURCE_JOBS.relative_to(REPO_ROOT)),
        "phase1_grid_source_sha256": source_hash,
        "schema_version": 1,
    }
    return jobs, manifest


def expected_files() -> dict[Path, bytes]:
    jobs, manifest = build_documents()
    return {
        JOBS_PATH: stable_json_bytes(jobs),
        MANIFEST_PATH: stable_json_bytes(manifest),
    }


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def check_files() -> None:
    for path, expected in expected_files().items():
        if not path.exists():
            raise SystemExit(f"missing generated file: {path}")
        actual = path.read_bytes()
        if actual != expected:
            raise SystemExit(
                f"stale generated file: {path}\n"
                "run make_manifest.py to regenerate it"
            )
    jobs, manifest = build_documents()
    print(
        f"OK {manifest['campaign_id']}: {manifest['grid_point_count']} grid points "
        f"x {len(DSLS)} DSLs = {len(jobs)} deterministic jobs; "
        f"jobs_sha256={manifest['jobs_sha256']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="verify generated files without writing"
    )
    args = parser.parse_args()
    if args.check:
        check_files()
        return 0

    files = expected_files()
    for path, data in files.items():
        atomic_write(path, data)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({sha256_bytes(data)})")
    check_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
