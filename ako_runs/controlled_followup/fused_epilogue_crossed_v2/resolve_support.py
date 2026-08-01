#!/usr/bin/env python3
"""Bind two complete measured probe indexes into the v2 support matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import support_probes
    from .core import (
        CAMPAIGN_ID, REPO_ROOT, SUPPORT_RESOLUTION_PATH, file_sha256, read_json,
        stable_write, validate_support_resolution,
    )
except ImportError:  # direct script execution
    import support_probes
    from core import (
    CAMPAIGN_ID,
    REPO_ROOT,
    SUPPORT_RESOLUTION_PATH,
    file_sha256,
    read_json,
    stable_write,
    validate_support_resolution,
    )


def bind(cuda_index: Path, triton_index: Path) -> dict:
    current = read_json(SUPPORT_RESOLUTION_PATH)
    if current.get("status") != "unresolved":
        raise RuntimeError("support resolution is already sealed; refusing to replace it")
    indexes = {
        "cuda_noptx_register": cuda_index.resolve(),
        "triton_smem": triton_index.resolve(),
    }
    probes = {}
    for key, path in indexes.items():
        try:
            relative = str(path.relative_to(REPO_ROOT.resolve()))
        except ValueError as exc:
            raise RuntimeError(f"probe index escapes the repository: {path}") from exc
        index = support_probes.load_result_index(path)
        if index.get("probe_key") != key:
            raise RuntimeError(f"wrong probe index for {key}: {path}")
        status = index.get("resolution", {}).get("status")
        if status not in {"supported", "unsupported"}:
            raise RuntimeError(f"probe remains unresolved: {key}")
        probes[key] = {
            **current["probes"][key],
            "result_index_path": relative,
            "result_index_sha256": file_sha256(path),
            "status": status,
        }
    value = {
        **current,
        "campaign_id": CAMPAIGN_ID,
        "probes": probes,
        "status": "resolved",
    }
    validate_support_resolution(value, require_resolved=True)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-noptx-index", type=Path, required=True)
    parser.add_argument("--triton-smem-index", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = bind(args.cuda_noptx_index, args.triton_smem_index)
    if args.write:
        stable_write(SUPPORT_RESOLUTION_PATH, value)
        print(f"wrote={SUPPORT_RESOLUTION_PATH}")
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
