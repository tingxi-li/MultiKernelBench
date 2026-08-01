#!/usr/bin/env python3
"""Generate and check the deterministic 320-trajectory factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .campaign import (
        CAMPAIGN_ID,
        MODELS,
        OPERATIONS,
        build_manifests,
        canonical_json,
        sha256_bytes,
        validate_manifest_rows,
    )
except ImportError:  # direct script execution
    from campaign import (  # type: ignore
        CAMPAIGN_ID,
        MODELS,
        OPERATIONS,
        build_manifests,
        canonical_json,
        sha256_bytes,
        validate_manifest_rows,
    )


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(canonical_json(row) for row in rows)


def expected_outputs() -> dict[str, bytes]:
    core, prompt = build_manifests()
    core_bytes = jsonl_bytes(core)
    prompt_bytes = jsonl_bytes(prompt)
    core_hours = sum(row["budget"]["completed_evaluation_s"] for row in core) / 3600
    prompt_hours = sum(row["budget"]["completed_evaluation_s"] for row in prompt) / 3600
    summary = {
        "schema_version": 2,
        "campaign_id": CAMPAIGN_ID,
        "requested_models": list(MODELS),
        "core_trajectories": len(core),
        "prompt_extension_trajectories": len(prompt),
        "total_trajectories": len(core) + len(prompt),
        "core_completed_evaluation_gpu_hours": core_hours,
        "prompt_extension_completed_evaluation_gpu_hours": prompt_hours,
        "total_completed_evaluation_gpu_hours": core_hours + prompt_hours,
        "gpu_assignment": "each eight-replicate cell has exactly two rows per GPU slot",
        "manifests": {
            "core_192.jsonl": sha256_bytes(core_bytes),
            "prompt_extension_128.jsonl": sha256_bytes(prompt_bytes),
        },
        "launch_state": "blocked_pending_validate_launch",
        "non_negotiable_blockers": [
            "both aliases resolved to provider-attested immutable model revisions",
            "four concrete distinct GPU UUIDs bound to the frozen slots",
            "prompt/system/tool hash lock verifies",
            "sum and SDPA robust gates completed and frozen",
            "hidden tuning and terminal holdout services bound to distinct datasets/principals",
            "provider credentials present without being serialized",
        ],
        "operation_budgets_s": OPERATIONS,
    }
    return {
        "core_192.jsonl": core_bytes,
        "prompt_extension_128.jsonl": prompt_bytes,
        "summary.json": json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path(__file__).with_name("manifests"))
    parser.add_argument("--check", action="store_true", help="verify existing files byte-for-byte")
    args = parser.parse_args()
    outputs = expected_outputs()
    if args.check:
        failures = [name for name, data in outputs.items() if not (args.outdir / name).is_file() or (args.outdir / name).read_bytes() != data]
        if failures:
            print(json.dumps({"ok": False, "mismatches": failures}, sort_keys=True))
            return 1
        core = [json.loads(line) for line in (args.outdir / "core_192.jsonl").read_text().splitlines()]
        prompt = [json.loads(line) for line in (args.outdir / "prompt_extension_128.jsonl").read_text().splitlines()]
        validate_manifest_rows(core, prompt)
        print(json.dumps({"ok": True, "files": sorted(outputs)}, sort_keys=True))
        return 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        (args.outdir / name).write_bytes(data)
    print((args.outdir / "summary.json").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
