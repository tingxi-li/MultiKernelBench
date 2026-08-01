#!/usr/bin/env python3
"""Create/check deterministic hash locks for the non-executed protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .campaign import CAMPAIGN_ID, sha256_file
except ImportError:  # direct script execution
    from campaign import CAMPAIGN_ID, sha256_file  # type: ignore


PROMPT_FILES = (
    "prompts/system.md",
    "prompts/neutral.md",
    "prompts/valid_mechanism_hint.md",
    "prompts/misleading_prior.md",
    "prompts/tool_contract.json",
)
GATE_FILES = (
    "gates/sum_gate_preregistration.json",
    "gates/sdpa_gate_preregistration.json",
)
PROTOCOL_FILES = PROMPT_FILES + GATE_FILES + (
    "manifests/core_192.jsonl",
    "manifests/prompt_extension_128.jsonl",
    "manifests/summary.json",
    "schemas/event.schema.json",
    "campaign.py",
    "make_manifest.py",
    "model_resolution.py",
    "provider_adapters.py",
    "hidden_gate.py",
    "controller.py",
    "analyze.py",
    "validate_launch.py",
)


def documents(base: Path) -> dict[Path, bytes]:
    def hashes(names: tuple[str, ...]) -> dict[str, str]:
        return {name: sha256_file(base / name) for name in names}

    prompt_lock = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "state": "frozen",
        "files": hashes(PROMPT_FILES),
        "mutation_policy": "any byte change requires a new campaign version",
    }
    gate_receipt = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "state": "preregistrations_frozen_not_executed",
        "files": hashes(GATE_FILES),
        "completed_gate_claimed": False,
    }
    protocol_receipt = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "state": "frozen_not_launch_ready",
        "files": hashes(PROTOCOL_FILES),
        "excludes_mutable_resolution_locks": True,
        "execution_claimed": False,
    }
    return {
        base / "locks/prompt_contract_lock.json": (json.dumps(prompt_lock, indent=2, sort_keys=True) + "\n").encode(),
        base / "gates/FREEZE_RECEIPT.json": (json.dumps(gate_receipt, indent=2, sort_keys=True) + "\n").encode(),
        base / "locks/protocol_freeze_receipt.json": (json.dumps(protocol_receipt, indent=2, sort_keys=True) + "\n").encode(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = documents(args.base)
    if args.check:
        mismatches = [str(path.relative_to(args.base)) for path, data in outputs.items() if not path.is_file() or path.read_bytes() != data]
        print(json.dumps({"ok": not mismatches, "mismatches": mismatches}, sort_keys=True))
        return 1 if mismatches else 0
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps({"written": [str(path.relative_to(args.base)) for path in outputs]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

