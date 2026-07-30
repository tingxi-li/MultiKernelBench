"""Deterministic domain-separated seeds for campaign tensors and job ordering."""

from __future__ import annotations

import argparse
import hashlib
from typing import Any

from . import SCHEMA_VERSION, SEED_ALGORITHM
from .schema import OPS, SPLITS, canonical_sha256, load_json, validate_manifest, write_json


TENSORS_BY_OP = {
    "matmul": ("a", "b"),
    "fused_softmax": ("x", "weight", "bias"),
    "sdpa": ("q", "k", "v"),
}


def derive_seed(
    namespace: str,
    op: str,
    case_id: str,
    split: str,
    tensor: str,
    index: int,
) -> int:
    """Return a stable non-negative 63-bit seed.

    NUL separators make domain components unambiguous.  Each tensor receives an
    independent stream, so changing input construction order cannot perturb
    existing operands.
    """
    for label, value in (("namespace", namespace), ("case_id", case_id)):
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError(f"{label} must be a non-empty NUL-free string")
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if tensor not in TENSORS_BY_OP[op] and not tensor.startswith("aux:"):
        raise ValueError(f"unknown tensor domain {tensor!r} for {op}")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("index must be a non-negative integer")
    parts = (namespace, op, case_id, split, tensor, str(index))
    payload = b"\0".join(part.encode("utf-8") for part in parts)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little") & ((1 << 63) - 1)


def tensor_seeds(
    manifest: dict[str, Any], op: str, case_id: str, split: str, index: int
) -> dict[str, int]:
    validate_manifest(manifest)
    known_cases = {
        case["id"] for case in manifest["operations"][op]["cases"]
    }
    if case_id not in known_cases:
        raise ValueError(f"unknown case {case_id!r} for {op}")
    return {
        tensor: derive_seed(
            manifest["seed_namespace"], op, case_id, split, tensor, index
        )
        for tensor in TENSORS_BY_OP[op]
    }


def resolve_seed_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_manifest(manifest)
    entries = []
    for op in OPS:
        for case in manifest["operations"][op]["cases"]:
            for split in SPLITS:
                for index in range(manifest["split_counts"][split]):
                    entries.append(
                        {
                            "op": op,
                            "case_id": case["id"],
                            "split": split,
                            "index": index,
                            "tensor_seeds": tensor_seeds(
                                manifest, op, case["id"], split, index
                            ),
                        }
                    )
    return {
        "schema_version": SCHEMA_VERSION,
        "seed_algorithm": SEED_ALGORITHM,
        "seed_namespace": manifest["seed_namespace"],
        "manifest_sha256": canonical_sha256(manifest),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    resolved = resolve_seed_manifest(manifest)
    write_json(args.out, resolved)
    print(f"wrote {len(resolved['entries'])} resolved seed rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
