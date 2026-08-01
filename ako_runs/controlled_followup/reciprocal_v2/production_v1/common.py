"""Shared paths and hashing for the append-only production-v1 supplement."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO_ROOT = BASE.parents[2]
SUPPLEMENT_ID = "reciprocal-v2-production-v1-20260731"

_spec = importlib.util.spec_from_file_location("reciprocal_v2_frozen_protocol", BASE / "protocol.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("cannot load frozen reciprocal-v2 protocol")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

LEGACY_SOURCE_FREEZE = BASE / "receipts/source_freeze.json"
LEGACY_PREREG_INDEX = BASE / "evidence/prereg_v1.index.json"
LEGACY_PREREG_BUNDLE = BASE / "evidence/prereg_v1.tar.gz"
SOURCE_FREEZE = HERE / "receipts/source_freeze.json"
REQUEST_ROOT = HERE / "requests"
ISOLATION_TEMPLATE_ROOT = REQUEST_ROOT / "isolation"
TRANSLATION_REQUEST_ROOT = REQUEST_ROOT / "translations"
OUTPUT_ROOT = HERE / "outputs"
ISOLATION_TRANSCRIPT_ROOT = OUTPUT_ROOT / "isolation"
TRANSLATION_RECEIPT_ROOT = OUTPUT_ROOT / "translations"
KC_OUTPUT_ROOT = OUTPUT_ROOT / "kc_resolution"
KC_PLAN = HERE / "dependencies/kc_resolution_plan.json"
KC_EXECUTION_AUTHORIZATION = HERE / "dependencies/kc_execution_authorization.json"
SCHEMA_ROOT = HERE / "schemas"

# Success artifacts retain the frozen campaign's preregistered locations. They
# are absent until real treatment/GPU evidence satisfies this supplement.
TRANSLATOR_ISOLATION_LOCK = base.TRANSLATOR_ISOLATION_LOCK
IMPLEMENTATION_REGISTRY = base.IMPLEMENTATION_REGISTRY
RESOLUTION_LOCK = base.RESOLUTION_LOCK

EXPECTED_LEGACY = {
    "source_freeze_sha256": "6222c1ae56d795dd4f36ba928129a56862829f75f4bbe258285b91e98966b5c0",
    "source_bundle_sha256": "cde61903dca690d797076a79373445cca5453709e9a74f65823033826fbb8cff",
    "prereg_index_sha256": "f2ed15eb5d6bbe76ca52d949c59d354cc6205bdc53cf43fa9e96a93827139946",
    "prereg_bundle_sha256": "2de75d240cfee932d2ced6d435cc33c8a05efb380d0cee50303739498e51f615",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def verify_legacy_identities() -> dict[str, str]:
    observed = {
        "source_freeze_sha256": file_sha256(LEGACY_SOURCE_FREEZE),
        "source_bundle_sha256": load_json(LEGACY_SOURCE_FREEZE)["source_bundle_sha256"],
        "prereg_index_sha256": file_sha256(LEGACY_PREREG_INDEX),
        "prereg_bundle_sha256": file_sha256(LEGACY_PREREG_BUNDLE),
    }
    if observed != EXPECTED_LEGACY:
        raise RuntimeError(f"legacy reciprocal-v2 identities changed: {observed}")
    return observed
