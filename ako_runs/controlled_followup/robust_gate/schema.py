"""Small dependency-free validators and canonical JSON helpers.

The checked-in JSON Schema documents are the interchange contract.  These
runtime checks intentionally cover the invariants the command-line tools rely
on without requiring the optional ``jsonschema`` package on benchmark hosts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION


OPS = ("matmul", "fused_softmax", "sdpa")
SPLITS = ("calibration", "tuning", "validation", "performance")


class SchemaError(ValueError):
    """Raised when a campaign document violates the local schema contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load either a JSON array or newline-delimited JSON records."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise SchemaError(f"{path}: expected a JSON array")
        if not all(isinstance(record, dict) for record in value):
            raise SchemaError(f"{path}: every array record must be an object")
        return value
    records = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}:{lineno}: {exc}") from exc
        if not isinstance(value, dict):
            raise SchemaError(f"{path}:{lineno}: record must be an object")
        records.append(value)
    return records


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically write canonical, human-readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record, sort_keys=True, ensure_ascii=True, allow_nan=False
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _require(mapping: dict[str, Any], key: str, expected: type, where: str) -> Any:
    if key not in mapping:
        raise SchemaError(f"{where}: missing {key!r}")
    value = mapping[key]
    if not isinstance(value, expected):
        raise SchemaError(
            f"{where}.{key}: expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise SchemaError("manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"manifest.schema_version must be {SCHEMA_VERSION!r}"
        )
    _require(manifest, "campaign_id", str, "manifest")
    _require(manifest, "seed_namespace", str, "manifest")
    counts = _require(manifest, "split_counts", dict, "manifest")
    for split in SPLITS:
        count = counts.get(split)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SchemaError(f"manifest.split_counts.{split} must be a positive integer")
    operations = _require(manifest, "operations", dict, "manifest")
    if set(operations) != set(OPS):
        raise SchemaError(f"manifest.operations must contain exactly {OPS}")
    for op, spec in operations.items():
        where = f"manifest.operations.{op}"
        _require(spec, "shape", dict, where)
        _require(spec, "cpu_test_shape", dict, where)
        cases = _require(spec, "cases", list, where)
        if not cases:
            raise SchemaError(f"{where}.cases must not be empty")
        ids = []
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise SchemaError(f"{where}.cases[{index}] must be an object")
            ids.append(_require(case, "id", str, f"{where}.cases[{index}]"))
        if len(ids) != len(set(ids)):
            raise SchemaError(f"{where}.cases contains duplicate ids")
        gates = _require(spec, "gates", dict, where)
        if not gates:
            raise SchemaError(f"{where}.gates must not be empty")
        for gate_id, gate in gates.items():
            gate_where = f"{where}.gates.{gate_id}"
            _require(gate, "reference", dict, gate_where)
            anchors = _require(gate, "anchors", list, gate_where)
            metrics = _require(gate, "calibrated_metrics", list, gate_where)
            _require(gate, "fixed_thresholds", dict, gate_where)
            if (
                len(anchors) < 2
                or len(set(anchors)) != len(anchors)
                or not all(isinstance(v, str) and v for v in anchors)
            ):
                raise SchemaError(
                    f"{gate_where}.anchors must contain at least two distinct names"
                )
            if not metrics or not all(isinstance(v, str) for v in metrics):
                raise SchemaError(
                    f"{gate_where}.calibrated_metrics must be non-empty strings"
                )
    calibration = _require(manifest, "calibration", dict, "manifest")
    factor = calibration.get("safety_factor")
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor < 1:
        raise SchemaError("manifest.calibration.safety_factor must be >= 1")
    if calibration.get("rounding") != "next-1-2-5":
        raise SchemaError("manifest.calibration.rounding must be 'next-1-2-5'")


def validate_gate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise SchemaError("gate spec must be an object")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"gate_spec.schema_version must be {SCHEMA_VERSION!r}")
    _require(spec, "campaign_id", str, "gate_spec")
    digest = _require(spec, "manifest_sha256", str, "gate_spec")
    if len(digest) != 64:
        raise SchemaError("gate_spec.manifest_sha256 must be a SHA256 hex digest")
    gates = _require(spec, "gates", dict, "gate_spec")
    if not gates:
        raise SchemaError("gate_spec.gates must not be empty")
    for key, gate in gates.items():
        where = f"gate_spec.gates.{key}"
        _require(gate, "op", str, where)
        _require(gate, "gate_id", str, where)
        thresholds = _require(gate, "thresholds", dict, where)
        if not thresholds:
            raise SchemaError(f"{where}.thresholds must not be empty")
        for metric, threshold in thresholds.items():
            if not isinstance(threshold, dict) or threshold.get("comparison") != "le":
                raise SchemaError(f"{where}.thresholds.{metric} must use comparison='le'")
            value = threshold.get("value")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise SchemaError(f"{where}.thresholds.{metric}.value must be >= 0")
