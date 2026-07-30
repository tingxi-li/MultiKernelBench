"""Immutable bindings and small I/O helpers for the v4 instrument audit."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from ako_runs.controlled_followup.robust_gate.schema import (
    canonical_sha256,
    file_sha256,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
MANIFEST_PATH = HERE / "manifest.json"
FREEZE_RECEIPT_PATH = HERE / "receipts" / "freeze_receipt.json"
LAUNCH_RECEIPT_PATH = HERE / "receipts" / "launch_receipt.json"


class AuditBindingError(ValueError):
    """Raised when frozen source or v4 evidence no longer matches its receipt."""


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise AuditBindingError(f"path escapes repository: {relative!r}") from exc
    return path


def audit_manifest() -> dict[str, Any]:
    value = load_json(MANIFEST_PATH)
    if value.get("campaign_id") != "controlled-followup-matmul-v4-instrument-audit-v1":
        raise AuditBindingError("unexpected audit campaign_id")
    if value.get("fixed_threshold_policy", {}).get("calibration_allowed") is not False:
        raise AuditBindingError("audit manifest does not forbid calibration")
    return value


def verify_original_v4(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify raw/canonical receipts and return the original gate object."""
    manifest = manifest or audit_manifest()
    originals = manifest["original_v4"]
    for label, spec in originals.items():
        path = repo_path(spec["path"])
        actual = file_sha256(path)
        if actual != spec["sha256"]:
            raise AuditBindingError(
                f"original v4 {label} raw SHA mismatch: {actual} != {spec['sha256']}"
            )
        if "canonical_sha256" in spec:
            canonical = canonical_sha256(load_json(path))
            if canonical != spec["canonical_sha256"]:
                raise AuditBindingError(
                    f"original v4 {label} canonical SHA mismatch: "
                    f"{canonical} != {spec['canonical_sha256']}"
                )
    gate = load_json(repo_path(originals["gate_spec"]["path"]))
    assert_registered_gate(manifest, gate)
    return gate


def assert_registered_gate(manifest: dict[str, Any], gate: dict[str, Any]) -> None:
    """Reject any in-memory gate whose canonical content differs from v4."""
    originals = manifest["original_v4"]
    actual = canonical_sha256(gate)
    expected = originals["gate_spec"]["canonical_sha256"]
    if actual != expected:
        raise AuditBindingError(f"gate content is not registered v4: {actual} != {expected}")
    if gate.get("manifest_sha256") != originals["manifest"]["canonical_sha256"]:
        raise AuditBindingError("original gate is not bound to the registered v4 manifest")
    expected_keys = {f"matmul/{gate_id}" for gate_id in manifest["gate_routes"]}
    if set(gate.get("gates", {})) != expected_keys:
        raise AuditBindingError("original gate keys differ from the audit routes")


def audit_manifest_hashes(manifest: dict[str, Any] | None = None) -> dict[str, str]:
    manifest = manifest or audit_manifest()
    return {
        "raw_sha256": file_sha256(MANIFEST_PATH),
        "canonical_sha256": canonical_sha256(manifest),
    }


def source_paths() -> list[Path]:
    """Runtime and audit sources that must be byte-frozen before a GPU launch."""
    robust = REPO_ROOT / "ako_runs" / "controlled_followup" / "robust_gate"
    phase1 = REPO_ROOT / "ako_runs" / "phase1_matmul"
    paths = [
        MANIFEST_PATH,
        robust / "__init__.py",
        robust / "audits" / "__init__.py",
        HERE / "__init__.py",
        HERE / "bindings.py",
        HERE / "runner.py",
        HERE / "analyze.py",
        HERE / "freeze.py",
        HERE / "README.md",
        HERE / "tests" / "__init__.py",
        HERE / "tests" / "test_audit.py",
        robust / "distributions.py",
        robust / "metrics.py",
        robust / "oracles.py",
        robust / "schema.py",
        robust / "seeds.py",
        phase1 / "common.py",
        phase1 / "variants" / "__init__.py",
        phase1 / "variants" / "triton_gemm.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AuditBindingError(f"source bundle is incomplete: {missing}")
    return paths


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def source_hashes() -> dict[str, str]:
    return {relative(path): file_sha256(path) for path in source_paths()}


def verify_freeze_receipt() -> dict[str, Any]:
    if not FREEZE_RECEIPT_PATH.is_file():
        raise AuditBindingError("GPU work requires receipts/freeze_receipt.json")
    receipt = load_json(FREEZE_RECEIPT_PATH)
    manifest = audit_manifest()
    verify_original_v4(manifest)
    expected_manifest = audit_manifest_hashes(manifest)
    if receipt.get("audit_manifest") != expected_manifest:
        raise AuditBindingError("freeze receipt audit-manifest binding mismatch")
    current = source_hashes()
    if receipt.get("source_files") != current:
        changed = sorted(
            key
            for key in set(current) | set(receipt.get("source_files", {}))
            if current.get(key) != receipt.get("source_files", {}).get(key)
        )
        raise AuditBindingError(f"source files changed after freeze: {changed}")
    if receipt.get("source_bundle_canonical_sha256") != canonical_sha256(current):
        raise AuditBindingError("freeze receipt source-bundle hash is inconsistent")
    return receipt


def verify_launch_receipt(arm: str, block_id: str | None, output: Path) -> dict[str, Any]:
    freeze = verify_freeze_receipt()
    if not LAUNCH_RECEIPT_PATH.is_file():
        raise AuditBindingError("GPU work requires receipts/launch_receipt.json")
    launch = load_json(LAUNCH_RECEIPT_PATH)
    if launch.get("freeze_receipt_sha256") != file_sha256(FREEZE_RECEIPT_PATH):
        raise AuditBindingError("launch receipt does not bind the current freeze receipt")
    resolved = output.resolve()
    matches = [
        item
        for item in launch.get("workloads", [])
        if item.get("arm") == arm
        and item.get("block_id") == block_id
        and repo_path(item["output"]) == resolved
    ]
    if len(matches) != 1:
        raise AuditBindingError("workload/output is not uniquely preregistered")
    if freeze.get("campaign_id") != launch.get("campaign_id"):
        raise AuditBindingError("launch and freeze campaign IDs differ")
    return launch


def threshold_failures(gate: dict[str, Any], metrics: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric, spec in gate["thresholds"].items():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{metric}=missing/nonfinite")
        elif value > spec["value"]:
            failures.append(f"{metric}={value:.17g}>{spec['value']:.17g}")
    return failures


class AtomicJsonl:
    """Append/fsync evidence into a retained .partial file, then rename once."""

    def __init__(self, target: Path):
        self.target = target
        self.partial = target.with_name(target.name + ".partial")
        if target.exists() or self.partial.exists():
            raise FileExistsError(f"refusing to overwrite {target} or {self.partial}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.partial, "x", encoding="utf-8")
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(
            json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1

    def finish(self) -> None:
        self.handle.close()
        os.replace(self.partial, self.target)

    def close_partial(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create a JSON document, refusing any existing target."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def raw_files_sha256(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        count = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        rows.append({"path": relative(path), "sha256": file_sha256(path), "records": count})
    return rows
