#!/usr/bin/env python3
"""Re-derive four frozen audit summaries and build the outer 1,216-row census."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from . import launch, protocol
except ImportError:
    import launch  # type: ignore
    import protocol  # type: ignore


HERE = Path(__file__).resolve().parent
OUTER_RESULTS = HERE / "results"


def _write_once(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise protocol.ProtocolError(f"cannot read retained output {path}: {exc}") from exc
        if observed != value:
            raise protocol.ProtocolError(f"retained immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def derive_instrument_summaries(roles: dict[str, Path]) -> list[Path]:
    summaries = []
    for tag in protocol.TAGS:
        root = launch.INSTRUMENT_RESULTS / tag
        for forbidden in ("screen", "confirmation"):
            if (root / forbidden).exists():
                raise protocol.ProtocolError(f"timing stage exists in no-timing tag: {tag}/{forbidden}")
        output = root / "audit_summary.json"
        temporary = output.with_name(f".{output.name}.partial.outer.{os.getpid()}")
        completed = subprocess.run(
            [
                sys.executable,
                str(roles["analyzer"]),
                "audit",
                "--result-root",
                str(root),
                "--out",
                str(temporary),
            ],
            cwd=launch.REPO_ROOT,
            check=False,
        )
        if completed.returncode:
            temporary.unlink(missing_ok=True)
            raise protocol.ProtocolError(f"frozen audit analyzer failed for {tag}")
        derived = launch.read_json(temporary)
        temporary.unlink(missing_ok=True)
        if output.exists():
            if launch.read_json(output) != derived:
                raise protocol.ProtocolError(f"retained audit summary differs from re-derivation: {tag}")
        else:
            _write_once(output, derived)
        summaries.append(output)
    return summaries


def collect_results(
    contract: dict[str, Any], manifest: dict[str, Any], roles: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_sha256 = protocol._canonical_sha256(manifest)
    lock_path = roles["instrument_launch_lock"]
    lock_sha256 = launch.file_sha256(lock_path)
    results: list[dict[str, Any]] = []
    summary_hashes = {
        tag: launch.file_sha256(launch.INSTRUMENT_RESULTS / tag / "audit_summary.json")
        for tag in protocol.TAGS
    }
    for row in manifest["rows"]:
        tag = row["campaign_tag"]
        record_relative = f"{tag}/audit/records/{row['cell_id'].replace('.', '__')}.json"
        record_path = launch.INSTRUMENT_RESULTS / record_relative
        record = launch.read_json(record_path)
        cell_index = record.get("cell", {}).get("cell_index")
        if not isinstance(cell_index, int) or isinstance(cell_index, bool):
            raise protocol.ProtocolError(f"instrument record lacks cell_index: {record_relative}")
        audit_relative = f"{tag}/audit/receipts/shard{cell_index % 4:02d}.json"
        audit_path = launch.INSTRUMENT_RESULTS / audit_relative
        summary_relative = f"{tag}/audit_summary.json"
        status = record.get("terminal_outcome")
        if status not in protocol.TERMINAL_STATUSES:
            raise protocol.ProtocolError(f"unknown instrument status: {record_relative}")
        provenance = {
            "instrument_record_path": record_relative,
            "instrument_record_sha256": launch.file_sha256(record_path),
            "instrument_audit_receipt_path": audit_relative,
            "instrument_audit_receipt_sha256": launch.file_sha256(audit_path),
            "instrument_audit_summary_path": summary_relative,
            "instrument_audit_summary_sha256": summary_hashes[tag],
            "instrument_launch_lock_sha256": lock_sha256,
        }
        receipt = {
            "schema_version": 1,
            "record_type": "ada_device_replication_v1_terminal_receipt",
            "campaign_id": protocol.CAMPAIGN_ID,
            "claim_scope": protocol.CLAIM_SCOPE,
            "contract_sha256": manifest["contract_sha256"],
            "manifest_sha256": manifest_sha256,
            "request_id": row["request_id"],
            "campaign_tag": tag,
            "device_uuid": row["device_uuid"],
            "cell_id": row["cell_id"],
            "terminal_status": status,
            "timing_allowed": False,
            **provenance,
        }
        receipt_relative = (
            f"terminal_receipts/{tag}/{row['cell_id'].replace('.', '__')}.json"
        )
        receipt_path = OUTER_RESULTS / receipt_relative
        _write_once(receipt_path, receipt)
        results.append(
            {
                "request_id": row["request_id"],
                "terminal_status": status,
                "evidence_path": receipt_relative,
                "evidence_sha256": launch.file_sha256(receipt_path),
                **provenance,
            }
        )
    summary = protocol.validate_results(
        contract,
        launch.REPO_ROOT,
        manifest,
        results,
        OUTER_RESULTS,
        launch.INSTRUMENT_RESULTS,
    )
    return results, summary


def _validate_launch_receipt(execution_lock: Path) -> dict[str, Any]:
    receipt_path = OUTER_RESULTS / "launch_receipt.json"
    receipt = launch.read_json(receipt_path)
    contract = receipt.get("contract", {})
    if (
        receipt.get("schema_version") != 1
        or receipt.get("record_type") != "ada_device_replication_v1_launch_receipt"
        or contract.get("campaign_id") != protocol.CAMPAIGN_ID
        or contract.get("execution_lock_sha256") != launch.file_sha256(execution_lock)
        or contract.get("schedule") != launch.schedule()
    ):
        raise protocol.ProtocolError("outer launch receipt lost its lock or schedule binding")
    for wave in range(4):
        status = launch.read_json(OUTER_RESULTS / "waves" / f"wave{wave}.json")
        if (
            status.get("complete") is not True
            or status.get("schedule") != [row for row in launch.schedule() if row["wave"] == wave]
            or status.get("returncodes") != [0, 0, 0, 0]
        ):
            raise protocol.ProtocolError(f"outer wave {wave} is incomplete")
    return receipt


def analyze(
    contract_path: Path,
    manifest_path: Path,
    execution_lock: Path,
    *,
    derive: bool,
    out: Path,
) -> dict[str, Any]:
    contract, manifest, roles, lock = launch.validated_inputs(contract_path, manifest_path)
    launch.validate_authorization(execution_lock, contract_path, manifest_path)
    launch_receipt = _validate_launch_receipt(execution_lock)
    if not derive:
        raise protocol.ProtocolError(
            "sealed analyzer re-derivation is mandatory; pass --derive"
        )
    derive_instrument_summaries(roles)
    results, validated = collect_results(contract, manifest, roles)
    result_index = OUTER_RESULTS / "terminal_results.json"
    _write_once(result_index, results)
    value = {
        **validated,
        "record_type": "ada_device_replication_v1_summary",
        "instrument_campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
        "reference_result_tag": protocol.REFERENCE_RESULT_TAG,
        "contract_path": launch.repo_path(contract_path),
        "contract_sha256": launch.file_sha256(contract_path),
        "manifest_path": launch.repo_path(manifest_path),
        "manifest_sha256": launch.file_sha256(manifest_path),
        "execution_lock_path": launch.repo_path(execution_lock),
        "execution_lock_sha256": launch.file_sha256(execution_lock),
        "outer_launch_receipt_path": launch.repo_path(OUTER_RESULTS / "launch_receipt.json"),
        "outer_launch_receipt_sha256": launch.file_sha256(OUTER_RESULTS / "launch_receipt.json"),
        "terminal_results_path": launch.repo_path(result_index),
        "terminal_results_sha256": launch.file_sha256(result_index),
        "instrument_launch_lock_sha256": launch.file_sha256(roles["instrument_launch_lock"]),
        "instrument_source_bundle_sha256": lock["source_bundle_sha256"],
        "instrument_audit_summaries": [
            {
                "device_uuid": protocol.GPU_UUIDS[gpu],
                "campaign_tag": tag,
                "path": launch.repo_path(launch.INSTRUMENT_RESULTS / tag / "audit_summary.json"),
                "sha256": launch.file_sha256(launch.INSTRUMENT_RESULTS / tag / "audit_summary.json"),
            }
            for gpu, tag in enumerate(protocol.TAGS)
        ],
        "launch_git_commit": launch_receipt["contract"]["git_commit"],
    }
    _write_once(out, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=HERE / "contract.json")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--execution-lock", type=Path, default=HERE / "execution_lock.json")
    parser.add_argument("--out", type=Path, default=OUTER_RESULTS / "summary.json")
    parser.add_argument("--derive", action="store_true")
    args = parser.parse_args()
    value = analyze(
        args.contract,
        args.manifest,
        args.execution_lock,
        derive=args.derive,
        out=args.out,
    )
    print(
        f"complete={value['complete_records']} concordant={value['concordant_cells']} "
        f"discordant={len(value['discordant_cells'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
