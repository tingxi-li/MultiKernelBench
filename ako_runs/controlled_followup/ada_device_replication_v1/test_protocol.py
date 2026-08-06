from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from . import protocol
except ImportError:
    import protocol


def make_contract(root: Path) -> dict:
    launch_lock = {
        "schema_version": 2,
        "campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
        "lock_stage": "campaign",
        "source_bundle_sha256": "a" * 64,
    }
    files = {
        "source.lock": b"source closure",
        "gate.lock": b"candidate-independent gate",
        "analyze.py": b"analyzer",
        "runner.py": b"runner",
        "launch_lock.json": (json.dumps(launch_lock, sort_keys=True) + "\n").encode(),
    }
    for relative, payload in files.items():
        (root / relative).write_bytes(payload)
    roles = dict(zip(protocol.DEPENDENCY_ROLES, files))
    return {
        "schema_version": 1,
        "campaign_id": protocol.CAMPAIGN_ID,
        "instrument_campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
        "reference_result_tag": protocol.REFERENCE_RESULT_TAG,
        "state": protocol.STATE,
        "claim_scope": protocol.CLAIM_SCOPE,
        "timing_allowed": False,
        "devices": [
            {
                "uuid": uuid,
                "name": protocol.GPU_NAME,
                "compute_capability": protocol.COMPUTE_CAPABILITY,
            }
            for uuid in reversed(protocol.GPU_UUIDS)
        ],
        "dependency_roles": roles,
        "dependency_sha256": {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in files.items()
        },
    }


def make_results(
    contract: dict, dependency_root: Path, manifest: dict, evidence_root: Path,
    instrument_root: Path, *, discordant: bool = False,
) -> list[dict]:
    manifest_sha256 = protocol._canonical_sha256(manifest)
    launch_lock_path = dependency_root / contract["dependency_roles"]["instrument_launch_lock"]
    launch_lock_sha256 = hashlib.sha256(launch_lock_path.read_bytes()).hexdigest()
    source_bundle_sha256 = json.loads(launch_lock_path.read_text())["source_bundle_sha256"]
    results = []
    result_material = []
    for device_index, (uuid, tag) in enumerate(zip(protocol.GPU_UUIDS, protocol.TAGS)):
        device_rows = [row for row in manifest["rows"] if row["device_uuid"] == uuid]
        tag_root = instrument_root / tag
        records_root = tag_root / "audit" / "records"
        gate_root = tag_root / "audit" / "gate"
        audit_receipts_root = tag_root / "audit" / "receipts"
        for path in (records_root, gate_root, audit_receipts_root):
            path.mkdir(parents=True, exist_ok=True)
        audit_receipts = {}
        for shard in range(4):
            assigned = [row["cell_id"] for index, row in enumerate(device_rows) if index % 4 == shard]
            audit_receipt = {
                "schema_version": 2,
                "record_type": "fused_crossed_v2_audit_receipt",
                "contract": {
                    "assigned_cell_ids": assigned,
                    "campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
                    "git_commit": "b" * 40,
                    "launch_lock_sha256": launch_lock_sha256,
                    "physical_gpu": device_index,
                    "shard_count": 4,
                    "shard_index": shard,
                    "source_bundle_sha256": source_bundle_sha256,
                },
                "gpu": {
                    "uuid": uuid,
                    "name": protocol.GPU_NAME,
                    "compute_cap": protocol.COMPUTE_CAPABILITY,
                    "index": str(device_index),
                },
            }
            data = (json.dumps(audit_receipt, sort_keys=True) + "\n").encode()
            relative = f"{tag}/audit/receipts/shard{shard:02d}.json"
            (instrument_root / relative).write_bytes(data)
            audit_receipts[shard] = (relative, hashlib.sha256(data).hexdigest())

        counts = Counter()
        for cell_index, row in enumerate(device_rows):
            status = "LAUNCH_FAILED" if discordant and device_index == 0 and cell_index == 0 else "BUILD_FAILED"
            cell = {
                "cell_id": row["cell_id"], "cell_index": cell_index,
                "strategy": row["strategy"], "lane": row["lane"],
                "grid_id": row["grid_id"], "requested": True, "support_declared": True,
            }
            record = {
                "schema_version": 2, "campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID,
                "cell": cell, "cell_sha256": protocol._canonical_sha256(cell),
                "launch_lock_sha256": launch_lock_sha256,
                "source_bundle_sha256": source_bundle_sha256,
                "physical_gpu": device_index, "build_attempted": True,
                "gate_attempted": status == "LAUNCH_FAILED", "terminal_outcome": status,
            }
            if status == "LAUNCH_FAILED":
                gate_relative = f"{tag}/audit/gate/{row['cell_id'].replace('.', '__')}.jsonl"
                (instrument_root / gate_relative).write_bytes(b"")
                record.update({
                    "gate_jsonl_path": gate_relative,
                    "gate_jsonl_sha256": hashlib.sha256(b"").hexdigest(),
                    "gate_summary": {"observed_records": 0, "failed_records": 0, "complete": False, "full_gate_pass": False},
                })
            record_data = (json.dumps(record, sort_keys=True) + "\n").encode()
            record_relative = f"{tag}/audit/records/{row['cell_id'].replace('.', '__')}.json"
            (instrument_root / record_relative).write_bytes(record_data)
            shard = cell_index % 4
            result_material.append((row, status, record_relative, hashlib.sha256(record_data).hexdigest(), *audit_receipts[shard]))
            counts[status] += 1

        summary = {
            "schema_version": 2, "record_type": "fused_crossed_v2_audit_summary",
            "campaign_id": protocol.INSTRUMENT_CAMPAIGN_ID, "complete": True,
            "requested_cells": protocol.CELLS_PER_DEVICE,
            "launch_lock_sha256": launch_lock_sha256,
            "source_bundle_sha256": source_bundle_sha256,
            "outcome_counts": {status: counts[status] for status in protocol.TERMINAL_STATUSES},
            "timing_eligible_cell_ids": [],
            "receipt_hashes": [{"path": path, "sha256": digest} for path, digest in audit_receipts.values()],
        }
        summary_data = (json.dumps(summary, sort_keys=True) + "\n").encode()
        (tag_root / "audit_summary.json").write_bytes(summary_data)

    for index, (row, status, record_path, record_sha, audit_path, audit_sha) in enumerate(result_material):
        summary_path = f"{row['campaign_tag']}/audit_summary.json"
        summary_sha = hashlib.sha256((instrument_root / summary_path).read_bytes()).hexdigest()
        provenance = {
            "instrument_record_path": record_path,
            "instrument_record_sha256": record_sha,
            "instrument_audit_receipt_path": audit_path,
            "instrument_audit_receipt_sha256": audit_sha,
            "instrument_audit_summary_path": summary_path,
            "instrument_audit_summary_sha256": summary_sha,
            "instrument_launch_lock_sha256": launch_lock_sha256,
        }
        receipt = {
            "schema_version": 1, "record_type": "ada_device_replication_v1_terminal_receipt",
            "campaign_id": protocol.CAMPAIGN_ID, "claim_scope": protocol.CLAIM_SCOPE,
            "contract_sha256": manifest["contract_sha256"], "manifest_sha256": manifest_sha256,
            "request_id": row["request_id"], "campaign_tag": row["campaign_tag"],
            "device_uuid": row["device_uuid"], "cell_id": row["cell_id"],
            "terminal_status": status, "timing_allowed": False, **provenance,
        }
        payload = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        relative = f"receipt_{index:04d}.json"
        (evidence_root / relative).write_bytes(payload)
        results.append({
            "request_id": row["request_id"], "terminal_status": status,
            "evidence_path": relative, "evidence_sha256": hashlib.sha256(payload).hexdigest(),
            **provenance,
        })
    return results


class ProtocolTests(unittest.TestCase):
    def test_exact_four_by_304_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = protocol.make_manifest(make_contract(root), root)
        self.assertEqual(manifest["requested_records"], 4 * 4 * 4 * 19)
        self.assertEqual(len(manifest["rows"]), 1216)
        self.assertEqual(
            {row["device_uuid"] for row in manifest["rows"]}, set(protocol.GPU_UUIDS)
        )
        self.assertFalse(manifest["architecture_factor_varied"])
        self.assertFalse(manifest["timing_allowed"])
        self.assertNotIn("timing", manifest["allowed_stages"])
        self.assertTrue(all(row["timing_allowed"] is False for row in manifest["rows"]))

    def test_contract_rejects_hardware_or_dependency_drift(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = make_contract(root)
            wrong_uuid = deepcopy(contract)
            wrong_uuid["devices"][0]["uuid"] = "GPU-not-bound"
            with self.assertRaisesRegex(protocol.ProtocolError, "UUID"):
                protocol.make_manifest(wrong_uuid, root)
            wrong_sku = deepcopy(contract)
            wrong_sku["devices"][0]["compute_capability"] = "9.0"
            with self.assertRaisesRegex(protocol.ProtocolError, "sm_89"):
                protocol.make_manifest(wrong_sku, root)
            (root / "runner.py").write_bytes(b"changed")
            with self.assertRaisesRegex(protocol.ProtocolError, "hash mismatch"):
                protocol.make_manifest(contract, root)

    def test_complete_receipts_and_discordance_are_retained(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            instrument = root / "instrument"
            evidence.mkdir()
            instrument.mkdir()
            contract = make_contract(root)
            manifest = protocol.make_manifest(contract, root)
            results = make_results(contract, root, manifest, evidence, instrument, discordant=True)
            summary = protocol.validate_results(contract, root, manifest, results, evidence, instrument)
            self.assertEqual(summary["status"], "noncontrolling_complete_census")
            self.assertEqual(summary["concordant_cells"], 303)
            self.assertEqual(summary["discordant_cells"], [manifest["rows"][0]["cell_id"]])
            self.assertEqual(summary["terminal_counts"]["BUILD_FAILED"], 1215)
            self.assertEqual(summary["terminal_counts"]["LAUNCH_FAILED"], 1)

            first = results[0]
            first["latency_ms"] = 1.0
            with self.assertRaisesRegex(protocol.ProtocolError, "timing"):
                protocol.validate_results(contract, root, manifest, results, evidence, instrument)
            del first["latency_ms"]
            first["terminal_status"] = "BUILD_FAILED"
            with self.assertRaisesRegex(protocol.ProtocolError, "differs from the instrument"):
                protocol.validate_results(contract, root, manifest, results, evidence, instrument)

    def test_manifest_tampering_and_launch_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "evidence"
            instrument = root / "instrument"
            evidence.mkdir()
            instrument.mkdir()
            contract = make_contract(root)
            manifest = protocol.make_manifest(contract, root)
            results = make_results(contract, root, manifest, evidence, instrument)
            fabricated = deepcopy(manifest)
            fabricated["rows"].pop()
            with self.assertRaisesRegex(protocol.ProtocolError, "manifest differs"):
                protocol.validate_results(contract, root, fabricated, results, evidence, instrument)
        with self.assertRaisesRegex(protocol.ProtocolError, "launch forbidden"):
            protocol.refuse_launch()


if __name__ == "__main__":
    unittest.main()
