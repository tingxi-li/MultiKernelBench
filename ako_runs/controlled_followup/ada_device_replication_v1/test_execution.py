from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from . import launch, protocol
except ImportError:
    import launch  # type: ignore
    import protocol  # type: ignore


class ExecutionScaffoldTests(unittest.TestCase):
    def test_cyclic_schedule_covers_each_shard_once_per_wave_and_device(self):
        rows = launch.schedule()
        self.assertEqual(len(rows), 16)
        for wave in range(4):
            block = [row for row in rows if row["wave"] == wave]
            self.assertEqual({row["physical_gpu"] for row in block}, set(range(4)))
            self.assertEqual({row["shard_index"] for row in block}, set(range(4)))
        for gpu in range(4):
            block = [row for row in rows if row["physical_gpu"] == gpu]
            self.assertEqual({row["shard_index"] for row in block}, set(range(4)))
            self.assertEqual({row["campaign_tag"] for row in block}, {protocol.TAGS[gpu]})

    def test_real_instrument_contract_and_manifest_rederive(self):
        with TemporaryDirectory(dir=launch.HERE) as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            manifest_path = root / "manifest.json"
            contract, manifest = launch.prepare(
                contract_path, manifest_path, write=True
            )
            observed_contract, observed_manifest, roles, lock = launch.validated_inputs(
                contract_path, manifest_path
            )
        self.assertEqual(observed_contract, contract)
        self.assertEqual(observed_manifest, manifest)
        self.assertEqual(manifest["requested_records"], 1216)
        self.assertEqual(lock["campaign_id"], protocol.INSTRUMENT_CAMPAIGN_ID)
        self.assertEqual(roles["runner"], launch.SEALED_ROOT / "campaign_runner.py")

    def test_execution_lock_is_explicit_and_timing_forbidden(self):
        with TemporaryDirectory(dir=launch.HERE) as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            manifest_path = root / "manifest.json"
            launch.prepare(contract_path, manifest_path, write=True)
            lock = {
                **launch.authorization_bindings(contract_path, manifest_path),
                "authorization_basis": "test-only authorization",
                "authorized_at_utc": "2026-08-06T00:00:00+00:00",
            }
            path = root / "execution_lock.json"
            path.write_text(json.dumps(lock), encoding="utf-8")
            self.assertEqual(
                launch.validate_authorization(path, contract_path, manifest_path), lock
            )
            lock["authorized"] = False
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(protocol.ProtocolError, "bindings differ"):
                launch.validate_authorization(path, contract_path, manifest_path)
        self.assertFalse(any("screen" in str(item) or "confirmation" in str(item) for item in launch.schedule()))

    def test_physical_gpu_mapping_is_exact(self):
        lines = [
            f"{index}, {uuid}, {protocol.GPU_NAME}, {protocol.COMPUTE_CAPABILITY}"
            for index, uuid in enumerate(protocol.GPU_UUIDS)
        ]
        self.assertEqual(
            [row["uuid"] for row in launch.validate_gpu_inventory(lines)],
            list(protocol.GPU_UUIDS),
        )
        swapped = list(lines)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaisesRegex(protocol.ProtocolError, "mapping changed"):
            launch.validate_gpu_inventory(swapped)

    def test_failed_wave_receipt_does_not_claim_completion(self):
        with TemporaryDirectory(dir=launch.HERE) as temporary:
            original = launch.OUTER_RESULTS
            launch.OUTER_RESULTS = Path(temporary)
            try:
                rows = [row for row in launch.schedule() if row["wave"] == 0]
                launch._failure_receipt(0, rows, [0, 1, 0, 0])
                receipts = list((launch.OUTER_RESULTS / "waves" / "failures").glob("*.json"))
                self.assertEqual(len(receipts), 1)
                self.assertFalse(json.loads(receipts[0].read_text())["complete"])
                self.assertFalse((launch.OUTER_RESULTS / "waves" / "wave0.json").exists())
            finally:
                launch.OUTER_RESULTS = original


if __name__ == "__main__":
    unittest.main()
