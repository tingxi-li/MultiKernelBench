from __future__ import annotations

import unittest
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from . import protocol
except ImportError:
    import protocol


def hardware_binding(root: Path, compute_capability="9.0"):
    gpu = {
        "uuid": "GPU-test-non-ada",
        "name": "Test non-Ada GPU",
        "driver_version": "test-driver",
        "compute_capability": compute_capability,
    }
    toolchain = {
        "python": "test", "torch": "test", "triton": "test",
        "tilelang": "test", "nvcc": "test",
    }
    files = {
        "identity.json": json.dumps({"gpu_identity": gpu}, sort_keys=True).encode(),
        "toolchain.json": json.dumps({"toolchain": toolchain}, sort_keys=True).encode(),
        "support_probe.py": b"probe",
        "gate_lock.json": b"gate",
        "runner.py": b"runner",
    }
    files["inventory.json"] = json.dumps({"source_paths": sorted(files)}, sort_keys=True).encode()
    for name, data in files.items():
        (root / name).write_bytes(data)
    return {
        "schema_version": 1,
        "gpu_identity": gpu,
        "toolchain": toolchain,
        "source_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
        "closure_roles": {
            "hardware_identity_capture": "identity.json",
            "toolchain_capture": "toolchain.json",
            "support_probe": "support_probe.py",
            "gate_lock": "gate_lock.json",
            "runner": "runner.py",
            "dependency_inventory": "inventory.json",
        },
    }


def terminal_results(manifest: dict, evidence_root: Path) -> list[dict]:
    rows = []
    for index, cell in enumerate(manifest["rows"]):
        receipt = {
            "schema_version": 1,
            "record_type": "finite_frontier_f0_terminal_receipt",
            "cell_id": cell["cell_id"],
            "terminal_status": "GATE_PASSED",
            "hardware_binding_sha256": manifest["hardware_binding_sha256"],
            "timing_allowed": False,
        }
        data = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        relative = f"receipt_{index:03d}.json"
        (evidence_root / relative).write_bytes(data)
        rows.append({
            "cell_id": cell["cell_id"],
            "terminal_status": "GATE_PASSED",
            "evidence_path": relative,
            "evidence_sha256": hashlib.sha256(data).hexdigest(),
        })
    return rows


class ProtocolTests(unittest.TestCase):
    def test_f0_exact_304_cell_census(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        rows = protocol.make_f0_manifest(hardware_binding(root), root)["rows"]
        self.assertEqual(len(rows), 4 * 4 * 19)
        self.assertEqual(len({row["cell_id"] for row in rows}), 304)
        self.assertEqual(Counter(row["strategy"] for row in rows), {item: 76 for item in protocol.STRATEGIES})
        self.assertEqual(Counter(row["lane"] for row in rows), {item: 76 for item in protocol.LANES})
        self.assertEqual(Counter(row["grid_id"] for row in rows), {item: 16 for item in protocol.GRID_IDS})

    def test_f0_rejects_sm89_or_missing_compute_capability(self):
        for value in ("8.9", "sm_89", [8, 9], "8.90", [8, -1], None):
            with self.subTest(value=value), self.assertRaisesRegex(protocol.ProtocolError, "non-sm_89|explicit"):
                with TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    protocol.make_f0_manifest(hardware_binding(root, value), root)

    def test_f0_forbids_timing(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        manifest = protocol.make_f0_manifest(hardware_binding(root), root)
        self.assertFalse(manifest["timing_allowed"])
        self.assertNotIn("timing", manifest["allowed_stages"])
        self.assertTrue(all(row["timing_allowed"] is False for row in manifest["rows"]))
        evidence = root / "evidence"
        evidence.mkdir()
        binding = hardware_binding(root)
        manifest = protocol.make_f0_manifest(binding, root)
        results = terminal_results(manifest, evidence)
        self.assertEqual(protocol.validate_f0_results(binding, root, manifest, results, evidence)["GATE_PASSED"], 304)
        results[0]["latency_ms"] = 1.0
        with self.assertRaisesRegex(protocol.ProtocolError, "timing"):
            protocol.validate_f0_results(binding, root, manifest, results, evidence)
        results = terminal_results(manifest, evidence)
        fabricated = dict(manifest)
        fabricated["rows"] = [dict(row, cell_id=f"fabricated_{index}") for index, row in enumerate(manifest["rows"])]
        with self.assertRaisesRegex(protocol.ProtocolError, "differs"):
            protocol.validate_f0_results(binding, root, fabricated, results, evidence)

    def test_f0_verifies_material_closure(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        binding = hardware_binding(root)
        (root / "runner.py").write_bytes(b"changed")
        with self.assertRaisesRegex(protocol.ProtocolError, "hash mismatch"):
            protocol.make_f0_manifest(binding, root)

    def test_launch_is_fail_closed(self):
        self.assertEqual(protocol.f1_design_contract()["status"], "design_only_blocked")
        for experiment in ("F0", "F1"):
            with self.subTest(experiment=experiment), self.assertRaises(protocol.ProtocolError):
                protocol.authorize_launch(experiment)


if __name__ == "__main__":
    unittest.main()
