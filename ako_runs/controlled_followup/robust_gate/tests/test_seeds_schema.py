from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from ako_runs.controlled_followup.robust_gate.schema import (
    SchemaError,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    load_records,
    validate_gate_spec,
    validate_manifest,
)
from ako_runs.controlled_followup.robust_gate.seeds import derive_seed, tensor_seeds


HERE = Path(__file__).resolve().parents[1]


class SeedAndSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_json(HERE / "manifest.json")

    def test_manifest_is_valid(self) -> None:
        validate_manifest(self.manifest)
        matmul_v3 = load_json(HERE / "manifest_matmul_v3.json")
        matmul_v4 = load_json(HERE / "manifest_matmul_v4.json")
        validate_manifest(matmul_v3)
        validate_manifest(matmul_v4)
        self.assertEqual(
            canonical_sha256(self.manifest),
            "d017ef44cd63ce2a27bf31ca0ea7631a1f9810d3cbd25f1ddc6155b75a05634a",
        )
        self.assertNotEqual(canonical_sha256(self.manifest), canonical_sha256(matmul_v3))
        self.assertEqual(
            canonical_sha256(matmul_v4),
            "c26683f1eb2add7d5b0b57da9f2ff66473042437c05e6019eac7adb3672a7314",
        )
        self.assertEqual(matmul_v4["split_counts"]["calibration"], 640)
        self.assertEqual(matmul_v4["split_counts"]["validation"], 512)
        self.assertEqual(matmul_v4["calibration"]["safety_factor"], 1.25)
        self.assertNotEqual(matmul_v3["seed_namespace"], matmul_v4["seed_namespace"])

    def test_seed_golden_and_domain_separation(self) -> None:
        args = ("MKB-gate-v1", "matmul", "gaussian", "validation")
        self.assertEqual(derive_seed(*args, "a", 0), 4933128627332891978)
        self.assertEqual(derive_seed(*args, "b", 0), 289634988694793044)
        variants = {
            derive_seed("MKB-gate-v1", "matmul", "gaussian", split, tensor, index)
            for split in ("calibration", "validation")
            for tensor in ("a", "b")
            for index in (0, 1)
        }
        self.assertEqual(len(variants), 8)
        self.assertTrue(all(0 <= seed < 2**63 for seed in variants))

    def test_seed_domains_reject_ambiguous_or_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            derive_seed("bad\0namespace", "matmul", "case", "validation", "a", 0)
        with self.assertRaises(ValueError):
            derive_seed("MKB", "matmul", "", "validation", "a", 0)
        with self.assertRaises(ValueError):
            tensor_seeds(self.manifest, "matmul", "not-a-case", "validation", 0)

    def test_strict_json_rejects_nonfinite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json_bytes({"value": math.nan})
        bad_gate = {
            "schema_version": "1.0",
            "campaign_id": "x",
            "manifest_sha256": "0" * 64,
            "gates": {
                "matmul/q": {
                    "op": "matmul",
                    "gate_id": "q",
                    "thresholds": {
                        "max_abs_err": {"comparison": "le", "value": math.inf}
                    },
                }
            },
        }
        with self.assertRaises(SchemaError):
            validate_gate_spec(bad_gate)

    def test_json_array_records_must_be_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([{"ok": True}, 3]), encoding="utf-8")
            with self.assertRaises(SchemaError):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
