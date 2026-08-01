"""Fail-closed tests for the reciprocal-v2 production supplement."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from . import capture_evidence, common, freeze, isolation, kc_ladder, registry, treatment_plan


class ProductionSupplementTests(unittest.TestCase):
    maxDiff = None

    def test_legacy_freeze_and_prereg_v1_identities_are_exact(self) -> None:
        self.assertEqual(common.verify_legacy_identities(), common.EXPECTED_LEGACY)
        receipt = common.load_json(common.LEGACY_SOURCE_FREEZE)
        self.assertEqual(len(receipt["source_sha256"]), 61)
        self.assertEqual(
            common.canonical_sha256(receipt["source_sha256"]),
            common.EXPECTED_LEGACY["source_bundle_sha256"],
        )

    def test_treatment_documents_are_exact_and_non_claiming(self) -> None:
        documents = treatment_plan.documents()
        isolation_rows = [
            json.loads(payload)
            for path, payload in documents.items()
            if path.parent == common.ISOLATION_TEMPLATE_ROOT
        ]
        translation_rows = [
            json.loads(payload)
            for path, payload in documents.items()
            if common.TRANSLATION_REQUEST_ROOT in path.parents
        ]
        self.assertEqual(len(isolation_rows), 2)
        self.assertEqual(len(translation_rows), 48)
        self.assertTrue(
            all(row["state"] == "prepared_not_verified" for row in isolation_rows)
        )
        self.assertTrue(
            all(row["state"] == "unexecuted_treatment_request" for row in translation_rows)
        )
        self.assertEqual(len({row["cell_id"] for row in translation_rows}), 48)
        self.assertEqual(
            len({row["output_contract"]["source_path"] for row in translation_rows}),
            48,
        )
        treatment_plan.check()

    def test_kc_plan_is_literal_24_cell_fixed_ladder(self) -> None:
        plan = treatment_plan.kc_plan()
        self.assertEqual(plan["kc_ladder"], [8192, 4096, 2048, 1024, 512])
        self.assertEqual(plan["cell_count"], 24)
        self.assertEqual(len(plan["cells"]), 24)
        self.assertEqual(len({row["cell_key"] for row in plan["cells"]}), 24)
        self.assertTrue(
            all("__literal__" in row["literal_cell_id"] for row in plan["cells"])
        )
        self.assertEqual(plan["state"], "preregistered_not_executed")

    def test_probe_outcomes_are_raw_exit_code_derived(self) -> None:
        expected = isolation.expected_probe_outcomes()
        rows = [
            {
                "probe": name,
                "command": ["boundary-probe", name],
                "exit_code": 0 if accessible else 1,
            }
            for name, accessible in expected.items()
        ]
        self.assertEqual(isolation.derive_probe_outcomes(rows), expected)
        rows[0]["exit_code"] = True
        with self.assertRaises(isolation.IsolationError):
            isolation.derive_probe_outcomes(rows)

    def test_kc_aggregate_and_order_are_derived(self) -> None:
        cells = [
            {"cell_key": row["cell_key"], "all_required_cases_pass": True}
            for row in treatment_plan.kc_plan()["cells"]
        ]
        value = {
            "record_type": "reciprocal_v2_kc_attempt_summary",
            "campaign_id": common.base.CAMPAIGN_ID,
            "kc": 8192,
            "coverage_complete": True,
            "all_cells_all_v4_cases_pass": True,
            "cells": cells,
        }
        kc_ladder.validate_attempt(value, 8192)
        value["all_cells_all_v4_cases_pass"] = False
        with self.assertRaises(kc_ladder.KCLadderError):
            kc_ladder.validate_attempt(value, 8192)
        value["all_cells_all_v4_cases_pass"] = True
        value["cells"] = list(reversed(cells))
        with self.assertRaises(kc_ladder.KCLadderError):
            kc_ladder.validate_attempt(value, 8192)

    def test_missing_real_artifacts_fail_closed_without_success_locks(self) -> None:
        state = isolation.status()
        self.assertEqual(len(state["missing_transcripts"]), 2)
        self.assertFalse(state["isolation_lock_present"])
        with self.assertRaises((isolation.IsolationError, FileNotFoundError)):
            isolation.lock_document()
        registry_state = registry.status()
        self.assertEqual(len(registry_state["missing_sources"]), 48)
        self.assertEqual(len(registry_state["missing_receipts"]), 48)
        self.assertFalse(registry_state["ready"])
        with self.assertRaises((registry.RegistryError, isolation.IsolationError)):
            registry.document()
        self.assertEqual(kc_ladder.next_kc(), 8192)
        self.assertFalse(common.TRANSLATOR_ISOLATION_LOCK.exists())
        self.assertFalse(common.IMPLEMENTATION_REGISTRY.exists())
        self.assertFalse(common.RESOLUTION_LOCK.exists())

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=common.REPO_ROOT / "tmp" if (common.REPO_ROOT / "tmp").is_dir() else None) as directory:
            path = Path(directory) / "receipt.json"
            isolation.exclusive_json(path, {"state": "first"})
            with self.assertRaises(FileExistsError):
                isolation.exclusive_json(path, {"state": "second"})
            self.assertEqual(common.load_json(path), {"state": "first"})

    def test_json_schemas_parse(self) -> None:
        schemas = sorted(common.SCHEMA_ROOT.glob("*.json"))
        self.assertEqual(len(schemas), 6)
        for path in schemas:
            value = common.load_json(path)
            self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(value["type"], "object")

    def test_freeze_selection_is_complete_and_pre_treatment(self) -> None:
        self.assertEqual(freeze.treatment_artifacts(), [])
        files = freeze.selected_source_files()
        self.assertEqual(len(files), 67)
        self.assertEqual(len({common.repo_path(path) for path in files}), 67)
        self.assertIn(common.HERE / "README.md", files)
        self.assertIn(common.KC_PLAN, files)
        self.assertNotIn(common.SOURCE_FREEZE, files)

    def test_direct_and_package_cli_imports(self) -> None:
        names = (
            "treatment_plan",
            "isolation",
            "registry",
            "kc_ladder",
            "freeze",
            "capture_evidence",
        )
        package = "ako_runs.controlled_followup.reciprocal_v2.production_v1"
        for name in names:
            with self.subTest(mode="package", name=name):
                result = subprocess.run(
                    [sys.executable, "-m", f"{package}.{name}", "--help"],
                    cwd=common.REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            with self.subTest(mode="direct", name=name):
                result = subprocess.run(
                    [sys.executable, str(common.HERE / f"{name}.py"), "--help"],
                    cwd=common.REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_supplement_freeze_and_prereg_verify_when_present(self) -> None:
        if common.SOURCE_FREEZE.exists():
            receipt = freeze.verify()
            self.assertEqual(receipt["state"], "preregistered_no_treatment")
            self.assertEqual(receipt["legacy_identities"], common.EXPECTED_LEGACY)
        index_path = common.BASE / "evidence/prereg_v2.index.json"
        if index_path.exists():
            index = capture_evidence.verify(index_path)
            self.assertEqual(index["manifest"]["stage"], "prereg")
            self.assertFalse(index["manifest"]["treatment_artifacts_included"])


if __name__ == "__main__":
    unittest.main()
