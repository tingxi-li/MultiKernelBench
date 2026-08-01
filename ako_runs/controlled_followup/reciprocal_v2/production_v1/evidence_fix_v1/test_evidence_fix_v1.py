"""Tests for the append-only production-v1 evidence-builder correction."""
from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from .. import common, freeze as production_freeze
from . import capture_evidence, freeze


class EvidenceFixTests(unittest.TestCase):
    def test_parent_and_defective_builder_are_exactly_bound(self) -> None:
        parent = freeze.verify_parent()
        self.assertEqual(
            common.file_sha256(common.SOURCE_FREEZE), freeze.PARENT_FREEZE_SHA256
        )
        self.assertEqual(
            common.file_sha256(freeze.DEFECTIVE_BUILDER),
            freeze.DEFECTIVE_BUILDER_SHA256,
        )
        self.assertEqual(
            parent["source_sha256"][common.repo_path(freeze.DEFECTIVE_BUILDER)],
            freeze.DEFECTIVE_BUILDER_SHA256,
        )

    def test_fix_source_selection_and_pre_treatment_state(self) -> None:
        self.assertEqual(production_freeze.treatment_artifacts(), [])
        files = freeze.selected_source_files()
        self.assertEqual(len(files), 5)
        self.assertIn(freeze.HERE / "capture_evidence.py", files)
        self.assertNotIn(freeze.SOURCE_FREEZE, files)

    def test_corrected_writer_targets_bundle_not_last_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_bytes(b"source remains intact\n")
            bundle = root / "bundle.tar.gz"
            value = {
                "entries": [
                    {
                        "path": "source.txt",
                        "size": source.stat().st_size,
                        "sha256": common.file_sha256(source),
                    }
                ]
            }
            capture_evidence._write_bundle(bundle, [source], value)
            self.assertTrue(bundle.is_file())
            self.assertEqual(source.read_bytes(), b"source remains intact\n")
            with tarfile.open(bundle, "r:gz") as archive:
                self.assertEqual(
                    [member.name for member in archive.getmembers()],
                    ["EVIDENCE_MANIFEST.json", "source.txt"],
                )

    def test_fix_cli_imports_direct_and_as_package(self) -> None:
        package = (
            "ako_runs.controlled_followup.reciprocal_v2.production_v1."
            "evidence_fix_v1"
        )
        for name in ("freeze", "capture_evidence"):
            package_result = subprocess.run(
                [sys.executable, "-m", f"{package}.{name}", "--help"],
                cwd=common.REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(package_result.returncode, 0, package_result.stderr)
            direct_result = subprocess.run(
                [sys.executable, str(freeze.HERE / f"{name}.py"), "--help"],
                cwd=common.REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(direct_result.returncode, 0, direct_result.stderr)

    def test_fix_freeze_and_prereg_verify_when_present(self) -> None:
        if freeze.SOURCE_FREEZE.exists():
            receipt = freeze.verify()
            self.assertEqual(receipt["state"], "frozen_append_only_erratum")
        index_path = common.BASE / "evidence/prereg_v2.index.json"
        if index_path.exists():
            index = capture_evidence.verify(index_path)
            self.assertEqual(index["manifest"]["evidence_fix_id"], freeze.FIX_ID)


if __name__ == "__main__":
    unittest.main()
