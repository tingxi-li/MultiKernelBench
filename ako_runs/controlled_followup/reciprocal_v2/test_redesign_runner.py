from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

try:
    from . import redesign_runner as runner
except ImportError:  # pragma: no cover
    import redesign_runner as runner


class RedesignRunnerTest(unittest.TestCase):
    def _registry(self, root: Path, policy: dict) -> Path:
        implementations = []
        for cell in runner.expected_cells(policy):
            relative = Path(
                "ako_runs/controlled_followup/reciprocal_v2/translators/translator_a/implementations"
            ) / f"{cell['cell_id']}.py"
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# {cell['cell_id']}\n", encoding="utf-8")
            implementations.append(
                {
                    **{key: cell[key] for key in (
                        "cell_id", "recipe_origin", "destination_dsl",
                        "transfer_mode", "translator",
                    )},
                    "source": relative.as_posix(),
                    "sha256": runner.common.file_sha256(source),
                }
            )
        value = {
            "schema_version": 1,
            "record_type": "reciprocal_v2_redesign_implementation_registry",
            "campaign_id": policy["campaign_id"],
            "state": "frozen",
            "redesign_policy_sha256": runner.common.file_sha256(runner.POLICY_PATH),
            "implementation_count": 24,
            "translator_skill_bound_claimed": False,
            "identical_source_groups_disclosed": [],
            "implementations": implementations,
        }
        path = root / "registry.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_policy_is_24_cell_single_translator_and_runner_fails_closed(self) -> None:
        policy = runner.load_policy()
        cells = runner.expected_cells(policy)
        self.assertEqual(len(cells), 24)
        self.assertEqual({row["translator"] for row in cells}, {"translator_a"})
        self.assertEqual(
            {row["analysis_role"] for row in cells if row["transfer_mode"] == "literal"},
            {"controlling"},
        )
        self.assertEqual(runner.execution_ceiling(policy), {
            "audit": 240, "screen": 480, "primary": 360, "total": 1080,
        })
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            status = runner.main(["--registry", str(Path(directory) / "missing.json")])
        self.assertEqual(status, 2)
        self.assertIn("content-addressed implementation registry missing", output.getvalue())
        self.assertIn("no implementation subprocess started", output.getvalue())
        with mock.patch.object(runner, "validate_registry", return_value={}), \
             redirect_stdout(io.StringIO()) as output:
            status = runner.main([])
        self.assertEqual(status, 2)
        self.assertIn(runner.ABI_BLOCKER, output.getvalue())

    def test_terminal_reports_are_content_bound_and_never_overwritten(self) -> None:
        policy = runner.load_policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self._registry(root, policy)
            first = runner.expected_cells(policy)[0]
            report = {
                "record_type": "reciprocal_v2_redesign_terminal_report",
                "cell_id": first["cell_id"],
                "stage": "audit",
                "attempt": 0,
                "rep": None,
                "block": None,
                "analysis_role": "controlling",
                "outcome": "BUILD_FAILED",
                "implementation_sha256": json.loads(registry.read_text())["implementations"][0]["sha256"],
                "redesign_policy_sha256": runner.common.file_sha256(runner.POLICY_PATH),
                "implementation_registry_sha256": runner.common.file_sha256(registry),
                "diagnostics": {"stderr": "compiler failure retained"},
            }
            output = runner.retain_terminal_report(
                report, registry_path=registry, outcome_root=root / "outcomes", repo_root=root
            )
            self.assertEqual(json.loads(output.read_text()), report)
            with self.assertRaises(FileExistsError):
                runner.retain_terminal_report(
                    report, registry_path=registry, outcome_root=root / "outcomes", repo_root=root
                )


if __name__ == "__main__":
    unittest.main()
