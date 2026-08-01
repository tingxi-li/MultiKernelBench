#!/usr/bin/env python3
"""Fail-closed admission and terminal-report journal for reciprocal-v2 redesign.

The translation contract content-addresses source bytes but does not define how
to build or call those bytes.  This module therefore never guesses an execution
ABI.  It validates the 24-cell successor census and a future content-addressed
registry, and provides the append-only terminal journal the eventual executor
must use.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # Support direct and package execution.
    from .production_v1 import common, isolation
except ImportError:  # pragma: no cover - exercised by CLI test
    from production_v1 import common, isolation


HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "redesign_v1.json"
REGISTRY_PATH = HERE / "dependencies/redesign_v1_implementation_registry.json"
OUTCOME_ROOT = HERE / "results/redesign_v1/terminal_outcomes"
STAGES = ("audit", "screen", "primary")
OUTCOMES = (
    "BUILD_FAILED",
    "AUDIT_FAILED",
    "GATE_FAILED",
    "EXECUTION_FAILED",
    "AUDIT_ELIGIBLE",
    "SCREENED",
    "PRIMARY_MEASURED",
)
STAGE_OUTCOMES = {
    "audit": {"BUILD_FAILED", "AUDIT_FAILED", "GATE_FAILED", "EXECUTION_FAILED", "AUDIT_ELIGIBLE"},
    "screen": {"EXECUTION_FAILED", "SCREENED"},
    "primary": {"EXECUTION_FAILED", "PRIMARY_MEASURED"},
}
ABI_BLOCKER = (
    "translated sources have no frozen build/call/timing ABI; choose and freeze "
    "that contract before any implementation subprocess may start"
)


class RunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    value = common.load_json(path)
    design = value.get("default_design", {})
    analysis = value.get("analysis_policy", {})
    budget = value.get("total_budget", {})
    require(design.get("translators") == ["translator_a"], "translator census must be exactly translator_a")
    require(design.get("transfer_modes") == ["literal", "retuned"], "transfer-mode census drift")
    cells = len(design.get("origins", [])) * len(design.get("destinations", [])) * 2
    require(cells == design.get("cell_count") == 24, "corrective census must contain 24 cells")
    require(
        analysis.get("controlling_interaction_estimand")
        == "origin_by_destination_within_literal_arm",
        "literal-arm controlling estimand drift",
    )
    require(
        analysis.get("retuned_arm_role") == "descriptive_only"
        and analysis.get("retuned_null_may_support_compiler_effect_claim") is False,
        "retuned arm must remain descriptive",
    )
    audit = budget.get("literal_cells", -1) * budget.get("literal_attempts_per_cell", -1)
    audit += budget.get("retuned_cells", -1) * budget.get("retuned_attempts_per_cell", -1)
    require(
        budget.get("literal_cells") == budget.get("retuned_cells") == 12,
        "literal/retuned cell budget drift",
    )
    require(audit == budget.get("audit_candidate_attempts_ceiling") == 240, "audit budget drift")
    require(budget.get("screen_repetitions_per_audit_eligible_attempt") == 2, "screen budget drift")
    require(budget.get("screen_measurement_records_ceiling") == 2 * audit == 480, "screen ceiling drift")
    require(budget.get("primary_blocks") == 15, "primary block count drift")
    require(budget.get("primary_measurement_records") == cells * 15 == 360, "primary budget drift")
    require(budget.get("post_audit_timing_records_ceiling") == 480 + 360 == 840, "post-audit budget drift")
    require(
        budget.get("total_enumerated_execution_records_ceiling")
        == audit + 480 + 360
        == 1080,
        "total budget drift",
    )
    return value


def expected_cells(policy: dict[str, Any]) -> list[dict[str, Any]]:
    design = policy["default_design"]
    return [
        {
            "cell_id": f"{origin}__to__{destination}__{mode}__{translator}",
            "recipe_origin": origin,
            "destination_dsl": destination,
            "transfer_mode": mode,
            "translator": translator,
            "analysis_role": "controlling" if mode == "literal" else "descriptive_only",
        }
        for origin in design["origins"]
        for destination in design["destinations"]
        for mode in design["transfer_modes"]
        for translator in design["translators"]
    ]


def execution_ceiling(policy: dict[str, Any]) -> dict[str, int]:
    budget = policy["total_budget"]
    return {
        "audit": budget["audit_candidate_attempts_ceiling"],
        "screen": budget["screen_measurement_records_ceiling"],
        "primary": budget["primary_measurement_records"],
        "total": budget["total_enumerated_execution_records_ceiling"],
    }


def validate_registry(
    policy: dict[str, Any], path: Path = REGISTRY_PATH, *,
    policy_path: Path = POLICY_PATH, repo_root: Path = common.REPO_ROOT,
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"content-addressed implementation registry missing: {path}")
    value = common.load_json(path)
    expected = expected_cells(policy)
    require(
        value.get("schema_version") == 1
        and value.get("record_type") == "reciprocal_v2_redesign_implementation_registry"
        and value.get("campaign_id") == policy["campaign_id"]
        and value.get("state") == "frozen"
        and value.get("redesign_policy_sha256") == common.file_sha256(policy_path),
        "implementation registry header/policy binding mismatch",
    )
    rows = value.get("implementations")
    require(isinstance(rows, list) and value.get("implementation_count") == len(rows) == 24, "implementation census must be 24")
    require([row.get("cell_id") for row in rows] == [row["cell_id"] for row in expected], "implementation census/order mismatch")
    seen_paths: set[Path] = set()
    seen_inodes: set[tuple[int, int]] = set()
    by_digest: dict[str, list[str]] = defaultdict(list)
    root = repo_root.resolve()
    for row, cell in zip(rows, expected, strict=True):
        for key in ("cell_id", "recipe_origin", "destination_dsl", "transfer_mode", "translator"):
            require(row.get(key) == cell[key], f"{cell['cell_id']}: {key} mismatch")
        require(row.get("translator") == "translator_a", f"{cell['cell_id']}: fabricated translator level")
        source_value, digest = row.get("source"), row.get("sha256")
        require(isinstance(source_value, str) and isinstance(digest, str), f"{cell['cell_id']}: source binding missing")
        unresolved = repo_root / source_value
        source = unresolved.resolve()
        inode = (source.stat().st_dev, source.stat().st_ino) if source.is_file() else (-1, -1)
        translator_root = (repo_root / "ako_runs/controlled_followup/reciprocal_v2/translators/translator_a").resolve()
        require(
            unresolved.is_file()
            and not unresolved.is_symlink()
            and source.is_relative_to(root)
            and source.is_relative_to(translator_root)
            and source not in seen_paths
            and inode not in seen_inodes
            and common.file_sha256(source) == digest,
            f"{cell['cell_id']}: source is missing, reused, escaped, symlinked, or hash-stale",
        )
        seen_paths.add(source)
        seen_inodes.add(inode)
        by_digest[digest].append(cell["cell_id"])
    disclosed = [
        {"sha256": digest, "cell_ids": cell_ids}
        for digest, cell_ids in sorted(by_digest.items())
        if len(cell_ids) > 1
    ]
    require(value.get("identical_source_groups_disclosed") == disclosed, "identical source disclosure mismatch")
    require(value.get("translator_skill_bound_claimed") is False, "single translator cannot support a skill bound")
    return value


def _coordinate(policy: dict[str, Any], record: dict[str, Any]) -> tuple[str, str]:
    cells = {row["cell_id"]: row for row in expected_cells(policy)}
    cell = cells.get(record.get("cell_id"))
    require(cell is not None, "unknown corrective cell")
    require(record.get("analysis_role") == cell["analysis_role"], "analysis role mismatch")
    stage = record.get("stage")
    require(stage in STAGES, "unknown stage")
    attempt = record.get("attempt")
    attempt_limit = 1 if cell["transfer_mode"] == "literal" else 19
    require(type(attempt) is int and 0 <= attempt < attempt_limit, "attempt outside frozen budget")
    if stage == "audit":
        require(record.get("rep") is None and record.get("block") is None, "audit coordinate drift")
        suffix = f"attempt_{attempt:02d}"
    elif stage == "screen":
        rep = record.get("rep")
        require(type(rep) is int and 0 <= rep < 2 and record.get("block") is None, "screen coordinate drift")
        suffix = f"attempt_{attempt:02d}_rep_{rep}"
    else:
        block = record.get("block")
        require(type(block) is int and 0 <= block < 15 and record.get("rep") is None, "primary coordinate drift")
        suffix = f"block_{block:02d}"
    return stage, f"{cell['cell_id']}.{suffix}.json"


def retain_terminal_report(
    record: dict[str, Any], *, policy_path: Path = POLICY_PATH,
    registry_path: Path = REGISTRY_PATH, outcome_root: Path = OUTCOME_ROOT,
    repo_root: Path = common.REPO_ROOT,
) -> Path:
    """Exclusive-create one executor report; validation remains downstream."""
    policy = load_policy(policy_path)
    registry = validate_registry(
        policy, registry_path, policy_path=policy_path, repo_root=repo_root
    )
    implementations = {row["cell_id"]: row for row in registry["implementations"]}
    require(record.get("record_type") == "reciprocal_v2_redesign_terminal_report", "terminal record type mismatch")
    require(record.get("outcome") in OUTCOMES, "unknown terminal outcome")
    require(record.get("implementation_sha256") == implementations.get(record.get("cell_id"), {}).get("sha256"), "terminal implementation binding mismatch")
    require(record.get("redesign_policy_sha256") == common.file_sha256(policy_path), "terminal policy binding mismatch")
    require(record.get("implementation_registry_sha256") == common.file_sha256(registry_path), "terminal registry binding mismatch")
    require(isinstance(record.get("diagnostics"), dict), "terminal diagnostics must be retained")
    stage, name = _coordinate(policy, record)
    require(record["outcome"] in STAGE_OUTCOMES[stage], "outcome is invalid for stage")
    output = outcome_root / stage / name
    isolation.exclusive_json(output, record)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = load_policy()
        if args.plan:
            print(json.dumps({"cells": len(expected_cells(policy)), "execution_ceiling": execution_ceiling(policy)}, sort_keys=True))
            return 0
        validate_registry(policy, args.registry)
    except (OSError, ValueError, RunnerError) as exc:
        print("BLOCKED:", exc)
        print("REFUSED: no implementation subprocess started")
        return 2
    print("BLOCKED:", ABI_BLOCKER)
    print("REFUSED: no implementation subprocess started")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
