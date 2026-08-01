#!/usr/bin/env python3
"""Generate non-claiming isolation, translation, and KC treatment requests."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:  # Support both ``python file.py`` and ``python -m package.module``.
    from . import common
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    import common


def translation_request_path(cell_id: str, translator: str) -> Path:
    return common.TRANSLATION_REQUEST_ROOT / translator / f"{cell_id}.json"


def implementation_source_path(cell_id: str, translator: str) -> Path:
    return common.BASE / "translators" / translator / "implementations" / f"{cell_id}.py"


def translation_receipt_path(cell_id: str, translator: str) -> Path:
    return common.TRANSLATION_RECEIPT_ROOT / translator / f"{cell_id}.json"


def isolation_template(translator: str) -> dict[str, Any]:
    other = next(item for item in common.base.TRANSLATORS if item != translator)
    return {
        "schema_version": 1,
        "record_type": "reciprocal_v2_isolation_request",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "translator": translator,
        "state": "prepared_not_verified",
        "legacy_source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_required_at_execution": common.repo_path(common.SOURCE_FREEZE),
        "source_root": f"ako_runs/controlled_followup/reciprocal_v2/translators/{translator}",
        "expected_transcript_path": common.repo_path(
            common.ISOLATION_TRANSCRIPT_ROOT / f"{translator}.json"
        ),
        "boundary_requirements": {
            "accepted_kinds": ["container", "mount_namespace", "distinct_os_user"],
            "repository_root_must_be_inaccessible": True,
            "peer_translator_root_must_be_inaccessible": True,
            "performance_results_must_be_inaccessible": True,
            "network_must_be_disabled": True,
            "own_output_root_must_be_writable": True,
            "distinct_execution_and_workspace_ids_required": True,
        },
        "required_probes": [
            {"probe": "repository_root_accessible", "required_exit_zero": False},
            {"probe": "other_translator_source_accessible", "required_exit_zero": False},
            {"probe": "performance_results_accessible", "required_exit_zero": False},
            {"probe": "network_accessible", "required_exit_zero": False},
            {"probe": "own_source_root_writable", "required_exit_zero": True},
        ],
        "unresolved_observations": None,
        "claim": "request_only_no_isolation_or_translation_has_occurred",
    }


def translation_request(job: dict[str, Any]) -> dict[str, Any]:
    cid, translator = job["cell_id"], job["translator"]
    retune_path = common.BASE / job["retune_plan"] if job["retune_plan"] else None
    return {
        "schema_version": 1,
        "record_type": "reciprocal_v2_translation_request",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "unexecuted_treatment_request",
        "cell_id": cid,
        "recipe_origin": job["recipe_origin"],
        "destination_dsl": job["destination_dsl"],
        "transfer_mode": job["transfer_mode"],
        "translator": translator,
        "recipe_card": job["recipe_card"],
        "retune_plan": (
            {"path": common.repo_path(retune_path), "sha256": common.file_sha256(retune_path)}
            if retune_path is not None
            else None
        ),
        "candidate_attempt_budget": job["candidate_attempt_budget"],
        "output_contract": {
            "source_path": common.repo_path(implementation_source_path(cid, translator)),
            "translation_receipt_path": common.repo_path(
                translation_receipt_path(cid, translator)
            ),
            "source_must_be_regular_nonempty_and_unique": True,
            "request_and_isolation_hash_bindings_required": True,
        },
        "access_policy": {
            "other_translator_source_must_be_inaccessible": True,
            "performance_results_must_be_inaccessible": True,
            "network_must_be_disabled": True,
        },
        "claim": "request_only_no_source_or_treatment_outcome_is_asserted",
    }


def kc_plan() -> dict[str, Any]:
    cells = []
    for ordinal, axes in enumerate(common.base.expected_resolution_cells()):
        origin, destination, translator = axes
        cells.append(
            {
                "ordinal": ordinal,
                "cell_key": common.base.resolution_cell_id(*axes),
                "literal_cell_id": common.base.cell_id(
                    origin, destination, "literal", translator
                ),
                "recipe_origin": origin,
                "destination_dsl": destination,
                "translator": translator,
            }
        )
    return {
        "schema_version": 1,
        "record_type": "reciprocal_v2_kc_resolution_plan",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "preregistered_not_executed",
        "kc_ladder": list(common.base.KC_LADDER),
        "selection_rule": "first_largest_kc_with_all_24_cells_passing_all_v4_cases",
        "request_rule": "only_next_kc_after_complete_failed_predecessor",
        "stop_rule": "stop_after_first_all_cell_pass_or_terminal_no_common_kc",
        "gate_lock_path": common.repo_path(common.base.GATE_LOCK),
        "gate_lock_sha256": common.file_sha256(common.base.GATE_LOCK),
        "cell_count": 24,
        "cells": cells,
        "raw_validation_required": True,
        "claim": "plan_only_no_gpu_evaluation_or_selected_kc_is_asserted",
    }


def documents() -> dict[Path, bytes]:
    manifest = common.load_json(common.BASE / "manifests/audit.json")
    result = {
        common.ISOLATION_TEMPLATE_ROOT / f"{translator}.json": common.stable_json_bytes(
            isolation_template(translator)
        )
        for translator in common.base.TRANSLATORS
    }
    for job in manifest["jobs"]:
        result[translation_request_path(job["cell_id"], job["translator"])] = (
            common.stable_json_bytes(translation_request(job))
        )
    result[common.KC_PLAN] = common.stable_json_bytes(kc_plan())
    return result


def write() -> None:
    for path, payload in documents().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def check() -> None:
    expected = documents()
    stale = [
        common.repo_path(path)
        for path, payload in expected.items()
        if not path.is_file() or path.read_bytes() != payload
    ]
    observed = set(common.REQUEST_ROOT.rglob("*.json"))
    if common.KC_PLAN.is_file():
        observed.add(common.KC_PLAN)
    extras = [common.repo_path(path) for path in sorted(observed - set(expected))]
    if stale or extras:
        raise SystemExit(f"treatment plans stale={stale}, extras={extras}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
