#!/usr/bin/env python3
"""Fail-closed static and launch-readiness checks for reciprocal v2."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import make_manifests
import make_retune_plans
import protocol


MANIFESTS = {
    "audit": make_manifests.AUDIT_MANIFEST,
    "primary": make_manifests.PRIMARY_MANIFEST,
}
SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _file_digest(path: Path) -> str | None:
    return protocol.file_sha256(path) if path.is_file() else None


def validate_static() -> dict[str, Any]:
    make_manifests.check()
    make_retune_plans.check()
    card = protocol.load_json(make_manifests.CUDA_CARD)
    _require(card["recipe_id"] == protocol.ORIGINS[2], "CUDA card identity drift")
    _require(
        card["correctness_amendment"]["candidate_kc_order"]
        == list(protocol.KC_LADDER),
        "CUDA card KC ladder drift",
    )
    documents: dict[str, Any] = {"cuda_card": card, "manifests": {}}
    expected = protocol.expected_cells()
    orders = protocol.block_orders()
    _require(len(orders) == protocol.CONFIRM_REPS, "confirmation block count drift")
    expected_ids = {protocol.cell_id(*axes) for axes in expected}
    _require(
        all(len(order) == 48 and set(order) == expected_ids for order in orders),
        "randomized block orders are not complete permutations",
    )
    _require(len({tuple(order) for order in orders}) == 15, "block orders repeat")
    for kind, path in MANIFESTS.items():
        value = protocol.load_json(path)
        _require(value.get("campaign_id") == protocol.CAMPAIGN_ID, "campaign drift")
        _require(value.get("manifest_kind") == kind, f"{kind}: kind drift")
        _require(value.get("job_count") == 48, f"{kind}: expected 48 cells")
        jobs = value.get("jobs")
        _require(isinstance(jobs, list) and len(jobs) == 48, f"{kind}: bad jobs")
        got = [
            (
                row.get("recipe_origin"),
                row.get("destination_dsl"),
                row.get("transfer_mode"),
                row.get("translator"),
            )
            for row in jobs
        ]
        _require(got == expected, f"{kind}: incomplete or reordered factorial")
        _require(len({row["job_id"] for row in jobs}) == 48, f"{kind}: job collision")
        for ordinal, row in enumerate(jobs):
            _require(row["ordinal"] == ordinal, f"{kind}: ordinal drift")
            _require(
                row["gate_in_loop"]["kc_ladder"] == list(protocol.KC_LADDER),
                f"{kind}: KC ladder drift",
            )
            attempts = protocol.RETUNE_ATTEMPTS if row["transfer_mode"] == "retuned" else 1
            _require(
                row["candidate_attempt_budget"] == attempts,
                f"{kind}: candidate budget drift",
            )
            _require(
                row["translator_source_root"]
                == f"translators/{row['translator']}",
                f"{kind}: translator source crossing",
            )
            binding = row["recipe_card"]
            card_path = protocol.REPO_ROOT / binding["path"]
            _require(card_path.is_file(), f"{kind}: missing recipe card")
            _require(
                protocol.file_sha256(card_path) == binding["file_sha256"],
                f"{kind}: recipe bytes drift",
            )
            _require(
                protocol.canonical_sha256(protocol.load_json(card_path))
                == binding["sha256"],
                f"{kind}: recipe semantics drift",
            )
            card_value = protocol.load_json(card_path)
            _require(
                card_value.get("recipe_id") == row["recipe_origin"],
                f"{kind}: recipe identity mismatch",
            )
            workload = card_value.get("workload", {})
            _require(
                workload.get("operation") == "matmul"
                and workload.get("shape") == {"M": 2048, "K": 8192, "N": 4096}
                and workload.get("output_dtype") == "fp32",
                f"{kind}: recipe workload/contract drift",
            )
            for evidence in card_value.get("evidence", []):
                evidence_path = protocol.REPO_ROOT / evidence.get("path", "")
                _require(
                    evidence_path.is_file()
                    and evidence.get("sha256") == protocol.file_sha256(evidence_path),
                    f"{kind}: recipe evidence drift",
                )
        documents["manifests"][kind] = value
    return documents


def _json_or_block(path: Path, label: str, blockers: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        blockers.append(f"{label} is missing: {protocol.repo_path(path)}")
        return None
    try:
        value = protocol.load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(f"{label} is unreadable: {exc}")
        return None
    if not isinstance(value, dict):
        blockers.append(f"{label} is not a JSON object")
        return None
    return value


def _safe_bound_file(path_value: Any, digest: Any, label: str, blockers: list[str]) -> Path | None:
    if not isinstance(path_value, str) or not isinstance(digest, str):
        blockers.append(f"{label} lacks a path/hash binding")
        return None
    path = (protocol.REPO_ROOT / path_value).resolve()
    try:
        path.relative_to(protocol.REPO_ROOT.resolve())
    except ValueError:
        blockers.append(f"{label} path escapes repository")
        return None
    if not path.is_file() or protocol.file_sha256(path) != digest:
        blockers.append(f"{label} is missing or hash-stale")
        return None
    return path


def _source_freeze_blockers(blockers: list[str]) -> dict[str, Any] | None:
    freeze = _json_or_block(protocol.SOURCE_FREEZE, "source freeze receipt", blockers)
    if freeze is None:
        return None
    if freeze.get("campaign_id") != protocol.CAMPAIGN_ID:
        blockers.append("source freeze campaign mismatch")
    sources = freeze.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        blockers.append("source freeze has no source map")
        return freeze
    if protocol.canonical_sha256(sources) != freeze.get("source_bundle_sha256"):
        blockers.append("source freeze bundle hash mismatch")
    for relative, expected in sources.items():
        _safe_bound_file(relative, expected, f"frozen source {relative}", blockers)
    if freeze.get("block_order_sha256") != protocol.block_order_sha256():
        blockers.append("source freeze block-order binding mismatch")
    return freeze


def _registry_blockers(
    manifest: dict[str, Any], blockers: list[str]
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    registry = _json_or_block(protocol.IMPLEMENTATION_REGISTRY, "implementation registry", blockers)
    observed: dict[str, dict[str, Any]] = {}
    if registry is None:
        return None, observed
    if registry.get("campaign_id") != protocol.CAMPAIGN_ID or registry.get("state") != "frozen":
        blockers.append("implementation registry campaign/state mismatch")
    if registry.get("translator_isolation_sha256") != _file_digest(protocol.TRANSLATOR_ISOLATION_LOCK):
        blockers.append("implementation registry does not bind translator isolation")
    entries = registry.get("implementations")
    if not isinstance(entries, list):
        blockers.append("implementation registry lacks implementations")
        return registry, observed
    jobs = {row["cell_id"]: row for row in manifest["jobs"]}
    for entry in entries:
        cid = entry.get("cell_id")
        if cid not in jobs or cid in observed:
            blockers.append("implementation registry has an unknown or duplicate cell")
            continue
        job = jobs[cid]
        translator = entry.get("translator")
        if translator != job["translator"]:
            blockers.append(f"{cid}: implementation translator mismatch")
            continue
        for key in ("recipe_origin", "destination_dsl", "transfer_mode"):
            if entry.get(key) != job[key]:
                blockers.append(f"{cid}: implementation {key} mismatch")
        source = entry.get("source")
        prefix = f"ako_runs/controlled_followup/reciprocal_v2/translators/{translator}/"
        if not isinstance(source, str) or not source.startswith(prefix):
            blockers.append(f"{cid}: implementation escapes translator root")
            continue
        path = _safe_bound_file(source, entry.get("sha256"), f"{cid} implementation", blockers)
        if path is not None:
            translator_root = (protocol.HERE / "translators" / translator).resolve()
            try:
                path.relative_to(translator_root)
            except ValueError:
                blockers.append(f"{cid}: implementation symlink/path escapes translator root")
                continue
            observed[cid] = entry
    if set(observed) != set(jobs):
        blockers.append(
            f"implementation registry census mismatch: expected=48 observed={len(observed)}"
        )
    return registry, observed


def _translator_isolation_blockers(blockers: list[str]) -> dict[str, Any] | None:
    isolation = _json_or_block(
        protocol.TRANSLATOR_ISOLATION_LOCK, "translator isolation lock", blockers
    )
    if isolation is None:
        return None
    if isolation.get("campaign_id") != protocol.CAMPAIGN_ID or isolation.get("state") != "frozen":
        blockers.append("translator isolation campaign/state mismatch")
    if isolation.get("source_freeze_sha256") != _file_digest(protocol.SOURCE_FREEZE):
        blockers.append("translator isolation does not bind source freeze")
    entries = isolation.get("translators")
    if not isinstance(entries, list) or [row.get("translator") for row in entries] != list(protocol.TRANSLATORS):
        blockers.append("translator isolation must bind translator_a then translator_b")
        return isolation
    worktree_ids = set()
    for entry in entries:
        translator = entry["translator"]
        if entry.get("source_root") != f"ako_runs/controlled_followup/reciprocal_v2/translators/{translator}":
            blockers.append(f"{translator}: isolated source root mismatch")
        worktree_id = entry.get("worktree_id")
        if not isinstance(worktree_id, str) or not worktree_id or worktree_id in worktree_ids:
            blockers.append(f"{translator}: worktree identity missing or reused")
        else:
            worktree_ids.add(worktree_id)
        if entry.get("other_translator_source_accessible") is not False:
            blockers.append(f"{translator}: other translator source was not isolated")
        transcript = _safe_bound_file(
            entry.get("isolation_transcript_path"), entry.get("isolation_transcript_sha256"),
            f"{translator} isolation transcript", blockers,
        )
        if transcript is None:
            continue
        summary = protocol.load_json(transcript)
        if not (
            summary.get("campaign_id") == protocol.CAMPAIGN_ID
            and summary.get("translator") == translator
            and summary.get("worktree_id") == worktree_id
            and summary.get("other_translator_source_accessible") is False
            and summary.get("performance_results_accessible") is False
        ):
            blockers.append(f"{translator}: isolation transcript does not substantiate boundary")
    return isolation


def _launch_receipt_blockers(stage: str, blockers: list[str]) -> dict[str, Any] | None:
    path = protocol.HERE / "results" / stage / "launch_receipt.json"
    receipt = _json_or_block(path, f"{stage} launch receipt", blockers)
    if receipt is None:
        return None
    payload = receipt.get("payload")
    runner_path = _safe_bound_file(
        receipt.get("runner_path"), receipt.get("runner_sha256"),
        f"{stage} execution runner", blockers,
    )
    expected_kind = "audit" if stage == "audit" else "primary"
    if not (
        receipt.get("campaign_id") == protocol.CAMPAIGN_ID
        and receipt.get("stage") == stage
        and receipt.get("full_census") is True
        and receipt.get("driver_preflight_passed") is True
        and isinstance(payload, dict)
        and receipt.get("payload_sha256") == protocol.canonical_sha256(payload)
        and payload.get("manifest_sha256") == protocol.file_sha256(MANIFESTS[expected_kind])
        and payload.get("source_freeze_sha256") == _file_digest(protocol.SOURCE_FREEZE)
        and payload.get("block_order_sha256") == protocol.block_order_sha256()
        and receipt.get("prelaunch_provenance_sha256") == _file_digest(protocol.PROVENANCE_LOCK)
        and runner_path is not None
    ):
        blockers.append(f"{stage} launch receipt binding mismatch")
    return receipt


def _resolution_blockers(
    registry: dict[str, Any] | None,
    implementations: dict[str, dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any] | None:
    resolution = _json_or_block(protocol.RESOLUTION_LOCK, "recipe-resolution lock", blockers)
    if resolution is None:
        return None
    if resolution.get("campaign_id") != protocol.CAMPAIGN_ID or resolution.get("state") != "frozen":
        blockers.append("recipe-resolution lock campaign/state mismatch")
    if resolution.get("kc_ladder") != list(protocol.KC_LADDER):
        blockers.append("recipe-resolution KC ladder/order mismatch")
    if resolution.get("gate_lock_sha256") != protocol.file_sha256(protocol.GATE_LOCK):
        blockers.append("recipe-resolution does not bind the v4 gate lock")
    if registry is None or resolution.get("implementation_registry_sha256") != protocol.file_sha256(protocol.IMPLEMENTATION_REGISTRY):
        blockers.append("recipe-resolution does not bind the implementation registry")
    selected = resolution.get("selected_kc")
    if selected not in protocol.KC_LADDER:
        blockers.append("recipe-resolution lock has no registered KC")
        return resolution
    attempts = resolution.get("attempts")
    selected_position = protocol.KC_LADDER.index(selected)
    if not isinstance(attempts, list) or [row.get("kc") for row in attempts] != list(protocol.KC_LADDER[: selected_position + 1]):
        blockers.append("recipe-resolution attempts are not the exact ladder prefix through selected KC")
        return resolution
    expected_resolution = [
        protocol.resolution_cell_id(*axes) for axes in protocol.expected_resolution_cells()
    ]
    literal_sources = {
        protocol.resolution_cell_id(job["recipe_origin"], job["destination_dsl"], job["translator"]): implementations.get(job["cell_id"], {}).get("sha256")
        for job in protocol.load_json(make_manifests.AUDIT_MANIFEST)["jobs"]
        if job["transfer_mode"] == "literal"
    }
    all_pass_values: list[bool] = []
    for attempt in attempts:
        cells = attempt.get("cells")
        if not isinstance(cells, list) or [cell.get("cell_key") for cell in cells] != expected_resolution:
            blockers.append(f"recipe-resolution KC={attempt.get('kc')} has incomplete/reordered 24-cell evidence")
            all_pass_values.append(False)
            continue
        cell_passes = []
        for cell in cells:
            key = cell["cell_key"]
            if cell.get("implementation_sha256") != literal_sources.get(key):
                blockers.append(f"recipe-resolution {key} implementation hash mismatch")
            summary_path = _safe_bound_file(
                cell.get("v4_summary_path"), cell.get("v4_summary_sha256"),
                f"recipe-resolution {key} v4 summary", blockers,
            )
            passed = False
            if summary_path is not None:
                summary = protocol.load_json(summary_path)
                passed = bool(
                    summary.get("campaign_id") == protocol.CAMPAIGN_ID
                    and summary.get("cell_key") == key
                    and summary.get("kc") == attempt.get("kc")
                    and summary.get("gate_spec_sha256") == protocol.file_sha256(protocol.GATE_SPEC)
                    and summary.get("coverage_complete") is True
                    and summary.get("all_required_cases_pass") is True
                )
            if cell.get("all_required_cases_pass") is not passed:
                blockers.append(f"recipe-resolution {key} pass flag is not summary-derived")
            cell_passes.append(passed)
        all_pass = len(cell_passes) == 24 and all(cell_passes)
        if attempt.get("all_cells_all_v4_cases_pass") is not all_pass:
            blockers.append(f"recipe-resolution KC={attempt.get('kc')} aggregate pass mismatch")
        all_pass_values.append(all_pass)
    if any(all_pass_values[:-1]) or not all_pass_values or not all_pass_values[-1]:
        blockers.append("selected KC is not the first/largest all-cell passing ladder value")
    if resolution.get("all_cells_all_v4_cases_pass") is not True:
        blockers.append("recipe-resolution selected KC is not globally passing")
    return resolution


def _retune_plan_blockers(manifest: dict[str, Any], blockers: list[str]) -> None:
    observed_paths: set[Path] = set()
    for row in manifest["jobs"]:
        if row["transfer_mode"] != "retuned":
            continue
        path = protocol.HERE / row["retune_plan"]
        observed_paths.add(path.resolve())
        plan = _json_or_block(path, f"{row['cell_id']} retune plan", blockers)
        if plan is None:
            continue
        for key in ("recipe_origin", "destination_dsl", "translator"):
            if plan.get(key) != row[key]:
                blockers.append(f"{row['cell_id']}: retune plan {key} mismatch")
        attempts = plan.get("candidates")
        if not isinstance(attempts, list) or len(attempts) != protocol.RETUNE_ATTEMPTS:
            blockers.append(f"{row['cell_id']}: retune plan must have 19 attempts")
            continue
        if [attempt.get("attempt") for attempt in attempts] != list(range(19)):
            blockers.append(f"{row['cell_id']}: retune attempt order mismatch")
        if any(attempt.get("failure_consumes_attempt") is not True for attempt in attempts):
            blockers.append(f"{row['cell_id']}: build failure does not consume budget")
        configs = [attempt.get("config") for attempt in attempts]
        if len({protocol.canonical_sha256(config) for config in configs}) != 19:
            blockers.append(f"{row['cell_id']}: retune configurations are not unique")
        if any(attempt.get("kc") != "from_recipe_resolution_lock" for attempt in attempts):
            blockers.append(f"{row['cell_id']}: retune plan bypasses common KC resolution")
        if plan.get("performance_split_visible") is not False or plan.get("frozen_before_screening") is not True:
            blockers.append(f"{row['cell_id']}: retune plan was not frozen blind")
    if len(observed_paths) != 24:
        blockers.append(f"retune plan census mismatch: expected=24 observed={len(observed_paths)}")


def _audit_receipt_blockers(
    job: dict[str, Any], implementation: dict[str, Any] | None,
    resolution: dict[str, Any] | None, blockers: list[str],
) -> tuple[dict[str, Any] | None, list[int]]:
    receipt_path = protocol.HERE / job["audit_receipt"]
    receipt = _json_or_block(receipt_path, f"{job['cell_id']} audit receipt", blockers)
    if receipt is None:
        return None, []
    required = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": job["cell_id"],
        "recipe_origin": job["recipe_origin"],
        "destination_dsl": job["destination_dsl"],
        "transfer_mode": job["transfer_mode"],
        "translator": job["translator"],
        "audit_manifest_sha256": protocol.file_sha256(make_manifests.AUDIT_MANIFEST),
        "gate_lock_sha256": protocol.file_sha256(protocol.GATE_LOCK),
        "implementation_registry_sha256": _file_digest(protocol.IMPLEMENTATION_REGISTRY),
        "recipe_resolution_lock_sha256": _file_digest(protocol.RESOLUTION_LOCK),
        "source_freeze_sha256": _file_digest(protocol.SOURCE_FREEZE),
        "audit_launch_receipt_sha256": _file_digest(protocol.HERE / "results/audit/launch_receipt.json"),
        "selected_kc": resolution.get("selected_kc") if resolution else None,
        "performance_measurements_present": False,
        "audit_complete": True,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            blockers.append(f"{job['cell_id']}: audit receipt {key} mismatch")
    attempts = receipt.get("attempts")
    budget = job["candidate_attempt_budget"]
    eligible: list[int] = []
    if not isinstance(attempts, list) or len(attempts) != budget or [row.get("attempt") for row in attempts] != list(range(budget)):
        blockers.append(f"{job['cell_id']}: audit attempt census/order mismatch")
        return receipt, eligible
    for attempt in attempts:
        index = attempt["attempt"]
        if attempt.get("implementation_sha256") != (implementation or {}).get("sha256"):
            blockers.append(f"{job['cell_id']} attempt {index}: implementation hash mismatch")
        build_ok = attempt.get("build_ok") is True
        if not build_ok:
            if attempt.get("failure_consumed") is not True or not isinstance(attempt.get("build_error"), str):
                blockers.append(f"{job['cell_id']} attempt {index}: malformed consumed build failure")
            continue
        required_audits = (
            "config_identity_verified", "source_identity_verified",
            "work_mapping_verified", "dynamic_tensor_core_work_verified",
            "generated_code_audit_pass", "prohibited_operations_absent",
            "tuning_gate_pass",
        )
        evidence = attempt.get("audit_evidence")
        evidence_passes = []
        expected_kinds = (
            "source_identity", "config_identity", "work_mapping",
            "dynamic_tensor_core_work", "generated_code",
        )
        if not isinstance(evidence, dict) or set(evidence) != set(expected_kinds):
            blockers.append(f"{job['cell_id']} attempt {index}: audit evidence map is incomplete")
            evidence_passes.append(False)
        else:
            for kind in expected_kinds:
                binding = evidence[kind]
                path = _safe_bound_file(
                    binding.get("path") if isinstance(binding, dict) else None,
                    binding.get("sha256") if isinstance(binding, dict) else None,
                    f"{job['cell_id']} attempt {index} {kind} audit", blockers,
                )
                passed_evidence = False
                if path is not None:
                    summary = protocol.load_json(path)
                    passed_evidence = bool(
                        summary.get("campaign_id") == protocol.CAMPAIGN_ID
                        and summary.get("cell_id") == job["cell_id"]
                        and summary.get("attempt") == index
                        and summary.get("audit_kind") == kind
                        and summary.get("pass") is True
                        and summary.get("implementation_sha256") == (implementation or {}).get("sha256")
                    )
                if isinstance(binding, dict) and binding.get("pass") is not passed_evidence:
                    blockers.append(f"{job['cell_id']} attempt {index}: {kind} pass flag is not evidence-derived")
                evidence_passes.append(passed_evidence)
        tuning_path = _safe_bound_file(
            attempt.get("tuning_v4_summary_path"), attempt.get("tuning_v4_summary_sha256"),
            f"{job['cell_id']} attempt {index} tuning v4 summary", blockers,
        )
        tuning_pass = False
        if tuning_path is not None:
            tuning = protocol.load_json(tuning_path)
            tuning_pass = bool(
                tuning.get("campaign_id") == protocol.CAMPAIGN_ID
                and tuning.get("cell_id") == job["cell_id"]
                and tuning.get("attempt") == index
                and tuning.get("split") == "tuning"
                and tuning.get("gate_spec_sha256") == protocol.file_sha256(protocol.GATE_SPEC)
                and tuning.get("coverage_complete") is True
                and tuning.get("all_required_cases_pass") is True
            )
        if attempt.get("tuning_gate_pass") is not tuning_pass:
            blockers.append(f"{job['cell_id']} attempt {index}: tuning gate flag is not summary-derived")
        passed = all(attempt.get(key) is True for key in required_audits) and all(evidence_passes) and tuning_pass
        if attempt.get("eligible_for_screen") is not passed:
            blockers.append(f"{job['cell_id']} attempt {index}: screen eligibility not audit-derived")
        if passed:
            eligible.append(index)
    if receipt.get("performance_screen_authorized") is not bool(eligible):
        blockers.append(f"{job['cell_id']}: screen authorization mismatch")
    return receipt, eligible


def _selection_receipt_blockers(
    job: dict[str, Any], audit: dict[str, Any] | None, eligible_attempts: list[int],
    blockers: list[str],
) -> dict[str, Any] | None:
    path = protocol.HERE / "results" / "selection" / "receipts" / f"{job['cell_id']}.json"
    selection = _json_or_block(path, f"{job['cell_id']} selection receipt", blockers)
    if selection is None:
        return None
    audit_path = protocol.HERE / job["audit_receipt"]
    required = {
        "campaign_id": protocol.CAMPAIGN_ID,
        "cell_id": job["cell_id"],
        "primary_manifest_sha256": protocol.file_sha256(make_manifests.PRIMARY_MANIFEST),
        "audit_receipt_sha256": _file_digest(audit_path),
        "source_freeze_sha256": _file_digest(protocol.SOURCE_FREEZE),
        "implementation_registry_sha256": _file_digest(protocol.IMPLEMENTATION_REGISTRY),
        "recipe_resolution_lock_sha256": _file_digest(protocol.RESOLUTION_LOCK),
        "screen_repetitions": protocol.SCREEN_REPS,
        "terminal_gate_lock_sha256": protocol.file_sha256(protocol.GATE_LOCK),
        "screen_launch_receipt_sha256": _file_digest(protocol.HERE / "results/screen/launch_receipt.json"),
    }
    for key, value in required.items():
        if selection.get(key) != value:
            blockers.append(f"{job['cell_id']}: selection receipt {key} mismatch")
    records = selection.get("screen_records")
    expected_keys = {(attempt, rep) for attempt in eligible_attempts for rep in range(protocol.SCREEN_REPS)}
    seen: dict[tuple[int, int], float] = {}
    process_ids: set[str] = set()
    if not isinstance(records, list):
        blockers.append(f"{job['cell_id']}: selection screen_records missing")
        records = []
    for record in records:
        key = (record.get("attempt"), record.get("rep"))
        latency = record.get("median_ms")
        if key not in expected_keys or key in seen or isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            blockers.append(f"{job['cell_id']}: malformed/duplicate screen record")
            continue
        if record.get("tuning_gate_pass") is not True:
            blockers.append(f"{job['cell_id']}: screen record bypassed tuning gate")
        process_id = record.get("process_instance_id")
        if not isinstance(process_id, str) or not process_id or process_id in process_ids:
            blockers.append(f"{job['cell_id']}: screen process instance is missing/reused")
        else:
            process_ids.add(process_id)
        if (
            record.get("physical_gpu") != protocol.TIMING_PHYSICAL_GPU
            or record.get("logical_device") != "cuda:0"
            or record.get("gpu_uuid") != protocol.TIMING_GPU_UUID
            or record.get("timing_distribution") != "rand_seed0_precast"
        ):
            blockers.append(f"{job['cell_id']}: screen timing hardware/distribution drift")
        seen[key] = float(latency)
    if set(seen) != expected_keys:
        blockers.append(f"{job['cell_id']}: screen census mismatch")
    medians = {
        attempt: statistics.median(seen[(attempt, rep)] for rep in range(protocol.SCREEN_REPS))
        for attempt in eligible_attempts
        if all((attempt, rep) in seen for rep in range(protocol.SCREEN_REPS))
    }
    expected_winner = min(medians, key=lambda attempt: (medians[attempt], attempt)) if medians else None
    if selection.get("selected_attempt") != expected_winner:
        blockers.append(f"{job['cell_id']}: selected attempt is not the frozen screen winner")
    terminal_path = _safe_bound_file(
        selection.get("terminal_v4_summary_path"), selection.get("terminal_v4_summary_sha256"),
        f"{job['cell_id']} terminal v4 summary", blockers,
    )
    terminal_pass = False
    if terminal_path is not None:
        terminal = protocol.load_json(terminal_path)
        terminal_pass = bool(
            terminal.get("campaign_id") == protocol.CAMPAIGN_ID
            and terminal.get("cell_id") == job["cell_id"]
            and terminal.get("attempt") == expected_winner
            and terminal.get("split") == "validation"
            and terminal.get("gate_spec_sha256") == protocol.file_sha256(protocol.GATE_SPEC)
            and terminal.get("selected_kc") == protocol.load_json(protocol.RESOLUTION_LOCK).get("selected_kc")
            and terminal.get("coverage_complete") is True
            and terminal.get("all_required_cases_pass") is True
        )
    if selection.get("terminal_gate_pass") is not terminal_pass or selection.get("eligible") is not (expected_winner is not None and terminal_pass):
        blockers.append(f"{job['cell_id']}: terminal eligibility is not summary-derived")
    return selection


def dependency_blockers(stage: str) -> list[str]:
    if stage not in protocol.STAGES:
        raise ValueError(stage)
    documents = validate_static()
    kind = "audit" if stage == "audit" else "primary"
    manifest = documents["manifests"][kind]
    blockers: list[str] = []

    freeze = _source_freeze_blockers(blockers)

    gate_lock = _json_or_block(protocol.GATE_LOCK, "v4 gate lock", blockers)
    if gate_lock is not None:
        expected = {
            "gate_spec_file_sha256": protocol.file_sha256(protocol.GATE_SPEC),
            "validation_summary_sha256": protocol.file_sha256(protocol.GATE_SUMMARY),
            "acceptance_receipt_sha256": protocol.file_sha256(protocol.GATE_RECEIPT),
            "state": "frozen",
        }
        for key, value in expected.items():
            if gate_lock.get(key) != value:
                blockers.append(f"v4 gate lock {key} mismatch")

    isolation = _translator_isolation_blockers(blockers)
    registry, implementations = _registry_blockers(manifest, blockers)
    resolution = _resolution_blockers(registry, implementations, blockers)
    _retune_plan_blockers(manifest, blockers)

    if stage in ("screen", "primary"):
        _launch_receipt_blockers("audit", blockers)
        if stage == "primary":
            _launch_receipt_blockers("screen", blockers)
        for row in manifest["jobs"]:
            audit, eligible = _audit_receipt_blockers(
                row, implementations.get(row["cell_id"]), resolution, blockers
            )
            if stage == "primary":
                selection = _selection_receipt_blockers(row, audit, eligible, blockers)
                if selection is not None and selection.get("eligible") is not True:
                    blockers.append(f"{row['cell_id']}: selection did not establish terminal eligibility")

    provenance = _json_or_block(
        protocol.PROVENANCE_LOCK, "prelaunch provenance lock", blockers
    )
    if provenance is not None:
        if provenance.get("campaign_id") != protocol.CAMPAIGN_ID:
            blockers.append("prelaunch provenance campaign mismatch")
        if freeze is None or provenance.get("source_freeze_sha256") != _file_digest(protocol.SOURCE_FREEZE):
            blockers.append("prelaunch provenance does not bind source freeze")
        if not SHA_RE.fullmatch(str(provenance.get("git_commit", ""))):
            blockers.append("prelaunch provenance lacks full immutable git commit")
        if provenance.get("remote_push_verified") is not True:
            blockers.append("prelaunch provenance has no verified remote push")
        if not provenance.get("external_timestamp_utc"):
            blockers.append("prelaunch provenance has no external timestamp")
        if isolation is None or provenance.get("translator_isolation_sha256") != _file_digest(protocol.TRANSLATOR_ISOLATION_LOCK):
            blockers.append("prelaunch provenance does not bind translator isolation")
    return blockers


def gpu_blocker() -> str | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "NVIDIA driver is unavailable: " + (
            completed.stderr.strip() or "nvidia-smi failed"
        )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 4:
        return f"expected four visible GPUs, observed {len(rows)}"
    parsed = [[item.strip() for item in row.split(",", 2)] for row in rows]
    expected = [
        str(protocol.TIMING_PHYSICAL_GPU), protocol.TIMING_GPU_UUID,
        protocol.TIMING_GPU_NAME,
    ]
    if expected not in parsed:
        return f"timing GPU identity mismatch: required {expected}"
    occupants = subprocess.run(
        [
            "nvidia-smi", f"--id={protocol.TIMING_PHYSICAL_GPU}",
            "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True,
    )
    if occupants.returncode != 0:
        return "timing GPU occupancy query failed"
    if occupants.stdout.strip():
        return f"timing GPU {protocol.TIMING_PHYSICAL_GPU} is occupied"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=protocol.STAGES, default="audit")
    parser.add_argument("--manifest", choices=protocol.KINDS, help=argparse.SUPPRESS)
    parser.add_argument("--launch-ready", action="store_true")
    args = parser.parse_args()
    validate_static()
    print("static validation: PASS (3 x 4 x 2 x 2 = 48 cells per manifest)")
    if not args.launch_ready:
        return 0
    stage = args.manifest or args.stage
    blockers = dependency_blockers(stage)
    hardware = gpu_blocker()
    if hardware:
        blockers.append(hardware)
    for blocker in blockers:
        print("BLOCKED:", blocker)
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
