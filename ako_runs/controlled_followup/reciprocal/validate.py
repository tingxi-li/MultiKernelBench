#!/usr/bin/env python3
"""CPU-only structural and dependency validation for reciprocal transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import make_manifests


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GATE_SPEC = (HERE / make_manifests.GATE_SPEC_RELATIVE).resolve()
VALIDATION_SUMMARY = (
    HERE / make_manifests.GATE_VALIDATION_SUMMARY_RELATIVE
).resolve()
ACCEPTANCE_RECEIPT = (
    HERE / make_manifests.GATE_ACCEPTANCE_RECEIPT_RELATIVE
).resolve()
GATE_LOCK = HERE / "dependencies/gate_lock.json"
RECIPE_LOCK = HERE / "dependencies/recipe_resolution_lock.json"
MANIFESTS = {
    "primary": HERE / "manifests/primary.json",
    "audit": HERE / "manifests/audit.json",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_GATE_CALIBRATION_SEEDS = 640
REQUIRED_GATE_VALIDATION_SEEDS = 512
EXPECTED_GATE_CANDIDATES = {
    "conformance_mixed": "native_mixed_holdout_v4",
    "semantic_mixed": "native_mixed_holdout_v4",
    "semantic_q32": "native_fp32_holdout_v4",
}


class ValidationError(ValueError):
    pass


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _is_hash(value: Any, *, nonzero: bool = False) -> bool:
    return bool(
        isinstance(value, str)
        and HEX64.fullmatch(value)
        and (not nonzero or value != "0" * 64)
    )


def gate_acceptance_blockers(
    spec: dict[str, Any], summary: dict[str, Any]
) -> list[str]:
    """Check that a validator summary accepts every frozen matmul gate."""
    blockers: list[str] = []
    if summary.get("schema_version") != "1.0":
        blockers.append("validation summary must use schema_version='1.0'")
    if summary.get("campaign_id") != spec.get("campaign_id"):
        blockers.append("validation summary campaign_id does not match gate spec")
    if summary.get("manifest_sha256") != spec.get("manifest_sha256"):
        blockers.append("validation summary manifest hash does not match gate spec")
    if summary.get("gate_spec_sha256") != canonical_sha256(spec):
        blockers.append("validation summary does not bind the canonical gate hash")
    if summary.get("success") is not True:
        blockers.append("validation summary does not report success=true")
    if summary.get("success_rule") != "all_metrics_all_cases_all_seeds_and_complete_coverage":
        blockers.append("validation summary success rule is not fail-closed")
    if summary.get("failures") != []:
        blockers.append("validation summary contains failures")

    gates = spec.get("gates")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(gates, dict):
        for gate in gates.values():
            if isinstance(gate, dict) and gate.get("op") == "matmul":
                expected[(gate.get("op"), gate.get("gate_id"))] = gate
    groups = summary.get("groups")
    observed: dict[tuple[Any, Any], dict[str, Any]] = {}
    if not isinstance(groups, list):
        blockers.append("validation summary groups must be a list")
        return blockers
    for group in groups:
        if not isinstance(group, dict):
            blockers.append("validation summary contains a non-object group")
            continue
        key = (group.get("op"), group.get("gate_id"))
        if key in observed:
            blockers.append(f"validation summary duplicates group {key}")
        observed[key] = group
    if set(observed) != set(expected):
        blockers.append("validation summary does not cover exactly every frozen matmul gate")
        return blockers
    for key, gate in expected.items():
        group = observed[key]
        expected_records = (
            len(gate.get("required_cases", []))
            * gate.get("required_validation_seeds_per_case", 0)
        )
        if (
            group.get("success") is not True
            or group.get("coverage_complete") is not True
            or group.get("n_failed_records") != 0
            or group.get("missing_records") != 0
            or group.get("n_records") != expected_records
            or group.get("observed_failure_rate") != 0.0
            or group.get("candidate") != EXPECTED_GATE_CANDIDATES.get(key[1])
        ):
            blockers.append(
                f"validation summary group {key} is not a complete zero-failure holdout"
            )
    return blockers


def acceptance_receipt_blockers(
    spec: dict[str, Any],
    summary: dict[str, Any],
    receipt: dict[str, Any],
) -> list[str]:
    """Verify the receipt and every raw holdout file it content-addresses."""
    blockers: list[str] = []
    if receipt.get("schema_version") != "1.0":
        blockers.append("gate acceptance receipt must use schema_version='1.0'")
    if receipt.get("campaign_id") != spec.get("campaign_id"):
        blockers.append("gate acceptance receipt campaign_id does not match gate spec")
    if receipt.get("acceptance_rule") != summary.get("success_rule"):
        blockers.append("gate acceptance receipt uses a different acceptance rule")
    if receipt.get("accepted_for_reciprocal_recipe_transfer") is not True:
        blockers.append("gate acceptance receipt does not authorize reciprocal transfer")
    if receipt.get("manifest_canonical_sha256") != spec.get("manifest_sha256"):
        blockers.append("gate acceptance receipt manifest hash does not match gate spec")

    frozen = receipt.get("frozen_gate")
    if not isinstance(frozen, dict):
        blockers.append("gate acceptance receipt lacks frozen_gate")
    else:
        if frozen.get("path") != _display_path(GATE_SPEC):
            blockers.append("gate acceptance receipt points to a different gate spec")
        if frozen.get("canonical_sha256") != canonical_sha256(spec):
            blockers.append("gate acceptance receipt has the wrong canonical gate hash")
        if frozen.get("file_sha256") != sha256_file(GATE_SPEC):
            blockers.append("gate acceptance receipt has the wrong gate file hash")

    validation = receipt.get("validation")
    if not isinstance(validation, dict):
        blockers.append("gate acceptance receipt lacks validation evidence")
        return blockers
    expected_records = sum(
        len(gate.get("required_cases", []))
        * gate.get("required_validation_seeds_per_case", 0)
        for gate in spec.get("gates", {}).values()
        if isinstance(gate, dict) and gate.get("op") == "matmul"
    )
    if (
        validation.get("records") != expected_records
        or validation.get("failed_records") != 0
        or validation.get("collection_failures") != 0
        or validation.get("coverage_complete") is not True
    ):
        blockers.append("gate acceptance receipt validation counters are incomplete")
    summary_binding = validation.get("summary")
    if not isinstance(summary_binding, dict):
        blockers.append("gate acceptance receipt lacks validation-summary binding")
    else:
        if summary_binding.get("path") != _display_path(VALIDATION_SUMMARY):
            blockers.append("gate acceptance receipt points to a different summary")
        if summary_binding.get("sha256") != sha256_file(VALIDATION_SUMMARY):
            blockers.append("gate acceptance receipt has the wrong summary file hash")
        if summary_binding.get("success") is not True:
            blockers.append("gate acceptance receipt summary binding is not successful")

    gates = {
        gate.get("gate_id"): gate
        for gate in spec.get("gates", {}).values()
        if isinstance(gate, dict) and gate.get("op") == "matmul"
    }
    raw = validation.get("raw_evidence")
    if not isinstance(raw, list):
        blockers.append("gate acceptance receipt raw_evidence must be a list")
        return blockers
    observed: dict[Any, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            blockers.append("gate acceptance receipt contains non-object raw evidence")
            continue
        gate_id = item.get("gate_id")
        if gate_id in observed:
            blockers.append(f"gate acceptance receipt duplicates raw gate {gate_id}")
        observed[gate_id] = item
    if set(observed) != set(gates):
        blockers.append("gate acceptance receipt does not cover every matmul gate")
        return blockers
    for gate_id, gate in gates.items():
        item = observed[gate_id]
        path_value = item.get("path")
        path = REPO_ROOT / path_value if isinstance(path_value, str) else None
        expected = (
            len(gate.get("required_cases", []))
            * gate.get("required_validation_seeds_per_case", 0)
        )
        if item.get("records") != expected:
            blockers.append(f"gate acceptance receipt has wrong count for {gate_id}")
        if path is None or not path.is_file():
            blockers.append(f"gate acceptance receipt raw file is missing for {gate_id}")
            continue
        try:
            path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            blockers.append(f"gate acceptance receipt raw path escapes repository for {gate_id}")
            continue
        if item.get("sha256") != sha256_file(path):
            blockers.append(f"gate acceptance receipt raw hash mismatch for {gate_id}")
        with path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        if line_count != expected:
            blockers.append(f"gate acceptance receipt raw line count mismatch for {gate_id}")
    return blockers


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def validate_recipe_card(card: dict[str, Any], path: Path) -> None:
    where = path.relative_to(REPO_ROOT).as_posix()
    _require(card.get("schema_version") == 1, f"{where}: bad schema_version")
    _require(
        card.get("card_kind") == "frozen_source_recipe",
        f"{where}: card is not frozen_source_recipe",
    )
    _require(
        card.get("launch_state")
        == "blocked_until_robust_gate_and_bindings_are_frozen",
        f"{where}: launch_state weakened",
    )
    _require(
        card.get("recipe_id") in make_manifests.ORIGINS,
        f"{where}: unexpected recipe_id",
    )
    workload = card.get("workload", {})
    _require(workload == make_manifests.WORKLOAD, f"{where}: workload drift")
    algorithm = card.get("algorithm", {})
    expected_algorithm = {
        "operand_dtype": "fp16",
        "tensor_core": True,
        "tensor_accumulator_dtype": "fp32",
        "outer_accumulator_dtype": "fp32",
        "output_dtype": "fp32",
        "cross_block_split_k": False,
        "vendor_gemm": False,
    }
    for key, value in expected_algorithm.items():
        _require(algorithm.get(key) == value, f"{where}: algorithm.{key} drift")

    literal = card.get("literal_treatment", {})
    configs = literal.get("configs")
    _require(isinstance(configs, list) and configs, f"{where}: no literal configs")
    for index, config in enumerate(configs):
        _require(isinstance(config, dict), f"{where}: config {index} is not an object")
        for axis in ("BM", "BN", "BK", "stages"):
            value = config.get(axis)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f"{where}: config {index} has invalid {axis}",
            )

    retuned = card.get("retuned_treatment", {})
    budget = retuned.get("candidate_budget", {})
    _require(
        budget.get("attempted_candidates") == 19,
        f"{where}: recipient retune budget must remain 19",
    )
    _require(
        budget.get("build_failures_consume_budget") is True,
        f"{where}: build failures must consume the recipient budget",
    )
    _require(
        retuned.get("selection", {}).get("performance_split_hidden_until_winner_freeze")
        is True,
        f"{where}: performance split is not hidden during tuning",
    )

    dependency = card.get("gate_dependency", {})
    _require(dependency.get("operation") == "matmul", f"{where}: wrong gate op")
    _require(
        dependency.get("gate_spec") == make_manifests.GATE_SPEC_RELATIVE,
        f"{where}: gate-spec path drift",
    )
    _require(
        dependency.get("validation_summary")
        == make_manifests.GATE_VALIDATION_SUMMARY_RELATIVE,
        f"{where}: validation-summary path drift",
    )
    _require(
        dependency.get("acceptance_receipt")
        == make_manifests.GATE_ACCEPTANCE_RECEIPT_RELATIVE,
        f"{where}: acceptance-receipt path drift",
    )
    _require(
        dependency.get("required_state") == "frozen",
        f"{where}: robust gate is not required frozen",
    )

    evidence = card.get("evidence")
    _require(isinstance(evidence, list) and len(evidence) >= 2, f"{where}: weak evidence")
    for item in evidence:
        source = REPO_ROOT / item.get("path", "")
        _require(source.is_file(), f"{where}: evidence source missing: {source}")
        _require(_is_hash(item.get("sha256")), f"{where}: bad evidence hash")
        _require(
            sha256_file(source) == item["sha256"],
            f"{where}: evidence source changed: {item['path']}",
        )

    recipe_id = card["recipe_id"]
    if recipe_id == "tilelang_phase1_confirmed":
        _require(len(configs) == 1, f"{where}: TileLang literal must be one point")
        _require(
            configs[0] == make_manifests.tilelang_confirmed_config(),
            f"{where}: TileLang literal no longer matches the confirmed winner",
        )
        _require(
            algorithm.get("grid_mapping") == "plain_2d_n_then_m",
            f"{where}: TileLang mapping drift",
        )
    elif recipe_id == "triton_grouped_autotuned":
        _require(len(configs) == 13, f"{where}: Triton literal must have 13 points")
        _require(
            configs == make_manifests.triton_native_configs(),
            f"{where}: Triton literal family no longer matches its donor source",
        )
        _require(
            algorithm.get("grid_mapping") == "one_dimensional_grouped_m"
            and algorithm.get("group_m") == 8,
            f"{where}: Triton grouped mapping drift",
        )
        amendment = card.get("correctness_amendment", {})
        _require(
            amendment.get("kind") == "in_block_accumulator_flush_only"
            and amendment.get("no_other_donor_change_permitted") is True,
            f"{where}: Triton correctness amendment widened",
        )


def validate_manifest(manifest: dict[str, Any], kind: str, path: Path) -> None:
    where = path.relative_to(REPO_ROOT).as_posix()
    _require(manifest.get("schema_version") == 1, f"{where}: bad schema_version")
    _require(
        manifest.get("campaign_id") == make_manifests.CAMPAIGN_ID,
        f"{where}: campaign_id drift",
    )
    _require(manifest.get("manifest_kind") == kind, f"{where}: kind mismatch")
    _require(
        manifest.get("launch_state") == "blocked_pending_frozen_dependencies",
        f"{where}: static manifest must remain dependency-blocked",
    )
    _require(manifest.get("job_count") == 16, f"{where}: expected 16 jobs")
    jobs = manifest.get("jobs")
    _require(isinstance(jobs, list), f"{where}: jobs is not a list")
    _require(
        make_manifests.sha256_bytes(make_manifests.stable_json_bytes(jobs))
        == manifest.get("jobs_sha256"),
        f"{where}: jobs hash mismatch",
    )
    expected_cells = [
        (origin, destination, mode)
        for origin in make_manifests.ORIGINS
        for destination in make_manifests.DESTINATIONS
        for mode in make_manifests.TRANSFER_MODES
    ]
    got_cells = [
        (job.get("recipe_origin"), job.get("destination_dsl"), job.get("transfer_mode"))
        for job in jobs
    ]
    _require(got_cells == expected_cells, f"{where}: factorial order/content drift")
    _require(
        len({job.get("job_id") for job in jobs}) == 16,
        f"{where}: duplicate job_id",
    )
    for ordinal, job in enumerate(jobs):
        _require(job.get("ordinal") == ordinal, f"{where}: ordinal drift at {ordinal}")
        card_path = HERE / job["recipe_card"]
        _require(card_path.is_file(), f"{where}: missing recipe card {card_path}")
        _require(
            sha256_file(card_path) == job.get("recipe_card_sha256"),
            f"{where}: job recipe-card hash mismatch",
        )
        required = job.get("required_files")
        _require(isinstance(required, list) and required, f"{where}: no dependencies")
        _require(
            "dependencies/gate_lock.json" in required,
            f"{where}: job does not require gate lock",
        )
        if job["transfer_mode"] == "retuned":
            _require(
                any(item.startswith("retune_plans/") for item in required),
                f"{where}: retuned job lacks frozen plan",
            )
        if kind == "primary":
            _require(
                any(item.startswith("audit_receipts/") for item in required),
                f"{where}: primary job lacks audit receipt",
            )

    dependency = manifest.get("robust_gate_dependency", {})
    _require(dependency.get("required_state") == "frozen", f"{where}: weak gate state")
    _require(
        dependency.get("required_hashes")
        == [
            "manifest_sha256",
            "calibration_records_sha256",
            "gate_spec_sha256",
            "gate_spec_file_sha256",
            "validation_summary_sha256",
            "acceptance_receipt_sha256",
        ],
        f"{where}: gate hashes drift",
    )
    _require(
        dependency.get("gate_spec") == make_manifests.GATE_SPEC_RELATIVE,
        f"{where}: gate-spec path drift",
    )
    _require(
        dependency.get("validation_summary")
        == make_manifests.GATE_VALIDATION_SUMMARY_RELATIVE,
        f"{where}: validation-summary path drift",
    )
    _require(
        dependency.get("acceptance_receipt")
        == make_manifests.GATE_ACCEPTANCE_RECEIPT_RELATIVE,
        f"{where}: acceptance-receipt path drift",
    )


def validate_static() -> dict[str, Any]:
    # Parse the schema even when the optional jsonschema package is absent.
    schema = load_json(make_manifests.RECIPE_SCHEMA)
    _require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "recipe-card schema is not draft 2020-12",
    )
    make_manifests.check_documents(verbose=False)
    cards: dict[str, dict[str, Any]] = {}
    for path in (make_manifests.TILELANG_CARD, make_manifests.TRITON_CARD):
        card = load_json(path)
        validate_recipe_card(card, path)
        cards[card["recipe_id"]] = card
    _require(tuple(cards) == make_manifests.ORIGINS, "recipe-card order drift")
    manifests = {}
    for kind, path in MANIFESTS.items():
        document = load_json(path)
        validate_manifest(document, kind, path)
        manifests[kind] = document
    return {"cards": cards, "manifests": manifests}


def gate_dependency_blockers(
    gate_spec_path: Path = GATE_SPEC,
    gate_lock_path: Path = GATE_LOCK,
    validation_summary_path: Path = VALIDATION_SUMMARY,
    acceptance_receipt_path: Path = ACCEPTANCE_RECEIPT,
) -> list[str]:
    blockers: list[str] = []
    if not gate_spec_path.is_file():
        blockers.append(
            f"robust gate is not frozen: missing {_display_path(gate_spec_path)}"
        )
    if not gate_lock_path.is_file():
        blockers.append(
            f"robust gate is not bound: missing {_display_path(gate_lock_path)}"
        )
    if not validation_summary_path.is_file():
        blockers.append(
            "robust gate is not validated: missing "
            f"{_display_path(validation_summary_path)}"
        )
    if not acceptance_receipt_path.is_file():
        blockers.append(
            "robust gate acceptance is not receipted: missing "
            f"{_display_path(acceptance_receipt_path)}"
        )
    if blockers:
        return blockers
    try:
        spec = load_json(gate_spec_path)
        lock = load_json(gate_lock_path)
        summary = load_json(validation_summary_path)
        receipt = load_json(acceptance_receipt_path)
    except ValidationError as exc:
        return [str(exc)]

    if lock.get("schema_version") != 1 or lock.get("state") != "frozen":
        blockers.append("gate lock must have schema_version=1 and state='frozen'")
    if lock.get("operation") != "matmul":
        blockers.append("gate lock must bind operation='matmul'")
    if lock.get("gate_spec") != make_manifests.GATE_SPEC_RELATIVE:
        blockers.append("gate lock does not bind the accepted v4 gate-spec path")
    if (
        lock.get("validation_summary")
        != make_manifests.GATE_VALIDATION_SUMMARY_RELATIVE
    ):
        blockers.append("gate lock does not bind the accepted v4 validation path")
    if (
        lock.get("acceptance_receipt")
        != make_manifests.GATE_ACCEPTANCE_RECEIPT_RELATIVE
    ):
        blockers.append("gate lock does not bind the accepted v4 receipt path")
    required_hashes = (
        "manifest_sha256",
        "calibration_records_sha256",
        "gate_spec_sha256",
        "gate_spec_file_sha256",
        "validation_summary_sha256",
        "acceptance_receipt_sha256",
    )
    for field in required_hashes:
        if not _is_hash(lock.get(field), nonzero=True):
            blockers.append(f"gate lock {field} is not a non-placeholder SHA256")
    if lock.get("gate_spec_sha256") != canonical_sha256(spec):
        blockers.append("gate lock does not bind the validator's canonical gate hash")
    if lock.get("gate_spec_file_sha256") != sha256_file(gate_spec_path):
        blockers.append("gate lock does not hash the exact gate-spec bytes")
    if lock.get("validation_summary_sha256") != sha256_file(validation_summary_path):
        blockers.append("gate lock does not hash the exact validation summary bytes")
    if lock.get("acceptance_receipt_sha256") != sha256_file(acceptance_receipt_path):
        blockers.append("gate lock does not hash the exact acceptance receipt bytes")
    if spec.get("schema_version") != "1.0":
        blockers.append("gate spec must use robust-gate schema_version='1.0'")
    if lock.get("campaign_id") != spec.get("campaign_id"):
        blockers.append("gate lock campaign_id does not match gate spec")
    for field in ("manifest_sha256", "calibration_records_sha256"):
        if lock.get(field) != spec.get(field):
            blockers.append(f"gate lock {field} does not match gate spec")
    gates = spec.get("gates")
    matmul_gates = [
        gate
        for gate in gates.values()
        if isinstance(gate, dict) and gate.get("op") == "matmul"
    ] if isinstance(gates, dict) else []
    if not matmul_gates:
        blockers.append("gate spec has no frozen matmul gate")
    for gate in matmul_gates:
        expected_calibration = (
            len(gate.get("anchors", []))
            * len(gate.get("required_cases", []))
            * REQUIRED_GATE_CALIBRATION_SEEDS
        )
        if expected_calibration <= 0 or gate.get("calibration_records") != expected_calibration:
            blockers.append(
                f"matmul/{gate.get('gate_id', '?')} lacks full "
                f"{REQUIRED_GATE_CALIBRATION_SEEDS}-seed anchor calibration"
            )
        if (
            gate.get("required_validation_seeds_per_case")
            != REQUIRED_GATE_VALIDATION_SEEDS
        ):
            blockers.append(
                f"matmul/{gate.get('gate_id', '?')} does not require "
                f"{REQUIRED_GATE_VALIDATION_SEEDS} validation seeds"
            )
    blockers.extend(gate_acceptance_blockers(spec, summary))
    blockers.extend(acceptance_receipt_blockers(spec, summary, receipt))
    return blockers


def recipe_resolution_blockers(
    gate_lock_path: Path = GATE_LOCK, recipe_lock_path: Path = RECIPE_LOCK
) -> list[str]:
    if not recipe_lock_path.is_file():
        return [
            f"recipe correctness amendment unresolved: missing "
            f"{_display_path(recipe_lock_path)}"
        ]
    if not gate_lock_path.is_file():
        return ["recipe resolution cannot be checked before gate lock exists"]
    try:
        gate_lock = load_json(gate_lock_path)
        lock = load_json(recipe_lock_path)
    except ValidationError as exc:
        return [str(exc)]
    blockers: list[str] = []
    if lock.get("schema_version") != 1 or lock.get("state") != "frozen":
        blockers.append("recipe resolution lock must be frozen schema version 1")
    if lock.get("gate_spec_sha256") != gate_lock.get("gate_spec_sha256"):
        blockers.append("recipe resolution lock binds a different robust gate")
    recipes = lock.get("recipes")
    if not isinstance(recipes, dict) or set(recipes) != set(make_manifests.ORIGINS):
        blockers.append("recipe resolution lock must resolve exactly both origins")
        return blockers
    tile_kc = recipes[make_manifests.ORIGINS[0]].get("kc")
    triton_kc = recipes[make_manifests.ORIGINS[1]].get("kc")
    if tile_kc != 2048:
        blockers.append("literal TileLang donor KC must remain 2048")
    allowed = [8192, 4096, 2048, 1024, 512]
    if triton_kc not in allowed:
        blockers.append(f"Triton KC must be one of {allowed}")
    card_hashes = lock.get("recipe_card_sha256")
    expected = {
        make_manifests.ORIGINS[0]: sha256_file(make_manifests.TILELANG_CARD),
        make_manifests.ORIGINS[1]: sha256_file(make_manifests.TRITON_CARD),
    }
    if card_hashes != expected:
        blockers.append("recipe resolution lock does not bind both exact recipe cards")
    return blockers


def file_dependency_blockers(manifest: dict[str, Any]) -> list[str]:
    missing: set[str] = set()
    for job in manifest["jobs"]:
        for relative in job["required_files"]:
            if relative in {
                "dependencies/gate_lock.json",
                "dependencies/recipe_resolution_lock.json",
            }:
                continue
            if not (HERE / relative).is_file():
                missing.add(relative)
    return [f"missing campaign binding: {relative}" for relative in sorted(missing)]


def launch_blockers(manifest: dict[str, Any]) -> list[str]:
    return (
        gate_dependency_blockers()
        + recipe_resolution_blockers()
        + file_dependency_blockers(manifest)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", choices=tuple(MANIFESTS), default="primary"
    )
    parser.add_argument(
        "--launch-ready",
        action="store_true",
        help="fail unless every frozen dependency and implementation binding exists",
    )
    args = parser.parse_args()
    documents = validate_static()
    selected = documents["manifests"][args.manifest]
    blockers = launch_blockers(selected)
    print(
        f"OK static reciprocal scaffold: {selected['job_count']} {args.manifest} jobs; "
        f"{len(blockers)} launch blocker(s)"
    )
    for blocker in blockers:
        print(f"BLOCKED: {blocker}")
    if args.launch_ready and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
