#!/usr/bin/env python3
"""CPU-only, fail-closed manifest protocol for trajectory-transfer T0/T1."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


DESTINATIONS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
MODES = ("literal", "retuned")
TERMINAL_STATUS = {
    "TRANSLATION_FAILED": "translation_failure",
    "AUDIT_FAILED": "source_or_generated_code_audit_failure",
    "UNSUPPORTED": "source_unsupported",
    "BUILD_FAILED": "build_or_resource_failure",
    "LAUNCH_FAILED": "launch_or_resource_failure",
    "GATE_FAILED": "correctness_failure",
    "GATE_PASSED": "gate_legal",
}
ID = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


class ProtocolError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _name(value: Any, field: str) -> str:
    require(isinstance(value, str) and ID.fullmatch(value) is not None, f"invalid {field}")
    return value


def _sha256(value: Any, field: str) -> str:
    require(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"invalid {field}",
    )
    return value


def validate_registry(value: Any) -> dict[str, Any]:
    """Validate a donor-prefix registry without resolving or executing sources."""
    require(isinstance(value, dict), "registry must be an object")
    require(value.get("schema_version") == 1, "schema_version must be 1")
    require(value.get("record_type") == "trajectory_transfer_donor_prefix_registry", "wrong record_type")
    _name(value.get("campaign_id"), "campaign_id")
    _name(value.get("operator"), "operator")
    require(value.get("policy_status") == "deferred", "transfer policy must remain deferred")
    require(value.get("destinations") == list(DESTINATIONS), "destination census/order drift")
    require(value.get("transfer_modes") == list(MODES), "transfer-mode census/order drift")
    require(isinstance(value.get("order_seed"), str) and value["order_seed"], "order_seed is required")
    _sha256(value.get("gate_lock_sha256"), "gate_lock_sha256")
    _sha256(value.get("input_contract_sha256"), "input_contract_sha256")
    require(
        type(value.get("retuned_attempts_per_cell")) is int
        and value["retuned_attempts_per_cell"] > 0,
        "retuned_attempts_per_cell must be a positive integer",
    )

    translators = value.get("translators")
    require(isinstance(translators, list) and translators, "at least one translator is required")
    for translator in translators:
        _name(translator, "translator")
    require(len(translators) == len(set(translators)), "duplicate translator")

    trajectories = value.get("trajectories")
    require(isinstance(trajectories, list) and len(trajectories) >= 3, "at least three trajectories are required")
    seen_origins: set[str] = set()
    seen_prefixes: set[str] = set()
    for trajectory in trajectories:
        require(isinstance(trajectory, dict), "trajectory must be an object")
        origin = _name(trajectory.get("origin"), "origin")
        require(origin not in seen_origins, f"duplicate origin: {origin}")
        seen_origins.add(origin)
        require(trajectory.get("origin_dsl") in DESTINATIONS, f"{origin}: invalid origin_dsl")
        prefixes = trajectory.get("prefixes")
        require(isinstance(prefixes, list) and prefixes, f"{origin}: prefixes missing")
        require(all(isinstance(row, dict) for row in prefixes), f"{origin}: prefix must be an object")
        require(
            [row.get("prefix_index") for row in prefixes] == list(range(len(prefixes))),
            f"{origin}: prefixes must be contiguous from zero",
        )
        mechanisms: set[str] = set()
        for index, prefix in enumerate(prefixes):
            require(isinstance(prefix, dict), f"{origin} p{index}: prefix must be an object")
            prefix_id = _name(prefix.get("prefix_id"), "prefix_id")
            require(prefix_id not in seen_prefixes, f"duplicate prefix_id: {prefix_id}")
            seen_prefixes.add(prefix_id)
            _sha256(prefix.get("source_sha256"), f"{prefix_id}.source_sha256")
            _sha256(prefix.get("gate_receipt_sha256"), f"{prefix_id}.gate_receipt_sha256")
            _sha256(prefix.get("mechanism_audit_sha256"), f"{prefix_id}.mechanism_audit_sha256")
            introduced = prefix.get("introduced_mechanisms")
            dependencies = prefix.get("depends_on_steps")
            require(isinstance(introduced, list), f"{prefix_id}: introduced_mechanisms must be a list")
            require(isinstance(dependencies, list), f"{prefix_id}: depends_on_steps must be a list")
            if index == 0:
                require(not introduced and not dependencies, f"{prefix_id}: baseline cannot introduce or depend on a step")
                require(prefix.get("step_artifact_sha256") is None, f"{prefix_id}: baseline has no step artifact")
            else:
                require(len(introduced) == 1, f"{prefix_id}: each step must introduce exactly one mechanism")
                mechanism = _name(introduced[0], "mechanism")
                require(mechanism not in mechanisms, f"{prefix_id}: duplicate mechanism")
                mechanisms.add(mechanism)
                _sha256(prefix.get("step_artifact_sha256"), f"{prefix_id}.step_artifact_sha256")
                require(prefix.get("composable") is True, f"{prefix_id}: step artifact must be composable")
                require(
                    all(type(step) is int and 1 <= step < index for step in dependencies)
                    and len(dependencies) == len(set(dependencies)),
                    f"{prefix_id}: dependencies must name unique prior steps",
                )
            status = prefix.get("terminal_status")
            require(isinstance(status, str) and status in TERMINAL_STATUS, f"{prefix_id}: unknown terminal_status")
            require(status == "GATE_PASSED", f"{prefix_id}: T1 donor prefixes must be gate-legal")
    require(len({row["origin_dsl"] for row in trajectories}) >= 3, "at least three origin DSLs are required")
    plans = value.get("retune_plan_sha256_by_origin")
    require(isinstance(plans, dict) and set(plans) == seen_origins, "retune plan origin census drift")
    attempt_plans = value.get("retune_attempt_plans_by_origin")
    require(isinstance(attempt_plans, dict) and set(attempt_plans) == seen_origins, "retune attempt plan origin census drift")
    for origin, digest in plans.items():
        _sha256(digest, f"retune plan for {origin}")
        attempts = attempt_plans[origin]
        require(
            isinstance(attempts, list)
            and len(attempts) == value["retuned_attempts_per_cell"],
            f"{origin}: retune attempt plan has the wrong census",
        )
        require(
            [row.get("attempt_index") for row in attempts if isinstance(row, dict)]
            == list(range(1, len(attempts) + 1)),
            f"{origin}: retune attempts must be contiguous from one",
        )
        config_hashes = [_sha256(row.get("config_sha256"), f"{origin}.config_sha256") for row in attempts]
        require(len(config_hashes) == len(set(config_hashes)), f"{origin}: duplicate retune config")
        require(_digest(attempts) == digest, f"{origin}: retune plan hash mismatch")
    return value


def target_outcome(
    manifest: Any,
    manifest_row: Any,
    terminal_status: Any,
    evidence_sha256: Any,
    *,
    support_probe_receipt_sha256: Any = None,
    target_source_sha256: Any = None,
) -> dict[str, Any]:
    """Preserve a target translation's terminal state without inventing timing."""
    require(isinstance(manifest, dict), "manifest must be an object")
    manifest_sha256 = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    require(manifest_sha256 == _digest(unsigned), "manifest hash mismatch")
    require(isinstance(manifest_row, dict), "manifest_row must be an object")
    cell_id = _name(manifest_row.get("cell_id"), "cell_id")
    members = manifest.get("rows", []) + manifest.get("order_control_rows", [])
    require(sum(row == manifest_row for row in members) == 1, "target cell is not an exact manifest member")
    require(terminal_status in TERMINAL_STATUS, "unknown target terminal_status")
    _sha256(evidence_sha256, "target evidence_sha256")
    if terminal_status == "UNSUPPORTED":
        _sha256(support_probe_receipt_sha256, "support_probe_receipt_sha256")
    else:
        require(support_probe_receipt_sha256 is None, "support probe receipt is only valid for UNSUPPORTED")
    if terminal_status == "GATE_PASSED":
        _sha256(target_source_sha256, "target_source_sha256")
    else:
        require(target_source_sha256 is None, "target source is only admitted after GATE_PASSED")
    positive_control_passed: bool | None = None
    if manifest_row.get("pipeline_positive_control"):
        positive_control_passed = False
    if manifest_row.get("pipeline_positive_control") and terminal_status == "GATE_PASSED":
        require(
            target_source_sha256 == manifest_row.get("donor_source_sha256"),
            "literal self-transfer must be byte-identical and gate-legal",
        )
        positive_control_passed = True
    return {
        "cell_id": cell_id,
        "manifest_sha256": manifest_sha256,
        "terminal_status": terminal_status,
        "outcome_class": TERMINAL_STATUS[terminal_status],
        "timing_eligible": terminal_status == "GATE_PASSED",
        "evidence_sha256": evidence_sha256,
        "support_probe_receipt_sha256": support_probe_receipt_sha256,
        "target_source_sha256": target_source_sha256,
        "positive_control_passed": positive_control_passed,
    }


def _topological_control(trajectory: dict[str, Any], seed: str) -> list[int] | None:
    """Select one stable dependency-valid order different from donor order."""
    prefixes = trajectory["prefixes"]
    donor = list(range(1, len(prefixes)))
    if len(donor) < 3:
        return None
    dependencies = {step: set(prefixes[step]["depends_on_steps"]) for step in donor}

    def choose(key: Any) -> list[int]:
        remaining = set(donor)
        ordered: list[int] = []
        while remaining:
            ready = [step for step in remaining if dependencies[step] <= set(ordered)]
            require(bool(ready), f"{trajectory['origin']}: dependency cycle")
            step = min(ready, key=key)
            ordered.append(step)
            remaining.remove(step)
        return ordered

    origin = trajectory["origin"]
    selected = choose(lambda step: _digest([seed, origin, step]))
    if selected == donor:
        selected = choose(lambda step: -step)
    return selected if selected != donor else None


def _cell_id(coordinate: dict[str, Any]) -> str:
    return "tt1_" + _digest(coordinate)[:24]


def derive_manifest(registry: dict[str, Any]) -> dict[str, Any]:
    """Derive the donor-order census and the separate valid-order controls."""
    validate_registry(registry)
    translators = sorted(registry["translators"])
    trajectories = sorted(registry["trajectories"], key=lambda row: row["origin"])
    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []

    def add_rows(target: list[dict[str, Any]], trajectory: dict[str, Any], order: list[int], arm: str) -> None:
        prefixes = trajectory["prefixes"]
        for count in range(len(prefixes)):
            applied = [prefixes[step]["prefix_id"] for step in order[:count]]
            step_artifacts = [prefixes[step]["step_artifact_sha256"] for step in order[:count]]
            donor_prefix = prefixes[count] if arm == "donor" else prefixes[0] if count == 0 else None
            for destination in DESTINATIONS:
                for mode in MODES:
                    for translator in translators:
                        coordinate = {
                            "campaign_id": registry["campaign_id"],
                            "operator": registry["operator"],
                            "origin": trajectory["origin"],
                            "prefix_index": count,
                            "applied_step_ids": applied,
                            "applied_step_artifact_sha256": step_artifacts,
                            "destination": destination,
                            "transfer_mode": mode,
                            "translator": translator,
                            "order_arm": arm,
                        }
                        target.append({
                            "cell_id": _cell_id(coordinate),
                            **coordinate,
                            "origin_dsl": trajectory["origin_dsl"],
                            "self_destination": destination == trajectory["origin_dsl"],
                            "pipeline_positive_control": arm == "donor" and destination == trajectory["origin_dsl"] and mode == "literal",
                            "identity_source_required": arm == "donor" and destination == trajectory["origin_dsl"] and mode == "literal",
                            "attempt_budget": 1 if mode == "literal" else registry["retuned_attempts_per_cell"],
                            "build_failures_consume_attempts": True,
                            "analysis_role": "controlling" if mode == "literal" else "descriptive_only",
                            "retune_plan_sha256": (
                                registry["retune_plan_sha256_by_origin"][trajectory["origin"]]
                                if mode == "retuned"
                                else None
                            ),
                            "donor_prefix_id": donor_prefix["prefix_id"] if donor_prefix else None,
                            "donor_source_sha256": donor_prefix["source_sha256"] if donor_prefix else None,
                            "donor_terminal_status": donor_prefix["terminal_status"] if donor_prefix else None,
                            "constructed_source_sha256_required": arm == "valid_order_control" and count > 0,
                            "target_terminal_status": None,
                        })

    eligible_orders: list[dict[str, Any]] = []
    for trajectory in trajectories:
        donor_order = list(range(1, len(trajectory["prefixes"])))
        add_rows(rows, trajectory, donor_order, "donor")
        control = _topological_control(trajectory, registry["order_seed"])
        if control is not None:
            eligible_orders.append({
                "origin": trajectory["origin"],
                "donor_step_order": donor_order,
                "control_step_order": control,
            })
            add_rows(controls, trajectory, control, "valid_order_control")

    prefix_count = sum(len(row["prefixes"]) for row in trajectories)
    factor = len(DESTINATIONS) * len(MODES) * len(translators)
    require(len(rows) == prefix_count * factor, "main census formula mismatch")
    require(len({row["cell_id"] for row in rows + controls}) == len(rows) + len(controls), "cell_id collision")
    require(all(any(row["pipeline_positive_control"] for row in rows if row["origin"] == trajectory["origin"])
                for trajectory in trajectories), "self-transfer positive control missing")

    manifest = {
        "schema_version": 1,
        "record_type": "trajectory_transfer_t1_manifest",
        "campaign_id": registry["campaign_id"],
        "operator": registry["operator"],
        "policy_status": "deferred",
        "donor_registry_sha256": _digest(registry),
        "terminal_status_taxonomy": TERMINAL_STATUS,
        "retuned_attempts_per_cell": registry["retuned_attempts_per_cell"],
        "census": {
            "donor_prefixes": prefix_count,
            "destinations": len(DESTINATIONS),
            "modes": len(MODES),
            "translators": len(translators),
            "donor_order_cells": len(rows),
            "valid_order_control_cells": len(controls),
            "total_cells": len(rows) + len(controls),
            "attempt_ceiling": sum(row["attempt_budget"] for row in rows + controls),
        },
        "valid_order_controls": eligible_orders,
        "rows": rows,
        "order_control_rows": controls,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def refuse_launch(registry: dict[str, Any]) -> None:
    blockers = ["trajectory-transfer execution is deferred by current policy"]
    if not registry.get("source_bindings"):
        blockers.append("material donor/translation source bindings are absent")
    if not registry.get("execution_abi"):
        blockers.append("build/call/gate/timing ABI is absent")
    raise ProtocolError("launch refused; " + "; ".join(blockers) + "; no subprocess started")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    try:
        registry = validate_registry(json.loads(args.registry.read_text(encoding="utf-8")))
        if args.launch:
            refuse_launch(registry)
        print(json.dumps(derive_manifest(registry), indent=2, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, ProtocolError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
