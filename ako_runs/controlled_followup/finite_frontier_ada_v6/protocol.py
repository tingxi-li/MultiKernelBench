#!/usr/bin/env python3
"""Fail-closed terminal-only successor contract for the Ada frontier study."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMPAIGN_ID = "finite-frontier-ada-v6"
CONTRACT_PATH = HERE / "contract.json"
SELECTION_BINDING_PATH = HERE / "selection_binding.json"
EXECUTION_LOCK_PATH = HERE / "execution_lock.json"
RESULTS_ROOT = HERE / "results"
PREDECESSOR = HERE.parent / "finite_frontier_ada_v5"
PREDECESSOR_INCIDENT_PATH = PREDECESSOR / "INCIDENT_TERMINAL_READINESS_20260806.json"
PREDECESSOR_INCIDENT_SHA256 = "6ff9073116e8992f9da0b1064cfd129c6ecc501bc65c96af068ab02e8a6cc120"
PREDECESSOR_RESULT_ROOT = PREDECESSOR / "results"
PREDECESSOR_RESULT_COMMIT = "d120fa42b9c6ca578f116579736db113a0678be2"
PREDECESSOR_SELECTION_PATH = PREDECESSOR_RESULT_ROOT / "selection_lock.json"
PREDECESSOR_EXECUTION_LOCK_PATH = PREDECESSOR / "execution_lock.json"
DSLS = ("tilelang", "triton")
DISTRIBUTIONS = ("positive", "withheld_signed")
SHAM_LABELS = ("sham_a", "sham_b")
WINNERS = {
    "tilelang": "register_fused.tilelang.g05",
    "triton": "register_fused.triton.g05",
}
SHAM_BASE_CELL = "register_common_postprocess.tilelang.g01"
TERMINAL_RECORDS = 120


class ProtocolError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON must contain an object: {path}")
    return value


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ProtocolError(f"path escapes repository: {path}") from exc


def _safe_repo_file(relative: Any) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ProtocolError("bound path must be safe and repository-relative")
    candidate = REPO_ROOT / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ProtocolError(f"bound file is missing or unsafe: {relative}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ProtocolError(f"bound path escapes repository: {relative}") from exc
    return resolved


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=30,
    )
    if completed.returncode:
        raise ProtocolError(completed.stderr.strip() or "git binding check failed")
    return completed.stdout.strip()


def _exact_local_module(name: str):
    expected = (HERE / f"{name}.py").resolve()
    qualified = f"ako_runs.controlled_followup.finite_frontier_ada_v6.{name}"
    for candidate in ("__main__", qualified, name):
        module = sys.modules.get(candidate)
        raw = getattr(module, "__file__", None)
        if isinstance(raw, str) and Path(raw).resolve() == expected:
            return module
    module = importlib.import_module(qualified)
    if Path(str(getattr(module, "__file__", ""))).resolve() != expected:
        raise ProtocolError(f"local {name} resolved outside the campaign root")
    return module


def local_launch_module():
    return _exact_local_module("launch")


def local_analyze_module():
    return _exact_local_module("analyze")


def _closure_rows(root: Path, excluded: set[str]) -> tuple[list[dict[str, Any]], list[Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("predecessor result closure root is missing or unsafe")
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ProtocolError(f"predecessor closure contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in excluded:
            paths.append(path)
            rows.append(
                {"path": relative, "sha256": file_sha256(path), "size": path.stat().st_size}
            )
    return rows, paths


def predecessor_result_closure_paths(incident: dict[str, Any] | None = None) -> list[Path]:
    incident = incident or read_json(PREDECESSOR_INCIDENT_PATH)
    closure = incident.get("artifact_selection_closure") if isinstance(incident, dict) else None
    required = {
        "algorithm", "excluded_runtime_files", "file_count", "git_tree_oid",
        "root", "sha256", "total_bytes",
    }
    if not isinstance(closure, dict) or set(closure) != required:
        raise ProtocolError("v5 artifact+selection closure schema changed")
    excluded = {
        "artifact_admission/active.lock", "selection_confirm/active.lock"
    }
    if (
        closure["root"] != repo_path(PREDECESSOR_RESULT_ROOT)
        or closure["excluded_runtime_files"] != sorted(excluded)
        or closure["algorithm"]
        != "sha256(canonical_json(sorted([{path,sha256,size}])))"
        or closure["file_count"] != 376
        or closure["total_bytes"] != 24792370
        or closure["git_tree_oid"] != "63d8d57df76154f4312e6aca7021caf42d51f7ba"
    ):
        raise ProtocolError("v5 artifact+selection closure census changed")
    rows, paths = _closure_rows(PREDECESSOR_RESULT_ROOT, excluded)
    if (
        len(rows) != closure["file_count"]
        or sum(row["size"] for row in rows) != closure["total_bytes"]
        or canonical_sha256(rows) != closure["sha256"]
    ):
        raise ProtocolError("sealed v5 artifact+selection bytes changed")
    return paths


def predecessor_source_paths(incident: dict[str, Any] | None = None) -> list[Path]:
    incident = incident or read_json(PREDECESSOR_INCIDENT_PATH)
    closure = incident.get("source_closure") if isinstance(incident, dict) else None
    required = {
        "algorithm", "campaign_git_tree_oid", "file_count", "files", "sha256",
        "total_bytes",
    }
    if not isinstance(closure, dict) or set(closure) != required:
        raise ProtocolError("v5 source closure schema changed")
    files = closure["files"]
    if (
        closure["algorithm"] != "sha256(canonical_json(sorted([{path,sha256,size}])))"
        or closure["campaign_git_tree_oid"] != "15449c05571cf39296b7c26a723dc4ce56e78eb4"
        or closure["file_count"] != 11
        or closure["total_bytes"] != 196476
        or not isinstance(files, list)
        or len(files) != 11
        or canonical_sha256(files) != closure["sha256"]
    ):
        raise ProtocolError("v5 source closure census changed")
    paths = []
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise ProtocolError("v5 source binding is malformed")
        path = _safe_repo_file(row["path"])
        if path.stat().st_size != row["size"] or file_sha256(path) != row["sha256"]:
            raise ProtocolError(f"sealed v5 source changed: {row['path']}")
        paths.append(path)
    return paths


def predecessor_incident() -> dict[str, Any]:
    if file_sha256(PREDECESSOR_INCIDENT_PATH) != PREDECESSOR_INCIDENT_SHA256:
        raise ProtocolError("v5 terminal-readiness incident changed")
    incident = read_json(PREDECESSOR_INCIDENT_PATH)
    policy = incident.get("policy", {})
    selection = incident.get("selection", {})
    census = incident.get("sealed_census", {})
    if (
        incident.get("campaign_id") != "finite-frontier-ada-v5"
        or incident.get("classification")
        != "NON_CONTROLLING_TERMINAL_READINESS_INCIDENT"
        or incident.get("result_commit") != PREDECESSOR_RESULT_COMMIT
        or policy.get("continuation_authorized") is not False
        or policy.get("executable_artifact_reuse_authorized") is not False
        or policy.get("selection_performance_reuse_authorized") is not False
        or policy.get("selection_winner_role_reuse_authorized") is not True
        or policy.get("successor_requires_exact_package_safe_modules_and_children") is not True
        or policy.get("successor_requires_fresh_three_artifact_admission") is not True
        or policy.get("successor_requires_new_lock") is not True
        or policy.get("successor_requires_terminal_only_timing") is not True
        or census.get("observed_artifact_admission_entries") != 7
        or census.get("observed_selection_timing_records") != 240
        or census.get("observed_terminal_timing_records") != 0
        or selection.get("result_role") != "preregistered_winner_selection_only"
        or selection.get("winners") != WINNERS
        or selection.get("terminal_authorized") is not True
        or selection.get("selection_sham_floor_log_ratio") != 0.01686377744677924
    ):
        raise ProtocolError("v5 incident is not the frozen terminal-readiness failure")
    predecessor_source_paths(incident)
    predecessor_result_closure_paths(incident)
    _git("cat-file", "-e", f"{PREDECESSOR_RESULT_COMMIT}^{{commit}}")
    if _git(
        "rev-parse",
        f"{PREDECESSOR_RESULT_COMMIT}:{repo_path(PREDECESSOR_RESULT_ROOT)}",
    ) != incident["artifact_selection_closure"]["git_tree_oid"]:
        raise ProtocolError("v5 result commit lost the sealed result tree")
    if _git(
        "rev-parse", f"{PREDECESSOR_RESULT_COMMIT}:{repo_path(PREDECESSOR)}"
    ) != incident["source_closure"]["campaign_git_tree_oid"]:
        raise ProtocolError("v5 result commit lost the frozen campaign tree")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREDECESSOR_RESULT_COMMIT, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, timeout=30,
    ).returncode:
        raise ProtocolError("v5 result commit is not in current HEAD history")
    return incident


def exact_median_interval(values: Iterable[float]) -> dict[str, Any]:
    """Exact sign-test interval used by the frozen source instrument."""
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value <= 0 for value in ordered):
        raise ProtocolError("median interval requires positive finite values")
    n = len(ordered)
    choices = []
    for k in range(1, (n + 1) // 2 + 1):
        coverage = 1.0 - 2.0 * sum(math.comb(n, index) for index in range(k)) / (2**n)
        if coverage >= 0.95:
            choices.append((k, coverage))
    if choices:
        k, coverage = max(choices)
        lo, hi, meets = ordered[k - 1], ordered[n - k], True
    else:
        k, coverage = 1, 1.0 - 2.0 / (2**n)
        lo, hi, meets = ordered[0], ordered[-1], False
    return {
        "n": n,
        "median": statistics.median(ordered),
        "ci_lo": lo,
        "ci_hi": hi,
        "order_k": k,
        "coverage": coverage,
        "meets_95": meets,
    }


def resolution_floor(intervals: Iterable[dict[str, Any]]) -> float:
    endpoints = [
        float(value)
        for interval in intervals
        for value in (interval["ci_lo"], interval["ci_hi"])
    ]
    if not endpoints or any(value <= 0 or not math.isfinite(value) for value in endpoints):
        raise ProtocolError("resolution floor requires positive finite sham intervals")
    return max(abs(math.log(value)) for value in endpoints)


def _predecessor_contract(incident: dict[str, Any] | None = None) -> dict[str, Any]:
    incident = incident or predecessor_incident()
    path = PREDECESSOR / "contract.json"
    source_hashes = {
        row["path"]: row["sha256"] for row in incident["source_closure"]["files"]
    }
    if source_hashes.get(repo_path(path)) != file_sha256(path):
        raise ProtocolError("sealed v5 contract is absent from the source closure")
    contract = read_json(path)
    if contract.get("campaign_id") != "finite-frontier-ada-v5":
        raise ProtocolError("sealed predecessor contract has another campaign ID")
    return contract


def _predecessor_timing_plan(
    stage: str, contract: dict[str, Any], winners: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    manifest = contract["manifest"]
    if stage == "selection_confirm":
        candidates = [
            cell_id
            for dsl in DSLS
            for cell_id in manifest["selection_confirm"]["candidate_ids"][dsl]
        ]
        seed = manifest["selection_confirm"]["order_seed"]
        expected = 240
    elif stage == "terminal_confirm":
        if winners != WINNERS:
            raise ProtocolError("terminal winners differ from the sealed selection")
        candidates = [winners[dsl] for dsl in DSLS]
        seed = manifest["terminal_confirm"]["order_seed"]
        expected = TERMINAL_RECORDS
    else:
        raise ProtocolError(f"unknown predecessor timing stage: {stage}")
    randomizer, plan = random.Random(seed), []
    for block in range(15):
        rows = [
            {
                "block": block,
                "cell_id": cell_id,
                "distribution": distribution,
                "label": cell_id,
                "record_kind": "candidate",
                "stage": stage,
            }
            for cell_id in candidates
            for distribution in DISTRIBUTIONS
        ]
        rows.extend(
            {
                "block": block,
                "cell_id": SHAM_BASE_CELL,
                "distribution": distribution,
                "label": label,
                "record_kind": "sham",
                "stage": stage,
            }
            for label in SHAM_LABELS
            for distribution in DISTRIBUTIONS
        )
        randomizer.shuffle(rows)
        for block_position, row in enumerate(rows):
            row["block_position"] = block_position
            row["position"] = len(plan)
            plan.append(row)
    if len(plan) != expected or len({
        (row["block"], row["label"], row["distribution"]) for row in plan
    }) != expected:
        raise ProtocolError("predecessor timing plan census changed")
    return plan


def _group_selection_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        row = record["row"]
        grouped[(row["label"], row["distribution"])].append(
            (row["block"], float(record["primary_tail_median_ms"]))
        )
    for values in grouped.values():
        values.sort()
    return grouped


def _selection_values(
    grouped: dict[tuple[str, str], list[tuple[int, float]]],
    label: str,
    distribution: str,
) -> list[float]:
    rows = grouped.get((label, distribution), [])
    if [block for block, _value in rows] != list(range(15)):
        raise ProtocolError(f"incomplete sealed selection blocks: {label}/{distribution}")
    return [value for _block, value in rows]


def _rederive_predecessor_selection(
    incident: dict[str, Any], contract: dict[str, Any],
) -> dict[str, Any]:
    del incident  # closure validation precedes this semantic re-derivation
    plan = _predecessor_timing_plan("selection_confirm", contract)
    root = PREDECESSOR_RESULT_ROOT / "selection_confirm"
    raw = root / "raw"
    validate_raw_census(raw, plan)
    receipt_path, status_path = root / "launch_receipt.json", root / "run_status.json"
    receipt, status = read_json(receipt_path), read_json(status_path)
    launch = receipt.get("contract", {})
    if (
        receipt.get("schema_version") != 1
        or receipt.get("record_type") != "finite_frontier_ada_launch_receipt"
        or launch.get("campaign_id") != "finite-frontier-ada-v5"
        or launch.get("stage") != "selection_confirm"
        or launch.get("plan") != plan
        or launch.get("plan_sha256") != canonical_sha256(plan)
        or launch.get("execution_lock_sha256") != file_sha256(PREDECESSOR_EXECUTION_LOCK_PATH)
        or launch.get("timing") != contract["manifest"]["timing"]
        or status.get("campaign_id") != "finite-frontier-ada-v5"
        or status.get("stage") != "selection_confirm"
        or status.get("complete") is not True
        or status.get("expected_records") != 240
        or status.get("observed_records") != 240
        or status.get("launch_receipt_sha256") != file_sha256(receipt_path)
    ):
        raise ProtocolError("sealed v5 selection receipt/status changed semantically")
    records, hashes, process_ids = [], [
        {"path": repo_path(path), "sha256": file_sha256(path)}
        for path in (receipt_path, status_path)
    ], set()
    for row in plan:
        path = raw / timing_filename(row)
        record = read_json(path)
        times = record.get("times_ms")
        summary = summarize_times(times if isinstance(times, list) else [])
        process_id = record.get("process_pid")
        if (
            record.get("schema_version") != 1
            or record.get("record_type") != "finite_frontier_ada_timing_record"
            or record.get("campaign_id") != "finite-frontier-ada-v5"
            or record.get("stage") != "selection_confirm"
            or record.get("row") != row
            or record.get("row_sha256") != canonical_sha256(row)
            or record.get("plan_position") != row["position"]
            or record.get("cell_id") != row["cell_id"]
            or record.get("label") != row["label"]
            or record.get("distribution") != row["distribution"]
            or record.get("record_kind") != row["record_kind"]
            or record.get("ok") is not True
            or record.get("trials") != 100
            or record.get("warmup_s") != 2.0
            or record.get("legacy_error", {}).get("gate_pass") is not True
            or record.get("build_metadata", {}).get("n_kernels") != 2
            or not isinstance(process_id, int)
            or process_id in process_ids
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("implementation_sha256"))) is None
            or any(record.get(key) != value for key, value in summary.items())
        ):
            raise ProtocolError(f"sealed v5 selection record changed: {path.name}")
        process_ids.add(process_id)
        records.append(record)
        hashes.append({"path": repo_path(path), "sha256": file_sha256(path)})
    grouped = _group_selection_records(records)
    candidates = contract["manifest"]["selection_confirm"]["candidate_ids"]
    selection_rows, winners = [], {}
    for dsl in DSLS:
        ranked = []
        for cell_id in candidates[dsl]:
            positive = _selection_values(grouped, cell_id, "positive")
            signed = _selection_values(grouped, cell_id, "withheld_signed")
            median = statistics.median(positive)
            ranked.append((median, cell_id))
            selection_rows.append(
                {
                    "cell_id": cell_id,
                    "dsl": dsl,
                    "positive_interval": exact_median_interval(positive),
                    "positive_median_ms": median,
                    "withheld_signed_diagnostic_interval": exact_median_interval(signed),
                }
            )
        winners[dsl] = min(ranked)[1]
    implementations: dict[str, set[str]] = defaultdict(set)
    for record in records:
        implementations[record["label"]].add(record["implementation_sha256"])
    if (
        any(len(values) != 1 for values in implementations.values())
        or implementations[SHAM_LABELS[0]] != implementations[SHAM_LABELS[1]]
    ):
        raise ProtocolError("sealed selection labels do not bind stable implementations")
    sham_intervals = {
        distribution: exact_median_interval(
            left / right for left, right in zip(
                _selection_values(grouped, SHAM_LABELS[0], distribution),
                _selection_values(grouped, SHAM_LABELS[1], distribution),
            )
        )
        for distribution in DISTRIBUTIONS
    }
    floor = resolution_floor(sham_intervals.values())
    terminal_plan = _predecessor_timing_plan("terminal_confirm", contract, winners)
    return {
        "schema_version": 1,
        "record_type": "finite_frontier_ada_selection_lock",
        "campaign_id": "finite-frontier-ada-v5",
        "complete": True,
        "execution_lock_sha256": file_sha256(PREDECESSOR_EXECUTION_LOCK_PATH),
        "selection_records": selection_rows,
        "selection_stage_hashes": hashes,
        "selection_stage_plan_sha256": canonical_sha256(plan),
        "sham": {
            "source_config_implementation_sha256": next(iter(implementations[SHAM_LABELS[0]])),
            "intervals": sham_intervals,
            "resolution_floor_log_ratio": floor,
        },
        "terminal_authorized": (
            floor <= contract["manifest"]["inference"]["max_selection_sham_floor_log_ratio"]
        ),
        "terminal_plan_sha256": canonical_sha256(terminal_plan),
        "winners": winners,
    }


def derive_selection_binding() -> dict[str, Any]:
    incident = predecessor_incident()
    predecessor_contract = _predecessor_contract(incident)
    retained = read_json(PREDECESSOR_SELECTION_PATH)
    derived = _rederive_predecessor_selection(incident, predecessor_contract)
    if retained != derived:
        raise ProtocolError("sealed v5 selection lock differs from independent re-derivation")
    hashes = derived["selection_stage_hashes"]
    raw = [row for row in hashes if "/selection_confirm/raw/" in row["path"]]
    plan = _predecessor_timing_plan("terminal_confirm", predecessor_contract, WINNERS)
    if (
        derived["terminal_authorized"] is not True
        or derived["winners"] != WINNERS
        or derived["sham"]["resolution_floor_log_ratio"] != 0.01686377744677924
        or len(hashes) != 242
        or len(raw) != 240
        or file_sha256(PREDECESSOR_SELECTION_PATH)
        != incident["selection"]["selection_lock_sha256"]
        or len(plan) != TERMINAL_RECORDS
        or canonical_sha256(plan) != incident["selection"]["source_terminal_plan_sha256"]
        or derived["terminal_plan_sha256"] != canonical_sha256(plan)
    ):
        raise ProtocolError("v5 selection winner/authorization projection changed")
    return {
        "campaign_id": CAMPAIGN_ID,
        "predecessor_incident_path": repo_path(PREDECESSOR_INCIDENT_PATH),
        "predecessor_incident_sha256": PREDECESSOR_INCIDENT_SHA256,
        "predecessor_result_closure_sha256": incident["artifact_selection_closure"]["sha256"],
        "record_type": "finite_frontier_ada_v6_selection_binding",
        "schema_version": 1,
        "selection_evidence_bundle_sha256": canonical_sha256(hashes),
        "selection_record_count": 240,
        "selection_role": "preregistered_winner_selection_only",
        "selection_sham_floor_log_ratio": 0.01686377744677924,
        "selection_stage_hashes": deepcopy(hashes),
        "source_campaign_id": "finite-frontier-ada-v5",
        "source_execution_lock_path": repo_path(PREDECESSOR_EXECUTION_LOCK_PATH),
        "source_execution_lock_sha256": file_sha256(PREDECESSOR_EXECUTION_LOCK_PATH),
        "source_result_commit": PREDECESSOR_RESULT_COMMIT,
        "source_selection_lock_path": repo_path(PREDECESSOR_SELECTION_PATH),
        "source_selection_lock_sha256": file_sha256(PREDECESSOR_SELECTION_PATH),
        "source_terminal_plan_sha256": canonical_sha256(plan),
        "terminal_authorized": True,
        "terminal_plan_sha256": canonical_sha256(plan),
        "winners": deepcopy(WINNERS),
    }


def validate_selection_binding(value: Any) -> dict[str, Any]:
    expected = derive_selection_binding()
    if value != expected:
        raise ProtocolError("v6 selection binding differs from sealed v5 re-derivation")
    return expected


def load_selection_binding() -> dict[str, Any]:
    return validate_selection_binding(read_json(SELECTION_BINDING_PATH))


def _build_contract(selection: dict[str, Any]) -> dict[str, Any]:
    predecessor_contract = _predecessor_contract(read_json(PREDECESSOR_INCIDENT_PATH))
    old_manifest = predecessor_contract["manifest"]
    material_registry = deepcopy(predecessor_contract["material_registry"])
    material_registry["roles"].update(
        {
            "predecessor_terminal_incident": repo_path(PREDECESSOR_INCIDENT_PATH),
            "predecessor_v5_execution_lock": repo_path(PREDECESSOR_EXECUTION_LOCK_PATH),
            "predecessor_v5_selection_lock": repo_path(PREDECESSOR_SELECTION_PATH),
            "selection_binding": repo_path(SELECTION_BINDING_PATH),
        }
    )
    material_registry["sha256"].update(
        {
            repo_path(PREDECESSOR_INCIDENT_PATH): PREDECESSOR_INCIDENT_SHA256,
            repo_path(PREDECESSOR_EXECUTION_LOCK_PATH): file_sha256(PREDECESSOR_EXECUTION_LOCK_PATH),
            repo_path(PREDECESSOR_SELECTION_PATH): file_sha256(PREDECESSOR_SELECTION_PATH),
            repo_path(SELECTION_BINDING_PATH): file_sha256(SELECTION_BINDING_PATH),
        }
    )
    roles, hashes = material_registry.get("roles"), material_registry.get("sha256")
    if (
        set(material_registry) != {"source_campaign_id", "source_result_tag", "roles", "sha256"}
        or not isinstance(roles, dict)
        or not isinstance(hashes, dict)
        or len(set(roles.values())) != len(roles)
        or set(hashes) != set(roles.values())
    ):
        raise ProtocolError("material registry is not one exact path/hash bijection")
    for relative, expected in hashes.items():
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or file_sha256(_safe_repo_file(relative)) != expected
        ):
            raise ProtocolError(f"registered material changed: {relative}")
    admission = deepcopy(old_manifest["artifact_admission"])
    admission.update({"candidate_artifacts": 2, "entry_count": 3})
    return {
        "broader_f1_outside_scope": predecessor_contract["broader_f1_outside_scope"],
        "campaign_id": CAMPAIGN_ID,
        "claim_scope": predecessor_contract["claim_scope"],
        "estimand": predecessor_contract["estimand"],
        "manifest": {
            "artifact_admission": admission,
            "hardware": deepcopy(old_manifest["hardware"]),
            "inference": deepcopy(old_manifest["inference"]),
            "selection_basis": {
                "selection_binding_path": repo_path(SELECTION_BINDING_PATH),
                "selection_binding_sha256": file_sha256(SELECTION_BINDING_PATH),
                "selection_record_count": selection["selection_record_count"],
                "selection_role": selection["selection_role"],
                "selection_sham_floor_log_ratio": selection["selection_sham_floor_log_ratio"],
                "sham_base_cell": SHAM_BASE_CELL,
                "sham_labels": list(SHAM_LABELS),
                "source_campaign_id": selection["source_campaign_id"],
                "source_result_commit": selection["source_result_commit"],
                "terminal_authorized": selection["terminal_authorized"],
                "terminal_plan_sha256": selection["terminal_plan_sha256"],
                "winners": deepcopy(selection["winners"]),
            },
            "terminal_confirm": deepcopy(old_manifest["terminal_confirm"]),
            "timing": deepcopy(old_manifest["timing"]),
            "toolchain": deepcopy(old_manifest["toolchain"]),
            "workload": deepcopy(old_manifest["workload"]),
        },
        "material_registry": material_registry,
        "question": predecessor_contract["question"],
        "schema_version": 1,
        "state": "prepared_terminal_only_requires_execution_lock_and_remote_commit",
    }


def make_contract() -> dict[str, Any]:
    predecessor_incident()
    return _build_contract(load_selection_binding())


def validate_contract(value: Any) -> dict[str, Any]:
    expected = make_contract()
    if value != expected:
        raise ProtocolError("contract differs from the sealed terminal-only projection")
    return expected


def load_contract() -> dict[str, Any]:
    return validate_contract(read_json(CONTRACT_PATH))


def timing_plan(
    stage: str = "terminal_confirm", winners: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if stage != "terminal_confirm":
        raise ProtocolError("v6 authorizes terminal-confirm only")
    selection = load_selection_binding()
    if winners is not None and winners != selection["winners"]:
        raise ProtocolError("terminal winners differ from the sealed v5 selection")
    predecessor_contract = _predecessor_contract(read_json(PREDECESSOR_INCIDENT_PATH))
    plan = _predecessor_timing_plan(
        "terminal_confirm", predecessor_contract, selection["winners"]
    )
    if (
        len(plan) != TERMINAL_RECORDS
        or canonical_sha256(plan) != selection["terminal_plan_sha256"]
        or any(row.get("stage") != "terminal_confirm" for row in plan)
    ):
        raise ProtocolError("terminal plan differs from the sealed v5 plan")
    return deepcopy(plan)


def validate_raw_census(raw: Path, plan: list[dict[str, Any]]) -> None:
    expected = {timing_filename(row) for row in plan}
    if len(expected) != len(plan):
        raise ProtocolError("terminal plan filenames are not unique")
    observed = {path.name for path in raw.iterdir()} if raw.is_dir() else set()
    if observed != expected:
        missing, extra = sorted(expected - observed), sorted(observed - expected)
        raise ProtocolError(
            f"raw timing census differs: missing={missing[:3]} extra={extra[:3]}"
        )


def timing_filename(row: dict[str, Any]) -> str:
    label = row["label"].replace(".", "__")
    return f"{label}__{row['distribution']}__block{row['block']:02d}.json"


def summarize_times(times: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in times]
    if len(values) != 100 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ProtocolError("timing requires 100 positive finite trials")
    first, last = statistics.median(values[:10]), statistics.median(values[90:])
    return {
        "first_decile_median_ms": first,
        "first_to_last_decile_ratio": last / first,
        "full_median_ms": statistics.median(values),
        "last_decile_median_ms": last,
        "primary_tail_median_ms": statistics.median(values[60:100]),
    }


def local_source_paths() -> list[Path]:
    names = (
        ".gitignore", "README.md", "__init__.py", "contract.json", "protocol.py",
        "artifacts.py", "admit.py", "launch.py", "analyze.py", "test_protocol.py",
    )
    paths = [HERE / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ProtocolError("terminal-only successor source set is incomplete")
    return paths
