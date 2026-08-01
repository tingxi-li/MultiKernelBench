#!/usr/bin/env python3
"""Request, execute, independently validate, and freeze the common KC ladder."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Support both ``python file.py`` and ``python -m package.module``.
    from . import common, isolation, registry, treatment_plan
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    import common
    import isolation
    import registry
    import treatment_plan

sys.path.insert(0, str(common.REPO_ROOT))
from ako_runs.controlled_followup.robust_gate.schema import load_records  # noqa: E402
from ako_runs.controlled_followup.robust_gate.validate import validate_records  # noqa: E402


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
ROBUST_MANIFEST = common.BASE.parent / "robust_gate/manifest_matmul_v4.json"


class KCLadderError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise KCLadderError(message)


def request_path(kc: int) -> Path:
    return common.KC_OUTPUT_ROOT / "requests" / f"kc_{kc}.json"


def attempt_path(kc: int) -> Path:
    return common.KC_OUTPUT_ROOT / f"kc_{kc}" / "attempt_summary.json"


def raw_path(kc: int, cell_key: str) -> Path:
    return common.KC_OUTPUT_ROOT / f"kc_{kc}" / "raw" / f"{cell_key}.jsonl"


def cell_summary_path(kc: int, cell_key: str) -> Path:
    return common.KC_OUTPUT_ROOT / f"kc_{kc}" / "validated" / f"{cell_key}.json"


def _registry() -> dict[str, Any]:
    require(common.IMPLEMENTATION_REGISTRY.is_file(), "implementation registry missing")
    value = common.load_json(common.IMPLEMENTATION_REGISTRY)
    require(value == registry.document(), "implementation registry stale")
    return value


def _literal(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["cell_id"]: row for row in value["implementations"] if row["transfer_mode"] == "literal"}


def validate_authorization(path: Path, runner: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "KC authorization missing/symlinked")
    require(runner.is_file() and not runner.is_symlink(), "KC runner missing/symlinked")
    value = common.load_json(path)
    expected = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_kc_execution_authorization",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "authorized",
        "legacy_source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_sha256": common.file_sha256(common.SOURCE_FREEZE),
        "translator_isolation_sha256": common.file_sha256(common.TRANSLATOR_ISOLATION_LOCK),
        "implementation_registry_sha256": common.file_sha256(common.IMPLEMENTATION_REGISTRY),
        "kc_plan_sha256": common.file_sha256(common.KC_PLAN),
        "gate_lock_sha256": common.file_sha256(common.base.GATE_LOCK),
        "runner_path": common.repo_path(runner.resolve()),
        "runner_sha256": common.file_sha256(runner.resolve()),
        "remote_push_verified": True,
    }
    for key, expected_value in expected.items():
        require(value.get(key) == expected_value, f"authorization {key} mismatch")
    require(COMMIT_RE.fullmatch(str(value.get("git_commit", ""))) is not None, "authorization commit missing")
    require(isinstance(value.get("external_timestamp_utc"), str) and value["external_timestamp_utc"], "authorization timestamp missing")
    gpu_uuids = value.get("authorized_gpu_uuids")
    require(
        isinstance(gpu_uuids, list)
        and len(gpu_uuids) == 4
        and len(set(gpu_uuids)) == 4
        and all(isinstance(item, str) and item.startswith("GPU-") for item in gpu_uuids),
        "authorization must bind four distinct GPUs",
    )
    return value


def validate_attempt(
    value: dict[str, Any], kc: int, *, verify_artifacts: bool = False
) -> None:
    require(
        value.get("record_type") == "reciprocal_v2_kc_attempt_summary"
        and value.get("campaign_id") == common.base.CAMPAIGN_ID
        and value.get("kc") == kc
        and value.get("coverage_complete") is True,
        f"KC={kc}: attempt header/coverage mismatch",
    )
    cells = value.get("cells")
    expected = [row["cell_key"] for row in treatment_plan.kc_plan()["cells"]]
    require(isinstance(cells, list) and [row.get("cell_key") for row in cells] == expected, f"KC={kc}: cell census/order mismatch")
    derived_flags = [row.get("all_required_cases_pass") is True for row in cells]
    if verify_artifacts:
        request = common.load_json(request_path(kc))
        validate_request(request, kc)
        derived_flags = []
        for row, request_cell in zip(cells, request["cells"], strict=True):
            key = request_cell["cell_key"]
            raw = raw_path(kc, key)
            summary_path = cell_summary_path(kc, key)
            require(
                row.get("raw_path") == common.repo_path(raw)
                and raw.is_file()
                and not raw.is_symlink()
                and row.get("raw_sha256") == common.file_sha256(raw),
                f"KC={kc} {key}: raw binding mismatch",
            )
            require(
                row.get("v4_summary_path") == common.repo_path(summary_path)
                and summary_path.is_file()
                and not summary_path.is_symlink()
                and row.get("v4_summary_sha256") == common.file_sha256(summary_path),
                f"KC={kc} {key}: summary binding mismatch",
            )
            result = _validate_raw_records(load_records(raw), request_cell, kc)
            summary = common.load_json(summary_path)
            passed = result["success"]
            require(
                summary.get("record_type") == "reciprocal_v2_kc_v4_summary"
                and summary.get("campaign_id") == common.base.CAMPAIGN_ID
                and summary.get("cell_key") == key
                and summary.get("kc") == kc
                and summary.get("implementation_sha256")
                == request_cell["implementation_sha256"]
                and summary.get("gate_spec_sha256")
                == common.file_sha256(common.base.GATE_SPEC)
                and summary.get("coverage_complete") is True
                and summary.get("raw_path") == common.repo_path(raw)
                and summary.get("raw_sha256") == common.file_sha256(raw)
                and summary.get("robust_validation") == result
                and summary.get("robust_validation_sha256")
                == common.canonical_sha256(result)
                and summary.get("all_required_cases_pass") is passed,
                f"KC={kc} {key}: summary is not raw-derived",
            )
            require(
                row.get("implementation_sha256")
                == request_cell["implementation_sha256"]
                and row.get("robust_validation_sha256")
                == common.canonical_sha256(result)
                and row.get("all_required_cases_pass") is passed,
                f"KC={kc} {key}: attempt cell is not raw-derived",
            )
            derived_flags.append(passed)
    derived = len(cells) == 24 and all(derived_flags)
    require(value.get("all_cells_all_v4_cases_pass") is derived, f"KC={kc}: aggregate flag is not derived")


def validate_request(value: dict[str, Any], kc: int) -> None:
    require(
        value.get("schema_version") == 1
        and value.get("record_type") == "reciprocal_v2_kc_attempt_request"
        and value.get("supplement_id") == common.SUPPLEMENT_ID
        and value.get("campaign_id") == common.base.CAMPAIGN_ID
        and value.get("state") == "requested_not_executed"
        and value.get("kc") == kc
        and value.get("cell_count") == 24
        and value.get("performance_measurements_authorized") is False,
        f"KC={kc}: request header mismatch",
    )
    registry_value = _registry()
    literal = _literal(registry_value)
    plan = treatment_plan.kc_plan()
    cells = value.get("cells")
    require(
        isinstance(cells, list)
        and [row.get("cell_key") for row in cells]
        == [row["cell_key"] for row in plan["cells"]],
        f"KC={kc}: request cell census/order mismatch",
    )
    for observed, planned in zip(cells, plan["cells"], strict=True):
        implementation = literal[planned["literal_cell_id"]]
        expected = {
            **planned,
            "kc": kc,
            "candidate": f"reciprocal_kc::{planned['cell_key']}::kc{kc}",
            "implementation_path": implementation["source"],
            "implementation_sha256": implementation["sha256"],
            "raw_output_path": common.repo_path(raw_path(kc, planned["cell_key"])),
        }
        require(observed == expected, f"KC={kc} {planned['cell_key']}: request cell mismatch")
    authorization = common.REPO_ROOT / str(value.get("execution_authorization_path", ""))
    runner = common.REPO_ROOT / str(value.get("runner_path", ""))
    auth = validate_authorization(authorization.resolve(), runner.resolve())
    expected_header = {
        "predecessor_attempt_sha256": (
            common.file_sha256(
                attempt_path(
                    common.base.KC_LADDER[
                        common.base.KC_LADDER.index(kc) - 1
                    ]
                )
            )
            if kc != common.base.KC_LADDER[0]
            else None
        ),
        "legacy_source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_sha256": common.file_sha256(common.SOURCE_FREEZE),
        "robust_manifest_path": common.repo_path(ROBUST_MANIFEST),
        "robust_manifest_sha256": common.file_sha256(ROBUST_MANIFEST),
        "gate_spec_path": common.repo_path(common.base.GATE_SPEC),
        "gate_spec_file_sha256": common.file_sha256(common.base.GATE_SPEC),
        "gate_lock_sha256": common.file_sha256(common.base.GATE_LOCK),
        "implementation_registry_sha256": common.file_sha256(common.IMPLEMENTATION_REGISTRY),
        "execution_authorization_path": common.repo_path(authorization),
        "execution_authorization_sha256": common.file_sha256(authorization),
        "runner_path": common.repo_path(runner),
        "runner_sha256": common.file_sha256(runner),
        "authorized_gpu_uuids": auth["authorized_gpu_uuids"],
    }
    for key, expected in expected_header.items():
        require(value.get(key) == expected, f"KC={kc}: request {key} mismatch")


def next_kc() -> int | None:
    ladder = list(common.base.KC_LADDER)
    for position, kc in enumerate(ladder):
        request_exists = request_path(kc).is_file()
        attempt_exists = attempt_path(kc).is_file()
        if attempt_exists:
            require(request_exists, f"KC={kc}: attempt exists without request")
            attempt = common.load_json(attempt_path(kc))
            validate_attempt(attempt, kc, verify_artifacts=True)
            if attempt["all_cells_all_v4_cases_pass"]:
                require(
                    not any(
                        request_path(lower).exists() or attempt_path(lower).exists()
                        for lower in ladder[position + 1 :]
                    ),
                    f"KC={kc}: lower-KC artifacts exist after a passing attempt",
                )
                return None
            continue
        if request_exists:
            require(
                not any(
                    request_path(lower).exists() or attempt_path(lower).exists()
                    for lower in ladder[position + 1 :]
                ),
                f"KC={kc}: multiple/out-of-order requests exist",
            )
            raise KCLadderError(f"KC={kc} request is outstanding; capture it before advancing")
        require(
            not any(
                request_path(lower).exists() or attempt_path(lower).exists()
                for lower in ladder[position + 1 :]
            ),
            f"KC={kc}: lower-KC artifact exists before its predecessor",
        )
        return kc
    return None


def make_request(runner: Path, authorization: Path) -> Path:
    kc = next_kc()
    require(kc is not None, "no next KC: ladder passed or exhausted")
    reg = _registry()
    auth = validate_authorization(authorization.resolve(), runner.resolve())
    literal = _literal(reg)
    cells = []
    for row in treatment_plan.kc_plan()["cells"]:
        implementation = literal[row["literal_cell_id"]]
        cells.append(
            {
                **row,
                "kc": kc,
                "candidate": f"reciprocal_kc::{row['cell_key']}::kc{kc}",
                "implementation_path": implementation["source"],
                "implementation_sha256": implementation["sha256"],
                "raw_output_path": common.repo_path(raw_path(kc, row["cell_key"])),
            }
        )
    document = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_kc_attempt_request",
        "supplement_id": common.SUPPLEMENT_ID,
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "requested_not_executed",
        "kc": kc,
        "predecessor_attempt_sha256": (
            common.file_sha256(attempt_path(common.base.KC_LADDER[common.base.KC_LADDER.index(kc) - 1]))
            if kc != common.base.KC_LADDER[0]
            else None
        ),
        "cell_count": 24,
        "cells": cells,
        "robust_manifest_path": common.repo_path(ROBUST_MANIFEST),
        "robust_manifest_sha256": common.file_sha256(ROBUST_MANIFEST),
        "legacy_source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_sha256": common.file_sha256(common.SOURCE_FREEZE),
        "gate_spec_path": common.repo_path(common.base.GATE_SPEC),
        "gate_spec_file_sha256": common.file_sha256(common.base.GATE_SPEC),
        "gate_lock_sha256": common.file_sha256(common.base.GATE_LOCK),
        "implementation_registry_sha256": common.file_sha256(common.IMPLEMENTATION_REGISTRY),
        "execution_authorization_path": common.repo_path(authorization),
        "execution_authorization_sha256": common.file_sha256(authorization),
        "runner_path": common.repo_path(runner),
        "runner_sha256": common.file_sha256(runner),
        "authorized_gpu_uuids": auth["authorized_gpu_uuids"],
        "performance_measurements_authorized": False,
    }
    isolation.exclusive_json(request_path(kc), document)
    return request_path(kc)


def _validate_raw_records(records: list[dict[str, Any]], cell: dict[str, Any], kc: int) -> dict[str, Any]:
    require(records, f"{cell['cell_key']}: raw records absent")
    for record in records:
        required = {
            "candidate": cell["candidate"],
            "reciprocal_campaign_id": common.base.CAMPAIGN_ID,
            "reciprocal_cell_key": cell["cell_key"],
            "reciprocal_kc": kc,
            "implementation_sha256": cell["implementation_sha256"],
        }
        for key, expected in required.items():
            require(record.get(key) == expected, f"{cell['cell_key']}: raw {key} mismatch")
    result = validate_records(
        common.load_json(ROBUST_MANIFEST),
        common.load_json(common.base.GATE_SPEC),
        records,
    )
    groups = result["groups"]
    require(
        len(groups) == 3
        and {row["gate_id"] for row in groups}
        == {"conformance_mixed", "semantic_mixed", "semantic_q32"}
        and all(row["coverage_complete"] for row in groups),
        f"{cell['cell_key']}: robust-v4 group coverage mismatch",
    )
    return result


def capture() -> Path:
    outstanding = [kc for kc in common.base.KC_LADDER if request_path(kc).is_file() and not attempt_path(kc).is_file()]
    require(len(outstanding) == 1, "expected exactly one outstanding KC request")
    kc = outstanding[0]
    request = common.load_json(request_path(kc))
    validate_request(request, kc)
    require(not attempt_path(kc).exists(), f"KC={kc}: attempt already exists")
    for cell in request["cells"]:
        require(
            not cell_summary_path(kc, cell["cell_key"]).exists(),
            f"{cell['cell_key']}: unreconciled validated summary exists",
        )
    validated = []
    for cell in request["cells"]:
        path = raw_path(kc, cell["cell_key"])
        require(path.is_file() and not path.is_symlink(), f"{cell['cell_key']}: raw file missing/symlinked")
        result = _validate_raw_records(load_records(path), cell, kc)
        validated.append((cell, path, result))
    cells = []
    for cell, path, result in validated:
        compatibility = {
            "schema_version": 1,
            "record_type": "reciprocal_v2_kc_v4_summary",
            "campaign_id": common.base.CAMPAIGN_ID,
            "cell_key": cell["cell_key"],
            "kc": kc,
            "implementation_sha256": cell["implementation_sha256"],
            # The frozen base validator names this field as a SHA of the file,
            # while robust_validation_sha256 below binds the recomputed result.
            "gate_spec_sha256": common.file_sha256(common.base.GATE_SPEC),
            "coverage_complete": True,
            "all_required_cases_pass": result["success"],
            "raw_path": common.repo_path(path),
            "raw_sha256": common.file_sha256(path),
            "robust_validation": result,
            "robust_validation_sha256": common.canonical_sha256(result),
        }
        validated_path = cell_summary_path(kc, cell["cell_key"])
        isolation.exclusive_json(validated_path, compatibility)
        cells.append(
            {
                "cell_key": cell["cell_key"],
                "implementation_sha256": cell["implementation_sha256"],
                "raw_path": common.repo_path(path),
                "raw_sha256": common.file_sha256(path),
                "v4_summary_path": common.repo_path(validated_path),
                "v4_summary_sha256": common.file_sha256(validated_path),
                "robust_validation_sha256": common.canonical_sha256(result),
                "all_required_cases_pass": result["success"],
            }
        )
    all_pass = len(cells) == 24 and all(row["all_required_cases_pass"] for row in cells)
    value = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_kc_attempt_summary",
        "campaign_id": common.base.CAMPAIGN_ID,
        "kc": kc,
        "request_path": common.repo_path(request_path(kc)),
        "request_sha256": common.file_sha256(request_path(kc)),
        "coverage_complete": True,
        "all_cells_all_v4_cases_pass": all_pass,
        "cells": cells,
    }
    isolation.exclusive_json(attempt_path(kc), value)
    return attempt_path(kc)


def execute(runner: Path, authorization: Path, physical_gpu: int) -> Path:
    validate_authorization(authorization.resolve(), runner.resolve())
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,name,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    require(completed.returncode == 0, "GPU preflight failed")
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    require(len(rows) == 1, "GPU preflight did not resolve one GPU")
    gpu_uuid, name, capability = [item.strip() for item in rows[0].split(",", 2)]
    auth = common.load_json(authorization.resolve())
    require(
        gpu_uuid in auth["authorized_gpu_uuids"]
        and name == common.base.TIMING_GPU_NAME
        and capability == "8.9",
        "GPU identity is not authorized",
    )
    request = make_request(runner, authorization)
    value = common.load_json(request)
    start = request.with_name(request.stem + ".started.json")
    isolation.exclusive_json(
        start,
        {
            "record_type": "reciprocal_v2_kc_external_action_start",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "request_sha256": common.file_sha256(request),
            "process_instance_id": str(uuid.uuid4()),
            "physical_gpu": physical_gpu,
            "gpu_uuid": gpu_uuid,
            "retry_authorized": False,
        },
    )
    run = subprocess.run(
        [str(runner.resolve())],
        input=json.dumps({**value, "physical_gpu": physical_gpu, "gpu_uuid": gpu_uuid}) + "\n",
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(physical_gpu)},
    )
    require(run.returncode == 0, "KC runner failed; start receipt requires reconciliation")
    return capture()


def freeze_resolution() -> Path:
    attempts = []
    selected = None
    literal = _literal(_registry())
    for kc in common.base.KC_LADDER:
        path = attempt_path(kc)
        require(path.is_file(), f"KC={kc}: attempt missing")
        value = common.load_json(path)
        validate_attempt(value, kc, verify_artifacts=True)
        rows = []
        for cell in value["cells"]:
            plan_cell = next(row for row in treatment_plan.kc_plan()["cells"] if row["cell_key"] == cell["cell_key"])
            rows.append(
                {
                    "cell_key": cell["cell_key"],
                    "implementation_sha256": literal[plan_cell["literal_cell_id"]]["sha256"],
                    "v4_summary_path": cell["v4_summary_path"],
                    "v4_summary_sha256": cell["v4_summary_sha256"],
                    "all_required_cases_pass": cell["all_required_cases_pass"],
                }
            )
        all_pass = value["all_cells_all_v4_cases_pass"]
        attempts.append({"kc": kc, "cells": rows, "all_cells_all_v4_cases_pass": all_pass})
        if all_pass:
            selected = kc
            break
    if selected is None:
        terminal = common.KC_OUTPUT_ROOT / "no_common_kc.json"
        isolation.exclusive_json(
            terminal,
            {
                "record_type": "reciprocal_v2_no_common_kc_receipt",
                "campaign_id": common.base.CAMPAIGN_ID,
                "ladder_exhausted": True,
                "attempt_sha256": {
                    str(kc): common.file_sha256(attempt_path(kc)) for kc in common.base.KC_LADDER
                },
                "launch_remains_blocked": True,
            },
        )
        raise KCLadderError(f"no common KC; wrote terminal blocked receipt {terminal}")
    resolution = {
        "schema_version": 1,
        "record_type": "reciprocal_v2_recipe_resolution_lock",
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "frozen",
        "kc_ladder": list(common.base.KC_LADDER),
        "selected_kc": selected,
        "all_cells_all_v4_cases_pass": True,
        "gate_lock_sha256": common.file_sha256(common.base.GATE_LOCK),
        "implementation_registry_sha256": common.file_sha256(common.IMPLEMENTATION_REGISTRY),
        "kc_plan_sha256": common.file_sha256(common.KC_PLAN),
        "attempts": attempts,
    }
    isolation.exclusive_json(common.RESOLUTION_LOCK, resolution)
    return common.RESOLUTION_LOCK


def status() -> dict[str, Any]:
    return {
        "next_kc": next_kc(),
        "requests": {str(kc): request_path(kc).is_file() for kc in common.base.KC_LADDER},
        "attempts": {str(kc): attempt_path(kc).is_file() for kc in common.base.KC_LADDER},
        "resolution_lock_present": common.RESOLUTION_LOCK.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    request = sub.add_parser("request")
    execute_parser = sub.add_parser("execute")
    sub.add_parser("capture")
    sub.add_parser("freeze")
    for item in (request, execute_parser):
        item.add_argument("--runner", type=Path, required=True)
        item.add_argument("--authorization", type=Path, default=common.KC_EXECUTION_AUTHORIZATION)
    execute_parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.command == "request":
        output = make_request(args.runner, args.authorization)
    elif args.command == "execute":
        output = execute(args.runner, args.authorization, args.physical_gpu)
    elif args.command == "capture":
        output = capture()
    else:
        output = freeze_resolution()
    print(json.dumps({"path": common.repo_path(output), "sha256": common.file_sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
