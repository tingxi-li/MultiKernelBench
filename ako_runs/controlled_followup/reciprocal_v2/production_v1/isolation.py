#!/usr/bin/env python3
"""Validate boundary-wrapper transcripts and freeze translator isolation."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # Support both ``python file.py`` and ``python -m package.module``.
    from . import common
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    import common


class IsolationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IsolationError(message)


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"unreconciled temporary file: {temporary}")
    temporary.write_bytes(common.stable_json_bytes(value))
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _supplement_freeze_hash() -> str:
    require(common.SOURCE_FREEZE.is_file(), "supplement source freeze is missing")
    return common.file_sha256(common.SOURCE_FREEZE)


def expected_probe_outcomes() -> dict[str, bool]:
    return {
        "repository_root_accessible": False,
        "other_translator_source_accessible": False,
        "performance_results_accessible": False,
        "network_accessible": False,
        "own_source_root_writable": True,
    }


def derive_probe_outcomes(probes: Any, translator: str = "transcript") -> dict[str, bool]:
    """Derive access from raw exit codes; reject summaries and malformed censuses."""
    expected = expected_probe_outcomes()
    require(
        isinstance(probes, list) and len(probes) == len(expected),
        f"{translator}: probe census mismatch",
    )
    derived: dict[str, bool] = {}
    for row in probes:
        require(isinstance(row, dict), f"{translator}: malformed probe")
        name = row.get("probe")
        require(name in expected and name not in derived, f"{translator}: duplicate/unknown probe")
        command = row.get("command")
        exit_code = row.get("exit_code")
        require(
            isinstance(command, list)
            and command
            and all(isinstance(item, str) and item for item in command)
            and isinstance(exit_code, int)
            and not isinstance(exit_code, bool),
            f"{translator}: probe lacks argv/exit code",
        )
        derived[name] = exit_code == 0
    return derived


def validate_transcript(path: Path, translator: str) -> dict[str, Any]:
    expected_path = common.ISOLATION_TRANSCRIPT_ROOT / f"{translator}.json"
    require(path.resolve() == expected_path.resolve(), f"{translator}: wrong transcript path")
    require(path.is_file() and not path.is_symlink(), f"{translator}: transcript missing/symlinked")
    value = common.load_json(path)
    template = common.ISOLATION_TEMPLATE_ROOT / f"{translator}.json"
    require(
        value.get("schema_version") == 1
        and value.get("record_type") == "reciprocal_v2_boundary_wrapper_transcript"
        and value.get("supplement_id") == common.SUPPLEMENT_ID
        and value.get("campaign_id") == common.base.CAMPAIGN_ID
        and value.get("translator") == translator
        and value.get("state") == "boundary_run_complete",
        f"{translator}: transcript header mismatch",
    )
    require(
        value.get("legacy_source_freeze_sha256")
        == common.EXPECTED_LEGACY["source_freeze_sha256"]
        and value.get("supplement_source_freeze_sha256") == _supplement_freeze_hash(),
        f"{translator}: source-freeze binding mismatch",
    )
    require(
        value.get("isolation_request_path") == common.repo_path(template)
        and value.get("isolation_request_sha256") == common.file_sha256(template),
        f"{translator}: isolation request binding mismatch",
    )
    require(
        value.get("boundary_kind") in {"container", "mount_namespace", "distinct_os_user"},
        f"{translator}: unsupported boundary kind",
    )
    for field in ("execution_id", "workspace_id", "orchestrator_identity", "boundary_runner_path"):
        require(isinstance(value.get(field), str) and value[field], f"{translator}: {field} missing")
    require(
        value.get("worktree_id") == value["workspace_id"],
        f"{translator}: frozen-validator worktree alias mismatch",
    )
    runner_path = common.REPO_ROOT / value["boundary_runner_path"]
    runner = runner_path.resolve()
    require(
        runner_path.is_file()
        and not runner_path.is_symlink()
        and runner.is_relative_to(common.REPO_ROOT.resolve())
        and value.get("boundary_runner_sha256") == common.file_sha256(runner),
        f"{translator}: boundary runner is not content-addressed",
    )
    timestamps = []
    for field in ("started_utc", "completed_utc"):
        try:
            parsed = datetime.fromisoformat(
                str(value.get(field, "")).replace("Z", "+00:00")
            )
            require(parsed.tzinfo is not None, f"{translator}: {field} lacks timezone")
            timestamps.append(parsed)
        except ValueError as exc:
            raise IsolationError(f"{translator}: {field} invalid") from exc
    require(timestamps[0] <= timestamps[1], f"{translator}: transcript time order invalid")
    require(value.get("network_namespace_disabled") is True, f"{translator}: network not disabled")
    visible = value.get("visible_roots")
    require(
        isinstance(visible, list)
        and visible
        and all(isinstance(root, str) and root for root in visible),
        f"{translator}: visible-root record missing/malformed",
    )
    forbidden_fragments = (
        "reciprocal_v2/results",
        f"translators/{next(x for x in common.base.TRANSLATORS if x != translator)}",
    )
    require(
        not any(fragment in root for root in visible for fragment in forbidden_fragments),
        f"{translator}: forbidden root appears in mount allowlist",
    )
    expected = expected_probe_outcomes()
    # Accessibility is derived from the wrapper's raw exit code; caller
    # supplied success booleans are checked for compatibility but not trusted.
    derived = derive_probe_outcomes(value.get("probe_observations"), translator)
    require(derived == expected, f"{translator}: boundary probes fail contract: {derived}")
    # These compatibility fields are checked against, but never trusted instead
    # of, the raw exit-code derivation above.  The frozen campaign validator
    # requires them in the transcript it dereferences.
    require(
        value.get("other_translator_source_accessible")
        is derived["other_translator_source_accessible"]
        and value.get("performance_results_accessible")
        is derived["performance_results_accessible"],
        f"{translator}: compatibility access flags are not probe-derived",
    )
    outputs = value.get("produced_sources")
    require(isinstance(outputs, list) and len(outputs) == 24, f"{translator}: output census is not 24")
    seen = set()
    manifest = common.load_json(common.BASE / "manifests/audit.json")
    expected_outputs = {
        (
            common.BASE
            / "translators"
            / translator
            / "implementations"
            / f"{job['cell_id']}.py"
        ).resolve()
        for job in manifest["jobs"]
        if job["translator"] == translator
    }
    for row in outputs:
        require(isinstance(row, dict), f"{translator}: malformed output binding")
        path_value, digest = row.get("path"), row.get("sha256")
        require(isinstance(path_value, str) and isinstance(digest, str), f"{translator}: output path/hash missing")
        unresolved = common.REPO_ROOT / path_value
        source = unresolved.resolve()
        root = (common.BASE / "translators" / translator / "implementations").resolve()
        require(
            unresolved.is_file() and not unresolved.is_symlink(),
            f"{translator}: produced source missing/symlinked",
        )
        require(source.is_relative_to(root) and source.stat().st_size > 0, f"{translator}: produced source escapes/is empty")
        require(common.file_sha256(source) == digest, f"{translator}: produced source hash mismatch")
        require(source not in seen, f"{translator}: duplicate produced source")
        seen.add(source)
    require(seen == expected_outputs, f"{translator}: produced source paths are not the exact 24-cell census")
    return value


def lock_document() -> dict[str, Any]:
    rows = []
    execution_ids, workspace_ids = set(), set()
    for translator in common.base.TRANSLATORS:
        path = common.ISOLATION_TRANSCRIPT_ROOT / f"{translator}.json"
        transcript = validate_transcript(path, translator)
        require(transcript["execution_id"] not in execution_ids, "execution identity reused")
        require(transcript["workspace_id"] not in workspace_ids, "workspace identity reused")
        execution_ids.add(transcript["execution_id"])
        workspace_ids.add(transcript["workspace_id"])
        rows.append(
            {
                "translator": translator,
                "source_root": f"ako_runs/controlled_followup/reciprocal_v2/translators/{translator}",
                "worktree_id": transcript["workspace_id"],
                "execution_id": transcript["execution_id"],
                "other_translator_source_accessible": False,
                "performance_results_accessible": False,
                "isolation_transcript_path": common.repo_path(path),
                "isolation_transcript_sha256": common.file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "record_type": "reciprocal_v2_translator_isolation_lock",
        "campaign_id": common.base.CAMPAIGN_ID,
        "state": "frozen",
        "source_freeze_sha256": common.EXPECTED_LEGACY["source_freeze_sha256"],
        "supplement_source_freeze_sha256": _supplement_freeze_hash(),
        "translators": rows,
        "claim_limit": "content_addressed_boundary_wrapper_observation_not_external_attestation",
    }


def freeze() -> Path:
    exclusive_json(common.TRANSLATOR_ISOLATION_LOCK, lock_document())
    return common.TRANSLATOR_ISOLATION_LOCK


def status() -> dict[str, Any]:
    missing = [
        common.repo_path(common.ISOLATION_TRANSCRIPT_ROOT / f"{name}.json")
        for name in common.base.TRANSLATORS
        if not (common.ISOLATION_TRANSCRIPT_ROOT / f"{name}.json").is_file()
    ]
    return {
        "state": "transcripts_missing" if missing else "transcripts_present_not_yet_frozen",
        "missing_transcripts": missing,
        "isolation_lock_present": common.TRANSLATOR_ISOLATION_LOCK.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("freeze")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "status":
        result = status()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2 if result["missing_transcripts"] else 0
    if args.command == "freeze":
        output = freeze()
    else:
        observed = common.load_json(common.TRANSLATOR_ISOLATION_LOCK)
        require(observed == lock_document(), "isolation lock is stale")
        output = common.TRANSLATOR_ISOLATION_LOCK
    print(json.dumps({"path": common.repo_path(output), "sha256": common.file_sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
