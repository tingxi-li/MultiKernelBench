#!/usr/bin/env python3
"""Frozen paths, hashes, plans, and validation helpers for this campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CAMPAIGN_PATH = HERE / "campaign.json"
SOURCE_RECEIPT_PATH = HERE / "source_receipt.json"
JOBS_PATH = HERE / "jobs.json"
LOCK_PATH = HERE / "launch_lock.json"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

LOCAL_SOURCE_NAMES = (
    ".gitignore",
    "README.md",
    "__init__.py",
    "analyze.py",
    "campaign.json",
    "capture_evidence.py",
    "freeze.py",
    "launch.py",
    "protocol.py",
    "run_one.py",
    "test_protocol.py",
)


class CampaignError(RuntimeError):
    """Fail-closed campaign validation error."""


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("rb") as stream:
        return json.load(stream)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = stable_json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_campaign() -> dict[str, Any]:
    campaign = read_json(CAMPAIGN_PATH)
    subjects = campaign.get("subjects", [])
    subject_ids = [row.get("subject_id") for row in subjects]
    if len(subjects) != 4 or len(set(subject_ids)) != 4:
        raise CampaignError("campaign must freeze exactly four unique subjects")
    if campaign.get("performance_protocol", {}).get("blocks") != 15:
        raise CampaignError("campaign must freeze 15 blocks")
    if len(campaign.get("preregistered_contrasts", [])) != 2:
        raise CampaignError("campaign must freeze exactly two contrasts")
    return campaign


def subject_map(campaign: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    campaign = campaign or load_campaign()
    return {row["subject_id"]: row for row in campaign["subjects"]}


def source_hashes(campaign: dict[str, Any] | None = None) -> dict[str, str]:
    campaign = campaign or load_campaign()
    paths = [HERE / name for name in LOCAL_SOURCE_NAMES]
    paths.extend(REPO_ROOT / row["source_path"] for row in campaign["subjects"])
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise CampaignError(f"missing frozen source: {path}")
        relative = str(path.relative_to(REPO_ROOT))
        if relative in result:
            raise CampaignError(f"duplicate frozen source: {relative}")
        result[relative] = sha256_file(path)
    for subject in campaign["subjects"]:
        observed = result[subject["source_path"]]
        if observed != subject["expected_source_sha256"]:
            raise CampaignError(
                f"historical source changed for {subject['subject_id']}: {observed}"
            )
    return dict(sorted(result.items()))


def make_plan(campaign: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    campaign = campaign or load_campaign()
    ids = [row["subject_id"] for row in campaign["subjects"]]
    rng = random.Random(campaign["performance_protocol"]["order_seed"])
    plan: list[dict[str, Any]] = []
    for block in range(campaign["performance_protocol"]["blocks"]):
        order = ids.copy()
        rng.shuffle(order)
        for position, subject_id in enumerate(order):
            plan.append(
                {
                    "job_id": f"block{block:02d}.{subject_id}",
                    "block": block,
                    "position": position,
                    "subject_id": subject_id,
                }
            )
    return plan


def expected_jobs(campaign: dict[str, Any] | None = None) -> dict[str, Any]:
    campaign = campaign or load_campaign()
    plan = make_plan(campaign)
    return {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_jobs",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": canonical_sha256(campaign),
        "plan": plan,
        "expected_records": len(plan),
    }


def expected_receipt(campaign: dict[str, Any] | None = None) -> dict[str, Any]:
    campaign = campaign or load_campaign()
    hashes = source_hashes(campaign)
    targets = {
        row["subject_id"]: {
            "path": row["source_path"],
            "sha256": hashes[row["source_path"]],
        }
        for row in campaign["subjects"]
    }
    return {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_source_receipt",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": canonical_sha256(campaign),
        "source_sha256": hashes,
        "target_artifacts": targets,
        "plan_canonical_sha256": canonical_sha256(expected_jobs(campaign)),
        "freeze_policy": "pre-launch byte receipt; any source change requires a new versioned campaign and result namespace",
    }


def expected_lock(campaign: dict[str, Any] | None = None) -> dict[str, Any]:
    campaign = campaign or load_campaign()
    receipt = expected_receipt(campaign)
    jobs = expected_jobs(campaign)
    return {
        "schema_version": 1,
        "record_type": "archived_current_fused_v1_launch_lock",
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": canonical_sha256(campaign),
        "source_receipt_canonical_sha256": canonical_sha256(receipt),
        "jobs_canonical_sha256": canonical_sha256(jobs),
        "source_receipt_file_sha256": hashlib.sha256(stable_json_bytes(receipt)).hexdigest(),
        "jobs_file_sha256": hashlib.sha256(stable_json_bytes(jobs)).hexdigest(),
    }


def _verify_stable(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"missing frozen artifact: {path}")
    observed = read_json(path)
    if observed != expected:
        raise CampaignError(f"frozen artifact differs from current inputs: {path}")
    if path.read_bytes() != stable_json_bytes(observed):
        raise CampaignError(f"frozen artifact is not stable JSON: {path}")
    return observed


def verify_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign = load_campaign()
    receipt = _verify_stable(SOURCE_RECEIPT_PATH, expected_receipt(campaign))
    jobs = _verify_stable(JOBS_PATH, expected_jobs(campaign))
    lock = _verify_stable(LOCK_PATH, expected_lock(campaign))
    if sha256_file(SOURCE_RECEIPT_PATH) != lock["source_receipt_file_sha256"]:
        raise CampaignError("source receipt file hash differs from launch lock")
    if sha256_file(JOBS_PATH) != lock["jobs_file_sha256"]:
        raise CampaignError("jobs file hash differs from launch lock")
    return campaign, receipt, jobs, lock


def validate_tag(tag: str) -> str:
    if not TAG_RE.fullmatch(tag):
        raise CampaignError(f"unsafe result tag: {tag!r}")
    return tag


def exact_median_interval(values: list[float]) -> dict[str, Any]:
    if len(values) != 15:
        raise CampaignError("registered median interval requires exactly 15 observations")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise CampaignError("median interval observations must be finite")
    ordered = sorted(float(value) for value in values)
    return {
        "n": 15,
        "median": ordered[7],
        "lo": ordered[3],
        "hi": ordered[11],
        "order_statistics": [4, 12],
        "achieved_coverage": 0.96484375,
        "sorted_observations": ordered,
    }


def exact_two_sided_sign_p(values: list[float], null: float = 1.0) -> dict[str, Any]:
    above = sum(value > null for value in values)
    below = sum(value < null for value in values)
    ties = len(values) - above - below
    n = above + below
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(above, below) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"above": above, "below": below, "ties": ties, "n_nonties": n, "p_raw": p}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def raw_relative(job: dict[str, Any]) -> Path:
    return Path("raw") / f"block{job['block']:02d}" / f"{job['subject_id']}.json"

