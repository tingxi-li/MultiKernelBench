#!/usr/bin/env python3
"""Frozen constants and deterministic identities for convergence-v2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CAMPAIGN_ID = "20260731_controlled_followup_convergence_v2"
SCHEMA_VERSION = 2
DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
OPERATIONS = {
    "sum_reduction_over_a_dimension": 480,
    "standard_matrix_multiplication": 600,
    "scaled_dot_product_attention": 900,
}
MODELS = (
    {
        "provider": "openai",
        "requested_alias": "gpt-5.6-sol",
        "display_name": "GPT-5.6-sol",
    },
    {
        "provider": "anthropic",
        "requested_alias": "claude-opus-4.8",
        "display_name": "Claude Opus 4.8",
    },
)
PROMPT_EXTENSION_ARMS = ("valid_mechanism_hint", "misleading_prior")
REPLICATES = 8
GPU_SLOTS = 4
SAFETY_AGENT_WALL_S = 7200
SAFETY_PROVIDER_TOKENS = 100_000


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def digest(*parts: object, size: int = 32) -> str:
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()[:size]


def deterministic_seed(*parts: object) -> int:
    return int(digest("MKB-convergence-v2", *parts, size=16), 16) & ((1 << 63) - 1)


def model_key(model: dict[str, str]) -> str:
    return f"{model['provider']}:{model['requested_alias']}"


def gpu_slot_for_cell(cell_parts: Iterable[object], replicate: int) -> int:
    """Assign every eight-replicate cell exactly twice to every GPU slot."""
    if not 0 <= replicate < REPLICATES:
        raise ValueError(f"replicate must be in [0,{REPLICATES}): {replicate}")
    offset = int(digest("gpu-balance", *cell_parts, size=8), 16) % GPU_SLOTS
    return (offset + replicate) % GPU_SLOTS


def trajectory(
    operation: str,
    dsl: str,
    model: dict[str, str],
    prompt_arm: str,
    replicate: int,
    *,
    manifest_kind: str,
) -> dict[str, Any]:
    mkey = model_key(model)
    cell = (operation, dsl, mkey, prompt_arm)
    trajectory_id = digest(CAMPAIGN_ID, *cell, replicate)
    gpu_slot = gpu_slot_for_cell(cell, replicate)
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "trajectory_id": trajectory_id,
        "manifest_kind": manifest_kind,
        "status": "planned_launch_blocked",
        "operation": operation,
        "dsl": dsl,
        "provider": model["provider"],
        "requested_model_alias": model["requested_alias"],
        "model_key": mkey,
        "immutable_model_resolution_required": True,
        "prompt_arm": prompt_arm,
        "replicate": replicate,
        "gpu_slot": gpu_slot,
        "gpu_uuid_placeholder": f"GPU_UUID_SLOT_{gpu_slot}",
        "seeds": {
            "provider_request": deterministic_seed(trajectory_id, "provider"),
            "candidate_inputs": deterministic_seed(trajectory_id, "inputs"),
            "hidden_tuning": deterministic_seed(trajectory_id, "tuning"),
            "terminal_holdout": deterministic_seed(trajectory_id, "terminal"),
        },
        "budget": {
            "completed_evaluation_s": OPERATIONS[operation],
            "safety_agent_wall_s": SAFETY_AGENT_WALL_S,
            "safety_provider_tokens": SAFETY_PROVIDER_TOKENS,
            "failed_builds_charged": True,
            "gate_failures_charged": True,
            "autotuning_charged": True,
            "timeouts_charged": True,
            "ncu_opportunities": 2,
            "ncu_s_logged_separately": True,
        },
        "gate_policy": {
            "tuning_result_visible_to_searcher": ["passed", "failed_metric_names"],
            "thresholds_visible_to_searcher": False,
            "hidden_inputs_visible_to_searcher": False,
            "terminal_holdout_access": "terminal_evaluator_only",
        },
        "resource_cap_policy": "right_censor_without_replacement",
        "isolated_worktree_required": True,
        "prior_artifacts_visible": False,
        "randomization_key": digest("order", trajectory_id, size=64),
    }


def build_manifests() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core = [
        trajectory(op, dsl, model, "neutral", replicate, manifest_kind="core")
        for op in OPERATIONS
        for dsl in DSLS
        for model in MODELS
        for replicate in range(REPLICATES)
    ]
    prompt_extension = [
        trajectory(
            "standard_matrix_multiplication",
            dsl,
            model,
            arm,
            replicate,
            manifest_kind="prompt_extension",
        )
        for dsl in DSLS
        for model in MODELS
        for arm in PROMPT_EXTENSION_ARMS
        for replicate in range(REPLICATES)
    ]
    validate_manifest_rows(core, prompt_extension)
    return core, prompt_extension


def validate_manifest_rows(core: list[dict[str, Any]], prompt: list[dict[str, Any]]) -> None:
    if len(core) != 192 or len(prompt) != 128:
        raise ValueError(f"wrong census: core={len(core)}, prompt={len(prompt)}")
    rows = core + prompt
    ids = [row["trajectory_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory_id")
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["operation"], row["dsl"], row["model_key"], row["prompt_arm"])
        cells.setdefault(key, []).append(row)
    for key, members in cells.items():
        reps = sorted(row["replicate"] for row in members)
        counts = {slot: 0 for slot in range(GPU_SLOTS)}
        for row in members:
            counts[row["gpu_slot"]] += 1
        if reps != list(range(REPLICATES)) or set(counts.values()) != {2}:
            raise ValueError(f"unbalanced cell {key}: reps={reps}, slots={counts}")

