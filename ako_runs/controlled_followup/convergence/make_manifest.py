#!/usr/bin/env python3
"""Generate the preregistered replicated-search trajectory manifests.

This generator does not invoke an optimizer model.  It freezes the factorial,
budgets, independent seeds, and censoring rules that a model runner must obey.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
OPS = {
    "sum_reduction_over_a_dimension": 480,
    "standard_matrix_multiplication": 600,
    "scaled_dot_product_attention": 900,
}
DEFAULT_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra")
SAFETY_WALL_S = 7200
SAFETY_TOKENS = 100_000


def digest(*parts: object, size: int = 16) -> str:
    text = "\0".join(str(x) for x in parts).encode()
    return hashlib.sha256(text).hexdigest()[:size]


def seed(*parts: object) -> int:
    return int(digest("MKB-search-v1", *parts, size=16), 16) & ((1 << 63) - 1)


def trajectory(campaign: str, op: str, dsl: str, model: str,
               prompt_arm: str, replicate: int, reused_core: bool = False) -> dict:
    trajectory_id = digest(campaign, op, dsl, model, prompt_arm, replicate)
    return {
        "schema_version": 1,
        "campaign_id": campaign,
        "trajectory_id": trajectory_id,
        "status": "planned",
        "operation": op,
        "dsl": dsl,
        "model_snapshot": model,
        "model_revision_must_be_immutable": True,
        "prompt_arm": prompt_arm,
        "replicate": replicate,
        "reuses_neutral_core_trajectory": reused_core,
        "seeds": {
            "agent_sampling": seed(trajectory_id, "agent"),
            "candidate_inputs": seed(trajectory_id, "inputs"),
            "hidden_evaluation": seed(trajectory_id, "hidden"),
        },
        "budget": {
            "completed_compute_s": OPS[op],
            "safety_agent_wall_s": SAFETY_WALL_S,
            "safety_provider_tokens": SAFETY_TOKENS,
            "iteration_cap": None,
            "failed_builds_charged": True,
            "gate_failures_charged": True,
            "autotune_benchmarks_charged": True,
            "ncu_opportunities": 2,
            "ncu_gpu_s_logged_separately": True,
        },
        "stop_owner": "controller",
        "resource_cap_policy": "retain_as_right_censored; never selectively replace",
        "prior_artifacts_visible": False,
        "isolated_worktree_required": True,
        "randomization_key": digest("order", trajectory_id, size=32),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="20260729_controlled_followup_v1")
    ap.add_argument("--outdir", type=Path, default=Path(__file__).with_name("manifests"))
    ap.add_argument("--model", action="append", dest="models",
                    help="exact immutable model revision; provide exactly two")
    args = ap.parse_args()
    models = tuple(args.models or DEFAULT_MODELS)
    if len(models) != 2 or len(set(models)) != 2:
        ap.error("provide exactly two distinct model revisions")

    core = [
        trajectory(args.campaign, op, dsl, model, "neutral", rep)
        for op in OPS
        for dsl in DSLS
        for model in models
        for rep in range(5)
    ]
    # The prompt audit's neutral replications 0..2 are already present in core.
    # Only the two additional arms are emitted here: 4*2*2*3 = 48 runs.
    prompt = [
        trajectory(args.campaign, "standard_matrix_multiplication", dsl,
                   model, arm, rep)
        for dsl in DSLS
        for model in models
        for arm in ("valid_mechanism_hint", "misleading_prior")
        for rep in range(3)
    ]
    ids = [r["trajectory_id"] for r in core + prompt]
    assert len(core) == 120
    assert len(prompt) == 48
    assert len(ids) == len(set(ids))
    write_jsonl(args.outdir / "core_120.jsonl", core)
    write_jsonl(args.outdir / "prompt_extension_48.jsonl", prompt)
    summary = {
        "schema_version": 1,
        "campaign_id": args.campaign,
        "models": list(models),
        "core_trajectories": len(core),
        "prompt_extension_trajectories": len(prompt),
        "neutral_completed_compute_gpu_hours": sum(OPS[r["operation"]] for r in core) / 3600,
        "prompt_extension_completed_compute_gpu_hours": sum(OPS[r["operation"]] for r in prompt) / 3600,
        "launch_blockers": [
            "resolve each model alias to an immutable provider revision",
            "freeze and hash prompt templates and system/tool contracts",
            "freeze robust hidden gate before exposing any trajectory",
            "provide a runner that records provider token usage and controller compute clock",
        ],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
