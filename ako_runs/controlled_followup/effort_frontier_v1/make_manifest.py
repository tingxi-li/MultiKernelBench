#!/usr/bin/env python3
"""Generate the deterministic RQ5 effort-frontier manifest."""
from __future__ import annotations

import argparse

try:
    from . import campaign
except ImportError:  # direct script execution
    import campaign  # type: ignore


def build() -> dict:
    jobs = campaign.trajectories()
    return {
        "schema_version": 1,
        "campaign_id": campaign.CAMPAIGN_ID,
        "status": "preregistered_not_launched",
        "research_question": (
            "How does best fused-v2-legal latency evolve with cumulative active "
            "optimization effort across abstraction levels?"
        ),
        "scope": "implementation-specific, one fused operation/shape, RTX 6000 Ada only",
        "workload": {
            "operation": "matmul_bias_gelu_softmax",
            "shape": {"M": 1024, "K": 8192, "N": 8192},
            "output_dtype": "fp32",
            "timed_region": "precast_inputs_to_fp32_output",
        },
        "lanes": {
            "control": {
                "id": campaign.CONTROL_LANE,
                "kind": "frozen_nonprogrammable_timing_anchor",
                "search_effort": 0,
            },
            "programmable": list(campaign.PROGRAMMABLE_LANES),
            "cublaslt_scope": {
                "implementations_permitted": [
                    "cublaslt_matmul_with_GELU_BIAS_epilogue_if_exactly_gate_legal",
                    "cublaslt_matmul_plus_exact_nonvendor_postprocess",
                ],
                "selection": "fused-v2 gate in loop; unsupported/failing paths consume effort",
                "claim_limit": "tested cuBLASLt implementation, not vendor-expert performance",
            },
        },
        "optimizer": {
            "provider": campaign.MODEL["provider"],
            "requested_alias": campaign.MODEL["requested_alias"],
            "immutable_resolution_lock": "locks/model_resolution_lock.json",
            "same_controller_and_prompt_across_programmable_lanes": True,
            "sampling_seed_note": (
                "provider sampling seeds are not assumed supported; search_seed labels "
                "independent contexts and stochastic request IDs"
            ),
        },
        "effort_clock": {
            "checkpoints_s": list(campaign.CHECKPOINTS_S),
            "checkpoint_labels": list(campaign.CHECKPOINT_LABELS),
            "primary_clock": (
                "completed active controller + provider wait + evaluator compute seconds; "
                "queue and infrastructure outage time excluded"
            ),
            "controller_compute_definition": (
                "measured prompt/feedback construction, response parsing, candidate "
                "persistence, evaluator-response validation, and checkpoint selection; "
                "event serialization and lifecycle bookkeeping are protocol overhead"
            ),
            "separately_logged": [
                "provider_wait_s",
                "controller_compute_s",
                "gpu_evaluation_s",
                "human_intervention_s",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            ],
            "human_intervention_rule": (
                "zero by default; any nonzero intervention is timestamped, described, "
                "and cannot be silently converted to agent effort"
            ),
        },
        "robust_gate": {
            "spec": campaign.repo_path(campaign.GATE_SPEC),
            "spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
            "acceptance_receipt": campaign.repo_path(campaign.GATE_RECEIPT),
            "acceptance_receipt_sha256": campaign.file_sha256(campaign.GATE_RECEIPT),
            "candidate_rule": (
                "lane policy, build, and all registered fused-v2 cases pass or candidate "
                "is ineligible"
            ),
            "threshold_changes_permitted": False,
            "hidden_terminal_holdout": True,
        },
        "search_replicates_per_programmable_lane": campaign.SEARCH_REPLICATES,
        "trajectory_count": len(jobs),
        "trajectories": jobs,
        "confirmation": {
            "physical_gpu": 0,
            "required_gpu_uuid": campaign.GPU_UUIDS[0],
            "randomized_complete_blocks": campaign.CONFIRM_BLOCKS,
            "warmup_iterations_per_record": campaign.CONFIRM_WARMUP,
            "timed_trials_per_record": campaign.CONFIRM_TRIALS,
            "distributions": list(campaign.TIMING_DISTRIBUTIONS),
            "include_control_every_block": True,
            "candidates": (
                "one frozen selection per trajectory/checkpoint when the best tuning "
                "candidate passes its hidden terminal holdout; failed/missing selections "
                "remain failures and are not replaced after checkpoint freeze"
            ),
            "order": (
                "deterministic SHA-256 permutation of all eligible selections plus the "
                "control independently within each distribution and block"
            ),
            "analysis_unit": (
                "the independent search trajectory; timing blocks are paired technical "
                "replicates and are not promoted to independent search replicates"
            ),
        },
        "analysis": {
            "primary": "latency versus cumulative active effort frontier curves",
            "replicate_distribution": "five independent search contexts per programmable lane",
            "failure_ordering": (
                "a checkpoint without a terminal-holdout-legal candidate ranks worse than "
                "every finite legal latency; failures tie; success counts are always shown"
            ),
            "pairwise": (
                "exact two-sided permutation Mann-Whitney tests (tie-aware enumeration of "
                "all 10 choose 5 assignments) with Holm correction over six lane pairs "
                "separately at each checkpoint"
            ),
            "confirmation_normalization": (
                "candidate/control latency ratio paired within distribution and block, "
                "then reduced to one median technical-block ratio per search trajectory"
            ),
            "control_inference": (
                "exact two-sided sign tests across five trajectory summaries with Holm "
                "correction over four lanes per checkpoint and distribution"
            ),
            "distribution_stability": (
                "positive-versus-signed lane-rank Spearman agreement with an exact "
                "four-item permutation p-value (Holm across checkpoints), plus exact "
                "paired trajectory sign tests (Holm across lanes within checkpoint)"
            ),
            "no_endpoint_only_claim": True,
        },
        "required_prelaunch_locks": [
            "locks/model_resolution_lock.json",
            "locks/executor_registry.json",
            "locks/prelaunch_provenance.json",
        ],
        "launch_blocker_classes": {
            "treatment_artifact": {
                "item": "content-addressed external executor registry and implementations",
                "reasoning": campaign.repo_path(campaign.TREATMENT_ARTIFACT_NOTE),
                "reasoning_sha256": campaign.file_sha256(campaign.TREATMENT_ARTIFACT_NOTE),
                "not_mechanically_derivable_from_existing_fixed_runners": True,
            },
            "external": [
                "provider-attested immutable model resolution and credential",
                "four frozen Ada GPUs visible through a working NVIDIA driver",
            ],
            "coordination": "clean committed/pushed externally timestamped prelaunch provenance",
            "known_local_controller_gaps": [],
        },
        "provenance_contract": {
            "campaign_files": list(campaign.PROVENANCE_FILES),
            "external_executor_sources": "content-addressed by locks/executor_registry.json",
            "result_policy": "append-only hash-chained plain JSONL; no partial result is complete",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = campaign.stable_json_bytes(build())
    if args.check:
        if not campaign.MANIFEST.is_file() or campaign.MANIFEST.read_bytes() != data:
            raise SystemExit("manifest.json is stale")
    else:
        campaign.atomic_write(campaign.MANIFEST, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
