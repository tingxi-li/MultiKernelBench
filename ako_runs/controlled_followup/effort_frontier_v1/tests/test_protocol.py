from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from ako_runs.controlled_followup.effort_frontier_v1 import (  # noqa: E402
    analyze,
    campaign,
    capture_evidence,
    confirmation,
    controller,
    events,
    make_manifest,
    validate,
)


USAGE = {
    "input_tokens": 10,
    "output_tokens": 4,
    "reasoning_tokens": 1,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "total_tokens": 14,
    "raw_categories": {"input_tokens": 10, "output_tokens": 4},
}


def clock(effort: float, **kwargs):
    value = events.clock_payload(effort)
    value.update(kwargs)
    return value


class StaticTests(unittest.TestCase):
    def test_manifest_is_current_and_balanced(self):
        manifest = validate.validate_static()
        self.assertEqual(manifest, make_manifest.build())
        self.assertEqual(len(manifest["trajectories"]), 20)
        self.assertEqual(
            {gpu: sum(row["physical_gpu"] == gpu for row in manifest["trajectories"])
             for gpu in range(4)},
            {0: 5, 1: 5, 2: 5, 3: 5},
        )

    def test_unresolved_model_is_launch_blocker(self):
        blockers = []
        self.assertIsNone(validate.model_resolution(blockers))
        self.assertTrue(any("not resolved" in blocker for blocker in blockers))

    def test_registry_rejects_path_escape_and_missing_control(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            lanes = {
                lane: {
                    "tuning_command": ["runner"],
                    "terminal_holdout_command": ["runner"],
                    "confirmation_command": ["runner"],
                    "timeout_s": 1,
                    "source_hashes": {"../escape": "0" * 64},
                }
                for lane in campaign.PROGRAMMABLE_LANES
            }
            path.write_text(json.dumps({
                "schema_version": 1,
                "campaign_id": campaign.CAMPAIGN_ID,
                "lanes": lanes,
            }))
            blockers = []
            with mock.patch.object(campaign, "EXECUTOR_REGISTRY", path):
                validate.executor_registry(blockers)
            self.assertTrue(any("escapes repository" in item for item in blockers))
            self.assertTrue(any("control" in item for item in blockers))

    def test_prereg_evidence_is_deterministic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = capture_evidence.build("prereg", "first", output)
            second = capture_evidence.build("prereg", "second", output)
            first_index = campaign.load_json(first)
            second_index = campaign.load_json(second)
            self.assertEqual(first_index["bundle_sha256"], second_index["bundle_sha256"])
            verified = capture_evidence.verify(first)
            self.assertTrue(verified["ok"])
            paths = [row["path"] for row in first_index["manifest"]["entries"]]
            self.assertIn(
                "ako_runs/controlled_followup/effort_frontier_v1/capture_evidence.py",
                paths,
            )
            self.assertFalse(any("__pycache__" in path or path.endswith(".pyc") for path in paths))
        self.assertTrue(capture_evidence._excluded(campaign.HERE / "results/x/build/a.o"))
        self.assertTrue(capture_evidence._excluded(campaign.HERE / "results/x/.launcher.lock"))
        self.assertTrue(capture_evidence._excluded(campaign.HERE / "results/x/a.partial.12"))
        self.assertTrue(capture_evidence._excluded(campaign.HERE / "results/x/evidence/old.tar.gz"))
        self.assertFalse(capture_evidence._excluded(campaign.MODEL_LOCK))


class EventTests(unittest.TestCase):
    def test_chain_envelope_and_effort_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            events.append_event(
                path,
                "effort_v1.cublaslt.r0",
                clock(0.0, event_type="trajectory_start"),
            )
            with self.assertRaisesRegex(ValueError, "override event envelope"):
                events.append_event(
                    path,
                    "effort_v1.cublaslt.r0",
                    {**clock(0.0), "event_type": "trajectory_complete", "campaign_id": "bad"},
                )
            with self.assertRaisesRegex(ValueError, "effort delta"):
                events.append_event(
                    path,
                    "effort_v1.cublaslt.r0",
                    clock(1.0, event_type="trajectory_complete"),
                )

    def test_exact_tests(self):
        result = analyze.exact_mann_whitney([1.0] * 5, [2.0] * 5)
        self.assertEqual(result["permutations"], 252)
        self.assertAlmostEqual(result["p_value_two_sided_exact"], 2 / 252)
        self.assertEqual(analyze._holm([0.01, 0.03, 0.2]), [0.03, 0.06, 0.2])
        sign = analyze.exact_sign_test([0.9] * 15)
        self.assertAlmostEqual(sign["p_value_two_sided_exact"], 2 / 2**15)

    def test_controller_evaluator_contract_is_fail_closed(self):
        job = campaign.trajectories()[0]
        base = {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "trajectory_id": job["trajectory_id"],
            "lane": job["lane"],
            "evaluation_split": "tuning",
            "candidate_sha256": "a" * 64,
            "gate_spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
            "physical_gpu": job["physical_gpu"],
            "logical_device": "cuda:0",
            "gpu_uuid": job["required_gpu_uuid"],
            "build_ok": True,
            "lane_policy_pass": True,
            "gate_pass": True,
            "median_ms": 1.25,
        }
        with self.assertRaisesRegex(ValueError, "gate-summary"):
            controller._normalize_result(
                base,
                split="tuning",
                candidate_sha256="a" * 64,
                lane=job["lane"],
                trajectory=job,
            )
        normalized = controller._normalize_result(
            {**base, "gate_summary_sha256": "b" * 64},
            split="tuning",
            candidate_sha256="a" * 64,
            lane=job["lane"],
            trajectory=job,
        )
        self.assertTrue(normalized["eligible"])
        with self.assertRaisesRegex(ValueError, "exactly"):
            controller._parse_proposal('{"source":"x","rationale":"y","extra":1}')

    def test_confirmation_requires_evidence_hash(self):
        item = {
            "record_id": "positive_rand_seed0.b00.p000",
            "treatment_id": "effort_v1.cublaslt.r0.0p5h",
            "treatment_type": "candidate",
            "distribution": "positive_rand_seed0",
            "block": 0,
            "position": 0,
            "candidate_sha256": "a" * 64,
        }
        raw = {
            "schema_version": 1,
            "campaign_id": campaign.CAMPAIGN_ID,
            "record_id": item["record_id"],
            "treatment_id": item["treatment_id"],
            "distribution": item["distribution"],
            "block": 0,
            "position": 0,
            "physical_gpu": 0,
            "logical_device": "cuda:0",
            "gpu_uuid": campaign.GPU_UUIDS[0],
            "candidate_sha256": "a" * 64,
            "lane_policy_pass": True,
            "gate_spec_sha256": campaign.file_sha256(campaign.GATE_SPEC),
            "gate_pass": True,
            "trial_times_ms": [1.0] * campaign.CONFIRM_TRIALS,
        }
        completed = mock.Mock(returncode=0, stdout=json.dumps(raw), stderr="")
        with mock.patch.object(confirmation.subprocess, "run", return_value=completed):
            result = confirmation._execute(["runner"], 1, {}, item)
        self.assertFalse(result["ok"])
        self.assertIn("evidence", result["error"])


class EndToEndAnalysisTests(unittest.TestCase):
    def _write_search(self, root: Path) -> None:
        manifest_sha = campaign.file_sha256(campaign.MANIFEST)
        global_bindings = {
            "manifest_sha256": manifest_sha,
            "gate_spec_sha256": "1" * 64,
            "gate_receipt_sha256": "2" * 64,
            "model_resolution_lock_sha256": "3" * 64,
            "executor_registry_sha256": "4" * 64,
            "prelaunch_provenance_sha256": "5" * 64,
            "immutable_model_revision": "gpt-5.6-sol-immutable-test-revision",
        }
        for job in campaign.trajectories():
            identifier = job["trajectory_id"]
            trajectory_root = root / identifier
            event_path = trajectory_root / "events.jsonl"
            candidate = trajectory_root / "candidates/candidate_00000.txt"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(f"source for {identifier}")
            digest = campaign.file_sha256(candidate)
            relative = "candidates/candidate_00000.txt"
            events.append_event(
                event_path,
                identifier,
                clock(
                    0.0,
                    event_type="trajectory_start",
                    lane=job["lane"],
                    replicate=job["replicate"],
                    search_seed=job["search_seed"],
                    physical_gpu=job["physical_gpu"],
                    required_gpu_uuid=job["required_gpu_uuid"],
                    bindings={
                        **global_bindings,
                        "executor_source_hashes": {f"executor/{job['lane']}.py": "6" * 64},
                    },
                    adapter_audit={
                        "provider": "openai",
                        "credential_source": "environment",
                        "sampling_seed_supported": False,
                    },
                ),
            )
            events.append_event(
                event_path,
                identifier,
                clock(
                    0.0,
                    event_type="provider_request",
                    iteration=0,
                    request_id="r0",
                    feedback_candidate_sha256=None,
                    feedback_tuning_median_ms=None,
                    terminal_feedback_included=False,
                    prompt_sha256="9" * 64,
                ),
            )
            events.append_event(
                event_path,
                identifier,
                {
                    **events.clock_payload(28800.0, provider=28799.0, controller=1.0),
                    "event_type": "provider_response",
                    "iteration": 0,
                    "request_id": "r0",
                    "resolved_model_revision": global_bindings["immutable_model_revision"],
                    "usage": USAGE,
                    "parse_error": None,
                    "candidate_relative_path": relative,
                    "candidate_sha256": digest,
                },
            )
            events.append_event(
                event_path,
                identifier,
                clock(
                    28800.0,
                    event_type="tuning_evaluation_start",
                    iteration=0,
                    candidate_relative_path=relative,
                    candidate_sha256=digest,
                ),
            )
            latency = 1.0 + campaign.PROGRAMMABLE_LANES.index(job["lane"]) + job["replicate"] / 100
            events.append_event(
                event_path,
                identifier,
                {
                    **events.clock_payload(28800.1, gpu=0.1),
                    "event_type": "tuning_evaluation",
                    "iteration": 0,
                    "candidate_relative_path": relative,
                    "candidate_sha256": digest,
                    "build_ok": True,
                    "lane_policy_pass": True,
                    "gate_pass": True,
                    "eligible": True,
                    "median_ms": latency,
                    "gate_summary_sha256": "a" * 64,
                    "gpu_uuid": job["required_gpu_uuid"],
                },
            )
            effort = 28800.1
            for label, target in zip(campaign.CHECKPOINT_LABELS, campaign.CHECKPOINTS_S):
                events.append_event(
                    event_path,
                    identifier,
                    clock(
                        effort,
                        event_type="terminal_evaluation_start",
                        iteration=0,
                        checkpoint_label=label,
                        candidate_relative_path=relative,
                        candidate_sha256=digest,
                    ),
                )
                events.append_event(
                    event_path,
                    identifier,
                    clock(
                        effort,
                        event_type="checkpoint_freeze",
                        iteration=0,
                        checkpoint_label=label,
                        checkpoint_target_active_effort_s=target,
                        checkpoint_overshoot_s=effort - target,
                        candidate_relative_path=relative,
                        candidate_sha256=digest,
                        tuning_median_ms=latency,
                        terminal_holdout={
                            "build_ok": True,
                            "lane_policy_pass": True,
                            "gate_pass": True,
                            "eligible": True,
                            "median_ms": latency,
                            "gate_summary_sha256": "b" * 64,
                            "gpu_uuid": job["required_gpu_uuid"],
                        },
                        selection_eligible=True,
                        terminal_feedback_exposed_to_future_search=False,
                    ),
                )
            events.append_event(
                event_path,
                identifier,
                clock(effort, event_type="trajectory_complete", checkpoint_count=3),
            )

    def test_search_plan_and_confirmation_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "search"
            self._write_search(root)
            lock_patches = (
                mock.patch.object(validate, "model_resolution", return_value=object()),
                mock.patch.object(validate, "executor_registry", return_value={}),
                mock.patch.object(validate, "provenance_blockers", return_value=None),
            )
            for patcher in lock_patches:
                patcher.start()
                self.addCleanup(patcher.stop)
            search, selections = analyze.validate_search(root)
            self.assertEqual(search["checkpoint_observation_count"], 60)
            self.assertEqual(len(selections), 60)
            self.assertTrue(all(row["permutations"] == 252 for row in search["pairwise_lane_tests"]))
            plan = analyze.build_confirmation_plan(root)
            self.assertEqual(len(plan["records"]), 1830)
            for distribution in campaign.TIMING_DISTRIBUTIONS:
                for block in range(15):
                    subset = [
                        row for row in plan["records"]
                        if row["distribution"] == distribution and row["block"] == block
                    ]
                    self.assertEqual(sum(row["treatment_type"] == "control" for row in subset), 1)
                    self.assertEqual(sorted(row["position"] for row in subset), list(range(61)))

            rows = []
            previous = events.GENESIS
            plan_sha = campaign.canonical_sha256(plan)
            lane_scale = {lane: index + 1 for index, lane in enumerate(campaign.PROGRAMMABLE_LANES)}
            for item in plan["records"]:
                if item["treatment_type"] == "control":
                    latency = 10.0
                    extra = {"gate_pass": None, "contract_pass": True}
                else:
                    distribution_scale = 1.0 if item["distribution"] == campaign.TIMING_DISTRIBUTIONS[0] else 1.1
                    latency = lane_scale[item["lane"]] * distribution_scale
                    extra = {"gate_pass": True, "contract_pass": None}
                row = {
                    "schema_version": 1,
                    "campaign_id": campaign.CAMPAIGN_ID,
                    "confirmation_plan_sha256": plan_sha,
                    "previous_record_sha256": previous,
                    "completed_at_utc": "2026-07-31T12:00:00Z",
                    **item,
                    "ok": True,
                    "physical_gpu": 0,
                    "logical_device": "cuda:0",
                    "gpu_uuid": campaign.GPU_UUIDS[0],
                    "trial_times_ms": [latency] * campaign.CONFIRM_TRIALS,
                    "median_ms": latency,
                    **extra,
                    "lane_policy_pass": (
                        True if item["treatment_type"] == "candidate" else None
                    ),
                    "gate_summary_sha256": (
                        "c" * 64 if item["treatment_type"] == "candidate" else None
                    ),
                    "contract_summary_sha256": (
                        "d" * 64 if item["treatment_type"] == "control" else None
                    ),
                }
                row["record_sha256"] = campaign.canonical_sha256(row)
                previous = row["record_sha256"]
                rows.append(row)
            record_path = Path(directory) / "records.jsonl"
            record_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            checked = analyze._validate_confirmation_records(plan, record_path)
            confirmation = analyze.analyze_confirmation(search, plan, checked)
            self.assertTrue(confirmation["complete"])
            self.assertEqual(len(confirmation["selection_distribution_summaries"]), 120)
            self.assertEqual(
                [row["spearman_rho"] for row in confirmation["distribution_rank_stability"]],
                [1.0, 1.0, 1.0],
            )

    def test_analysis_refuses_unresolved_prelaunch_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "prelaunch locks fail"):
                analyze.validate_search(Path(directory))


if __name__ == "__main__":
    unittest.main()
