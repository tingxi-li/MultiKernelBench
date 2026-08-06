from __future__ import annotations

import json
import os
import statistics
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from ako_runs.controlled_followup.robust_gate.seeds import tensor_seeds

from .analyze import (
    _arm_coordinates,
    _validate_gate_rows,
    admission_artifact_hashes,
    admission_summary,
    load_verified_admission,
    timing_artifact_hashes,
    timing_summary,
)
from .campaign_runner import (
    GPU_LOCK_ID,
    GPU_LOCK_ENV,
    GPU_LOCK_PATH,
    PHASE1,
    _legacy_timing_command,
    _profile_command,
    _validate_inherited_gpu_lock,
)
from .protocol import (
    CAMPAIGN_PATH,
    HERE,
    MATERIALS_PATH,
    GPU0_UUID,
    ProtocolError,
    artifact_identity,
    build_lock,
    canonical_sha256,
    file_sha256,
    live_toolchain,
    load_lock,
    make_timing_manifest,
    live_upstream_head,
    read_json,
    repo_path,
    validate_campaign,
    validate_materials,
    validate_timing_manifest,
)


GPU = {
    "physical_index": 0,
    "uuid": GPU0_UUID,
    "product_name": "NVIDIA RTX 6000 Ada Generation",
    "compute_capability": "8.9",
    "driver_version": "test",
    "nvcc_sha256": "1" * 64,
    "ncu_sha256": "2" * 64,
}


class ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = read_json(CAMPAIGN_PATH)
        cls.materials = read_json(MATERIALS_PATH)
        cls.toolchain = live_toolchain()

    def _lock(self, directory: Path) -> Path:
        value = build_lock(
            self.campaign,
            self.materials,
            GPU,
            git_commit="a" * 40,
            upstream_commit="a" * 40,
            toolchain=self.toolchain,
        )
        path = directory / "lock.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return path

    def _remote(self, lock: Path) -> dict:
        return {
            "head_commit": "b" * 40,
            "upstream_commit": "b" * 40,
            "lock_path": str(lock.resolve().relative_to(repo_path("."))),
            "lock_sha256": file_sha256(lock),
            "lock_git_blob": "c" * 40,
            "head_equals_configured_upstream": True,
            "live_upstream_commit": "b" * 40,
            "upstream_remote": "origin",
            "upstream_ref": "refs/heads/study/topic/with/slashes",
        }

    def _idle(self, phase: str, pid: int, when: float, active: bool = False) -> dict:
        return {
            "phase": phase,
            "checked_at_unix": when,
            "self_pid": pid,
            "compute_processes": ([{"pid": pid, "used_memory_mib": "64"}] if active else []),
            "stderr": "",
        }

    def _artifact(self, directory: Path, token: str) -> dict:
        directory.mkdir(parents=True)
        files = {}
        for label, suffix in (("cuda", ".cu"), ("ptx", ".ptx"), ("sass", ".sass")):
            path = directory / ("kernel" + suffix)
            path.write_bytes(f"{token}:{label}\n".encode())
            files[label] = {
                "path": str(path.resolve().relative_to(repo_path("."))),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        return {
            "complete": True,
            "files": files,
            "bundle_sha256": canonical_sha256(files),
            "identity_sha256": artifact_identity(files),
        }

    def _projected_admission(self, lock: Path, eligible: set[str]) -> dict:
        pairs = []
        for pair in self.campaign["pairs"]:
            included = pair["pair_id"] in eligible
            pairs.append({
                "pair_id": pair["pair_id"],
                "gate_available": pair["gate"]["available"],
                "timing_eligible": included,
                "classification": "runtime_estimand" if included else "excluded_fail_closed",
                "artifact_identity_sha256": {"high": "b" * 64, "low": "c" * 64} if included else {},
            })
        return {
            "campaign_id": self.campaign["campaign_id"],
            "campaign_lock_sha256": file_sha256(lock),
            "complete": True,
            "pairs": pairs,
        }

    def _valid_matmul_admission(self, directory: Path, lock: Path) -> tuple[Path, dict]:
        root = directory / "admission"
        root.mkdir()
        lock_value = read_json(lock)
        lock_sha = file_sha256(lock)
        remote = self._remote(lock)
        (root / "launch_receipt.json").write_text(json.dumps({
            "campaign_id": self.campaign["campaign_id"],
            "campaign_lock_sha256": lock_sha,
            "dependency_bundle_sha256": lock_value["dependency_bundle_sha256"],
            "toolchain": lock_value["toolchain"],
            "expected_pair_ids": [pair["pair_id"] for pair in self.campaign["pairs"]],
            "physical_gpu": 0,
            "gpu": GPU,
            "gpu_lock_id": GPU_LOCK_ID,
            "gpu_idle_preflight": self._idle("admission_stage_pre", 99, 0.0),
            "remote_authorization": remote,
        }))
        pair = next(value for value in self.campaign["pairs"] if value["family"] == "matmul")
        manifest = read_json(repo_path(self.materials["entries"]["matmul_v4_manifest"]["path"]))
        spec = read_json(repo_path(self.materials["entries"]["matmul_v4_gate"]["path"]))
        identities = {}
        for side, token in (("high", "high-executable"), ("low", "low-executable")):
            receipt_path = root / f"{pair['pair_id']}__{side}.json"
            generated = self._artifact(root / f"{pair['pair_id']}__{side}_generated", token)
            identity = generated["identity_sha256"]
            identities[side] = identity
            expected = _arm_coordinates(pair, side, lock_sha, lock_value, self.materials)
            receipt = {
                **expected,
                "toolchain": lock_value["toolchain"],
                "remote_authorization": remote,
                "created_utc": "2026-08-05T00:00:00+00:00",
                "implementation_sha256": identity,
                "implementation_source_sha256": self.materials["entries"]["tilelang_matmul_abstraction"]["sha256"],
                "generated": generated,
                "metadata": {"config": {}, "reported_artifacts": {}},
                "gpu_idle_preflight": self._idle("admission_child_pre", 100 + len(identities), 1.0),
                "gpu_idle_postflight": self._idle("admission_child_post", 100 + len(identities), 4.0, True),
                "parent_pid": 99,
                "process_pid": 100 + len(identities),
                "t_start": 2.0,
                "t_end": 5.0,
            }
            rows = []
            for case in manifest["operations"]["matmul"]["cases"]:
                for seed_index in range(manifest["split_counts"]["validation"]):
                    seeds = tensor_seeds(manifest, "matmul", case["id"], "validation", seed_index)
                    for gate_id in pair["gate"]["gate_ids"]:
                        gate = spec["gates"][f"matmul/{gate_id}"]
                        metrics = {name: 0.0 for name in gate["thresholds"]}
                        rows.append({
                            **expected,
                            "remote_authorization": remote,
                            "created_utc": receipt["created_utc"],
                            "implementation_sha256": identity,
                            "record_type": "tilelang_abstraction_v3_matmul_gate",
                            "op": "matmul",
                            "gate_id": gate_id,
                            "case_id": case["id"],
                            "seed_index": seed_index,
                            "tensor_seeds": seeds,
                            "metrics": metrics,
                            "threshold_failures": [],
                            "ok": True,
                            "gate_pass": True,
                        })
            gate_path = receipt_path.with_suffix(".gate.jsonl")
            gate_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            receipt.update({
                "gate_path": str(gate_path.resolve().relative_to(repo_path("."))),
                "gate_sha256": file_sha256(gate_path),
                "gate_summary": _validate_gate_rows(pair, side, receipt, rows, expected, self.materials),
                "terminal_outcome": "GATE_PASSED",
            })
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))

            profile_path = root / f"{pair['pair_id']}__{side}.profile.json"
            profile_generated = self._artifact(root / f"{pair['pair_id']}__{side}.profile_generated", token)
            raw_path = profile_path.with_suffix(".raw.json")
            raw_path.write_text(json.dumps({"records": [{
                "key": f"tilelang_abs.{pair[side]['variant']}.primary",
                "dsl": "tilelang_abs",
                "variant": pair[side]["variant"],
                "geom": "primary",
                "set": pair[side]["set"],
                "ok": True,
                "metrics": {
                    "hmma_inst": 1024.0, "grid": 512.0, "block": 256.0,
                    "regs": 64.0, "smem_static_B": 0.0, "smem_dyn_B": 32768.0,
                    "dram_rd_B": 1.0, "dram_wr_B": 1.0,
                },
            }]}))
            profile_path.write_text(json.dumps({
                "schema_version": 1,
                "campaign_id": self.campaign["campaign_id"],
                "campaign_lock_sha256": lock_sha,
                "dependency_bundle_sha256": lock_value["dependency_bundle_sha256"],
                "toolchain": lock_value["toolchain"],
                "pair_id": pair["pair_id"], "family": pair["family"], "side": side,
                "variant": pair[side]["variant"], "set": pair[side]["set"],
                "physical_gpu": 0, "gpu": GPU, "remote_authorization": remote,
                "command": _profile_command(pair, side, 0, raw_path),
                "returncode": 0, "ok": True,
                "raw_path": str(raw_path.resolve().relative_to(repo_path("."))),
                "raw_sha256": file_sha256(raw_path),
                "generated": profile_generated,
                "expected_artifact_identity_sha256": identity,
                "implementation_sha256": profile_generated["identity_sha256"],
                "artifact_identity_match": True,
                "artifact_error": None,
                "gpu_idle_preflight": self._idle("profile_child_pre", 200 + len(identities), 5.0),
                "gpu_idle_postflight": self._idle("profile_child_post", 200 + len(identities), 8.0),
                "gpu_idle_postflight_error": None,
                "parent_pid": 99,
                "process_pid": 200 + len(identities),
                "t_start": 6.0,
                "t_end": 9.0,
            }, sort_keys=True))

        fused = next(value for value in self.campaign["pairs"] if value["family"] == "fused_softmax")
        for side in ("high", "low"):
            (root / f"{fused['pair_id']}__{side}.json").write_text(json.dumps({
                "campaign_lock_sha256": lock_sha, "terminal_outcome": "GATE_FAILED",
            }))
        sdpa = next(value for value in self.campaign["pairs"] if value["family"] == "sdpa")
        (root / f"{sdpa['pair_id']}__unavailable.json").write_text(json.dumps({
            "campaign_id": self.campaign["campaign_id"], "campaign_lock_sha256": lock_sha,
            "pair_id": sdpa["pair_id"], "terminal_outcome": "CURRENT_GATE_UNAVAILABLE",
            "reason": sdpa["gate"]["reason"], "build_attempted": False, "timing_authorized": False,
        }))
        hashes = admission_artifact_hashes(root, self.campaign)
        (root / "run_status.json").write_text(json.dumps({
            "schema_version": 1,
            "record_type": "tilelang_abstraction_v3_admission_run_status",
            "campaign_id": self.campaign["campaign_id"],
            "campaign_lock_sha256": lock_sha,
            "complete": True,
            "artifact_sha256": hashes,
            "artifact_bundle_sha256": canonical_sha256(hashes),
            "gpu_idle_postflight": self._idle("admission_stage_post", 99, 10.0),
            "launch_receipt_sha256": file_sha256(root / "launch_receipt.json"),
        }, sort_keys=True))
        summary = admission_summary(root, lock)
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        self.assertEqual(summary["timing_eligible_pair_ids"], [pair["pair_id"]])
        self.assertEqual(summary["pairs"][0]["artifact_identity_sha256"], identities)
        return summary_path, summary

    def test_registry_rehash_and_two_step_lock_policy(self) -> None:
        validate_campaign(self.campaign)
        validate_materials(self.materials, self.campaign)
        broken = deepcopy(self.materials)
        broken["entries"]["matmul_v4_gate"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "material changed"):
            validate_materials(broken, self.campaign)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            lock = read_json(self._lock(Path(temporary)))
        self.assertEqual(lock["state"], "frozen_pending_remote_registration")
        self.assertEqual(lock["toolchain"], self.toolchain)
        self.assertTrue(lock["remote_preregistration"]["lock_must_be_committed_and_pushed_before_launch"])

    def test_live_upstream_slash_ref_and_direct_child_lock_guard(self) -> None:
        from ako_runs.controlled_followup.finite_frontier_ada_v1.launch import GPU_LOCK_PATH as Q1_GPU_LOCK_PATH

        self.assertEqual(GPU_LOCK_PATH, Q1_GPU_LOCK_PATH)
        def fake_git(*args: str) -> str:
            return {
                ("branch", "--show-current"): "study/topic",
                ("config", "--get", "branch.study/topic.remote"): "origin",
                ("config", "--get", "branch.study/topic.merge"): "refs/heads/study/topic/with/slashes",
                ("ls-remote", "--exit-code", "origin", "refs/heads/study/topic/with/slashes"):
                    "a" * 40 + "\trefs/heads/study/topic/with/slashes",
            }[args]

        with patch("ako_runs.controlled_followup.tilelang_abstraction_v3.protocol._git", side_effect=fake_git):
            self.assertEqual(live_upstream_head(), {
                "remote": "origin",
                "ref": "refs/heads/study/topic/with/slashes",
                "commit": "a" * 40,
            })
        with patch.dict(os.environ, {GPU_LOCK_ENV: ""}):
            with self.assertRaisesRegex(RuntimeError, "inherited physical-GPU0 lock"):
                _validate_inherited_gpu_lock()

    def test_gpu_child_rederives_full_toolchain(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            lock = self._lock(Path(temporary))
            changed = deepcopy(self.toolchain)
            changed["triton_version"] = "foreign"
            with (
                patch(
                    "ako_runs.controlled_followup.tilelang_abstraction_v3.protocol.gpu_snapshot",
                    return_value=GPU,
                ),
                patch(
                    "ako_runs.controlled_followup.tilelang_abstraction_v3.protocol.live_toolchain",
                    return_value=changed,
                ),
                self.assertRaisesRegex(ProtocolError, "GPU/toolchain identity differs"),
            ):
                load_lock(lock, check_gpu=True)

    def test_manifest_is_paired_randomized_and_rejects_forged_sdpa(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            lock = self._lock(Path(temporary))
            eligible = {self.campaign["pairs"][0]["pair_id"], self.campaign["pairs"][1]["pair_id"]}
            admission = self._projected_admission(lock, eligible)
            first = make_timing_manifest(self.campaign, file_sha256(lock), admission)
            second = make_timing_manifest(self.campaign, file_sha256(lock), admission)
            self.assertEqual(first, second)
            self.assertEqual(len(first["rows"]), 2 * 2 * 15 * 4)
            for pair_id in eligible:
                for distribution in ("positive", "withheld_signed"):
                    for block in range(15):
                        rows = [row for row in first["rows"] if (row["pair_id"], row["distribution"], row["block"]) == (pair_id, distribution, block)]
                        self.assertEqual({row["role"] for row in rows}, {"high", "low", "sham_a", "sham_b"})
                        self.assertEqual(sorted(row["position"] for row in rows), list(range(4)))
            forged = self._projected_admission(lock, {self.campaign["pairs"][2]["pair_id"]})
            forged["pairs"][2].update({
                "gate_available": True, "classification": "runtime_estimand",
                "artifact_identity_sha256": {"high": "b" * 64, "low": "c" * 64},
            })
            with self.assertRaisesRegex(ProtocolError, "cannot authorize timing"):
                make_timing_manifest(self.campaign, file_sha256(lock), forged)
            tampered = deepcopy(first)
            tampered["rows"][0]["position"] = 99
            with self.assertRaisesRegex(RuntimeError, "deterministic"):
                validate_timing_manifest(tampered, self.campaign, file_sha256(lock), admission)

    def test_gate_hash_and_coordinates_are_rederived(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            directory = Path(temporary)
            lock = self._lock(directory)
            summary_path, summary = self._valid_matmul_admission(directory, lock)
            self.assertTrue(summary_path.is_file())
            foreign = summary_path.parent / "foreign.json"
            foreign.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "unexpected admission artifacts"):
                load_verified_admission(summary_path, lock)
            foreign.unlink()
            row = next(value for value in summary["pairs"] if value["family"] == "matmul")
            gate_path = repo_path(row["arm_receipts"]["high"]["admission_path"])
            gate = json.loads(gate_path.read_text())
            original_gate = deepcopy(gate)
            gate["gpu"] = {**GPU, "uuid": "GPU-foreign"}
            gate_path.write_text(json.dumps(gate, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "run status"):
                admission_summary(summary_path.parent, lock)
            gate_path.write_text(json.dumps(original_gate, sort_keys=True))

            forged_summary = deepcopy(summary)
            forged_summary["pairs"][2].update({
                "gate_available": True, "timing_eligible": True,
                "classification": "runtime_estimand",
                "artifact_identity_sha256": {"high": "d" * 64, "low": "e" * 64},
            })
            summary_path.write_text(json.dumps(forged_summary, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "differs from independently re-derived"):
                load_verified_admission(summary_path, lock)
            summary_path.write_text(json.dumps(summary, sort_keys=True))

            evidence = repo_path(gate["gate_path"])
            original = evidence.read_text()
            evidence.write_text(original + "{}\n")
            with self.assertRaisesRegex(ValueError, "run status"):
                admission_summary(summary_path.parent, lock)

    def test_timing_rederives_summaries_gpu_commands_and_artifact_shams(self) -> None:
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            directory = Path(temporary)
            lock = self._lock(directory)
            admission_path, admission = self._valid_matmul_admission(directory, lock)
            manifest = make_timing_manifest(self.campaign, file_sha256(lock), admission)
            manifest_path = directory / "timing_manifest.json"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            raw = directory / "timing" / "raw"
            raw.mkdir(parents=True)
            lock_value = read_json(lock)
            remote = self._remote(lock)
            (raw.parent / "launch_receipt.json").write_text(json.dumps({
                "campaign_id": self.campaign["campaign_id"],
                "campaign_lock_sha256": file_sha256(lock),
                "dependency_bundle_sha256": lock_value["dependency_bundle_sha256"],
                "toolchain": lock_value["toolchain"],
                "manifest_sha256": canonical_sha256(manifest),
                "manifest_file_sha256": file_sha256(manifest_path),
                "admission_summary_sha256": canonical_sha256(admission),
                "admission_summary_file_sha256": file_sha256(admission_path),
                "expected_records": len(manifest["rows"]),
                "physical_gpu": 0, "gpu": GPU,
                "gpu_lock_id": GPU_LOCK_ID,
                "gpu_idle_preflight": self._idle("timing_stage_pre", 999, 20.0),
                "remote_authorization": remote,
            }))
            positions = raw.parent / "position_receipts"
            positions.mkdir()
            pair = next(value for value in self.campaign["pairs"] if value["family"] == "matmul")
            admission_pair = next(value for value in admission["pairs"] if value["family"] == "matmul")
            record_paths = []
            predecessor_sha = None
            for index, row in enumerate(manifest["rows"], 1):
                path = raw / f"{index:04d}__{row['row_id']}.json"
                token = "high-executable" if row["implementation_side"] == "high" else "low-executable"
                generated = self._artifact(raw / (path.stem + "_generated"), token)
                expected_identity = admission_pair["artifact_identity_sha256"][row["implementation_side"]]
                value = 0.9 if row["role"] == "low" else 1.0
                times = [value] * 100
                launched = (20 + index * 10) * 1_000_000_000
                child_pid = 1000 + index
                t_start, t_end = 21 + index * 10, 22 + index * 10
                path.write_text(json.dumps({
                    "schema_version": 1, "campaign_id": self.campaign["campaign_id"], "ok": True,
                    "campaign_lock_sha256": file_sha256(lock),
                    "manifest_sha256": canonical_sha256(manifest),
                    "manifest_row": row, "manifest_row_sha256": canonical_sha256(row),
                    "dependency_bundle_sha256": lock_value["dependency_bundle_sha256"],
                    "toolchain": lock_value["toolchain"],
                    "physical_gpu": 0, "gpu": GPU, "remote_authorization": remote,
                    "command": _legacy_timing_command(pair, row), "cwd": str(PHASE1.resolve()),
                    "returncode": 0, "generated": generated,
                    "expected_artifact_identity_sha256": expected_identity,
                    "implementation_sha256": generated["identity_sha256"],
                    "artifact_identity_match": True, "artifact_error": None,
                    "gpu_idle_preflight": self._idle("timing_child_pre", child_pid, t_start - 0.5),
                    "gpu_idle_postflight": self._idle("timing_child_post", child_pid, t_end - 0.5),
                    "gpu_idle_postflight_error": None,
                    "parent_pid": 999, "process_pid": child_pid,
                    "times_ms": times,
                    "primary_tail_median_ms": statistics.median(times[60:100]),
                    "full_median_ms": statistics.median(times),
                    "first_decile_median_ms": statistics.median(times[:10]),
                    "last_decile_median_ms": statistics.median(times[-10:]),
                    "t_start": t_start, "t_end": t_end,
                }, sort_keys=True))
                position_path = positions / f"{index:04d}__{row['row_id']}.json"
                position_path.write_text(json.dumps({
                    "schema_version": 1,
                    "record_type": "tilelang_abstraction_v3_position_receipt",
                    "campaign_id": self.campaign["campaign_id"],
                    "manifest_sha256": canonical_sha256(manifest),
                    "row_id": row["row_id"],
                    "global_position": index,
                    "predecessor_position_receipt_sha256": predecessor_sha,
                    "child_pid": child_pid,
                    "child_launched_unix_ns": launched,
                    "child_completed_unix_ns": (23 + index * 10) * 1_000_000_000,
                    "returncode": 0,
                    "raw_path": str(path.resolve().relative_to(repo_path("."))),
                    "raw_sha256": file_sha256(path),
                    "gpu_idle_after_child": self._idle("timing_parent_after_child", 999, 22.5 + index * 10),
                    "gpu_postflight_error": None,
                }, sort_keys=True))
                predecessor_sha = file_sha256(position_path)
                record_paths.append(path)
            timing_hashes = timing_artifact_hashes(raw.parent, manifest)
            (raw.parent / "run_status.json").write_text(json.dumps({
                "schema_version": 1,
                "record_type": "tilelang_abstraction_v3_timing_run_status",
                "campaign_id": self.campaign["campaign_id"],
                "campaign_lock_sha256": file_sha256(lock),
                "complete": True,
                "expected_records": len(manifest["rows"]),
                "observed_records": len(manifest["rows"]),
                "artifact_sha256": timing_hashes,
                "artifact_bundle_sha256": canonical_sha256(timing_hashes),
                "gpu_idle_postflight": self._idle("timing_stage_post", 999, 2000.0),
                "launch_receipt_sha256": file_sha256(raw.parent / "launch_receipt.json"),
            }, sort_keys=True))
            result = timing_summary(raw, manifest_path, admission_path, lock)
            self.assertEqual(result["records"], 120)
            self.assertTrue(all(effect["clears_sham_resolution_floor"] for effect in result["effects"]))
            self.assertTrue(all(effect["classification"]["direction"] == "lower_level_faster" for effect in result["effects"]))

            second_position = positions / f"0002__{manifest['rows'][1]['row_id']}.json"
            original_position = second_position.read_text()
            status_path = raw.parent / "run_status.json"
            original_status = status_path.read_text()
            changed_position = json.loads(original_position)
            changed_position["predecessor_position_receipt_sha256"] = None
            second_position.write_text(json.dumps(changed_position, sort_keys=True))
            changed_status = json.loads(original_status)
            changed_hashes = timing_artifact_hashes(raw.parent, manifest)
            changed_status["artifact_sha256"] = changed_hashes
            changed_status["artifact_bundle_sha256"] = canonical_sha256(changed_hashes)
            status_path.write_text(json.dumps(changed_status, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "position receipt"):
                timing_summary(raw, manifest_path, admission_path, lock)
            second_position.write_text(original_position)
            status_path.write_text(original_status)

            target = record_paths[0]
            record = json.loads(target.read_text())
            original_record = deepcopy(record)
            record["gpu"] = {**GPU, "driver_version": "foreign"}
            target.write_text(json.dumps(record, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                timing_summary(raw, manifest_path, admission_path, lock)
            record = deepcopy(original_record)
            record["toolchain"]["triton_version"] = "foreign"
            target.write_text(json.dumps(record, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                timing_summary(raw, manifest_path, admission_path, lock)
            record = deepcopy(original_record)
            record["command"] = record["command"] + ["--foreign"]
            target.write_text(json.dumps(record, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                timing_summary(raw, manifest_path, admission_path, lock)
            record = deepcopy(original_record)
            record["full_median_ms"] = 7.0
            target.write_text(json.dumps(record, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "coordinates differ"):
                timing_summary(raw, manifest_path, admission_path, lock)
            record["full_median_ms"] = statistics.median(record["times_ms"])
            target.write_text(json.dumps(record, sort_keys=True))
            sass = repo_path(record["generated"]["files"]["sass"]["path"])
            sass.write_text("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                timing_summary(raw, manifest_path, admission_path, lock)


if __name__ == "__main__":
    unittest.main()
