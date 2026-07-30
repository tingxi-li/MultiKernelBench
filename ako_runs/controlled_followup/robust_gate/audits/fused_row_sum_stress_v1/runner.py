"""Run deterministic boundary controls or fresh gain-16 winner stress seeds."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ako_runs.controlled_followup.fused_grid.robust_adapter import (
    build_phase2_candidates,
    load_repository,
    select_jobs,
)
from ako_runs.controlled_followup.robust_gate.distributions import make_fused_inputs
from ako_runs.controlled_followup.robust_gate.metrics import compute_metrics
from ako_runs.controlled_followup.robust_gate.oracles import resolve_output
from ako_runs.controlled_followup.robust_gate.schema import (
    canonical_sha256,
    file_sha256,
    load_json,
)
from ako_runs.controlled_followup.robust_gate.seeds import derive_seed


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
MANIFEST_PATH = HERE / "manifest.json"
FREEZE_PATH = HERE / "receipts" / "freeze_receipt.json"
LAUNCH_PATH = HERE / "receipts" / "launch_receipt.json"
GATES = ("semantic_mixed", "conformance_mixed")


class AtomicJsonl:
    def __init__(self, output: Path):
        self.output = output
        self.partial = output.with_name(output.name + ".partial")
        if output.exists() or self.partial.exists():
            raise FileExistsError(f"refusing overwrite of {output} or {self.partial}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.partial, "x", encoding="utf-8")
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1

    def finish(self) -> None:
        self.handle.close()
        os.replace(self.partial, self.output)

    def retain(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def repo_path(relative: str) -> Path:
    path = (REPO_ROOT / relative).resolve()
    path.relative_to(REPO_ROOT)
    return path


def verify_campaign(require_freeze: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(MANIFEST_PATH)
    for label, spec in manifest["original_fused_v2"].items():
        path = repo_path(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"original fused-v2 {label} raw SHA mismatch")
        if "canonical_sha256" in spec and canonical_sha256(load_json(path)) != spec[
            "canonical_sha256"
        ]:
            raise ValueError(f"original fused-v2 {label} canonical SHA mismatch")
    adapter = manifest["runner_adapter"]
    if file_sha256(repo_path(adapter["path"])) != adapter["sha256"]:
        raise ValueError("frozen fused-grid adapter manifest changed")
    gate = load_json(repo_path(manifest["original_fused_v2"]["gate_spec"]["path"]))
    if canonical_sha256(gate) != manifest["original_fused_v2"]["gate_spec"][
        "canonical_sha256"
    ]:
        raise ValueError("gate object is not registered fused-v2")
    for gate_id in GATES:
        threshold = gate["gates"][f"fused_softmax/{gate_id}"]["thresholds"][
            "row_sum_error_max"
        ]["value"]
        if threshold != manifest["registered_row_sum_threshold"]:
            raise ValueError("registered row-sum threshold changed")
    if require_freeze:
        if not FREEZE_PATH.is_file():
            raise ValueError("production work requires receipts/freeze_receipt.json")
        freeze = load_json(FREEZE_PATH)
        if freeze.get("manifest_sha256") != file_sha256(MANIFEST_PATH):
            raise ValueError("stress manifest changed after freeze")
        if freeze.get("manifest_canonical_sha256") != canonical_sha256(manifest):
            raise ValueError("stress manifest canonical hash changed after freeze")
        for relative, expected in freeze.get("source_sha256", {}).items():
            if file_sha256(repo_path(relative)) != expected:
                raise ValueError(f"stress source changed after freeze: {relative}")
        if canonical_sha256(freeze["source_sha256"]) != freeze.get(
            "source_bundle_canonical_sha256"
        ):
            raise ValueError("freeze source bundle is inconsistent")
    return manifest, gate


def threshold_failures(gate: dict[str, Any], metrics: dict[str, float]) -> list[str]:
    failures = []
    for metric, threshold in gate["thresholds"].items():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{metric}=missing/nonfinite")
        elif value > threshold["value"]:
            failures.append(f"{metric}={value:.17g}>{threshold['value']:.17g}")
    return failures


def _base(
    manifest: dict[str, Any], freeze_bundle: str, *, arm: str, gate_id: str
) -> dict[str, Any]:
    original = manifest["original_fused_v2"]["gate_spec"]
    return {
        "schema_version": "1.0",
        "record_type": "fused_v2_row_sum_stress_measurement",
        "campaign_id": manifest["campaign_id"],
        "stress_manifest_sha256": file_sha256(MANIFEST_PATH),
        "stress_manifest_canonical_sha256": canonical_sha256(manifest),
        "original_gate_sha256": original["sha256"],
        "original_gate_canonical_sha256": original["canonical_sha256"],
        "source_bundle_canonical_sha256": freeze_bundle,
        "arm": arm,
        "op": "fused_softmax",
        "gate_id": gate_id,
    }


def run_boundary(
    manifest: dict[str, Any], gate_spec: dict[str, Any], writer: AtomicJsonl, bundle: str
) -> None:
    shape = manifest["cpu_boundary_shape"]
    reference = torch.full(
        (shape["M"], shape["N"]),
        1.0 / shape["N"],
        dtype=torch.float64,
    )
    for control in manifest["boundary_controls"]:
        candidate = reference * (1.0 + control["row_sum_delta"])
        metrics = compute_metrics("fused_softmax", reference, candidate)
        for gate_id in GATES:
            gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
            failures = threshold_failures(gate, metrics)
            raw_cutoff = (
                gate["thresholds"]["row_sum_error_max"]["observed_anchor_max"]
                * gate["thresholds"]["row_sum_error_max"]["safety_factor"]
            )
            row = _base(manifest, bundle, arm="boundary", gate_id=gate_id)
            row.update(
                {
                    "candidate": control["control_id"],
                    "role": "deterministic_boundary_control",
                    "shape": shape,
                    "device": "cpu",
                    "row_sum_delta_requested": control["row_sum_delta"],
                    "raw_safety_cutoff": raw_cutoff,
                    "registered_threshold": gate["thresholds"]["row_sum_error_max"][
                        "value"
                    ],
                    "expected_gate_outcome": control["expected_gate_outcome"],
                    "expected_raw_safety_outcome": control[
                        "expected_raw_safety_outcome"
                    ],
                    "metrics": metrics,
                    "ok": True,
                    "threshold_failures": failures,
                    "gate_pass": not failures,
                    "raw_safety_exceeded": metrics["row_sum_error_max"] > raw_cutoff,
                    "failure_metrics": [value.split("=", 1)[0] for value in failures],
                }
            )
            writer.write(row)


def _seed_map(namespace: str, case_id: str, index: int) -> dict[str, int]:
    return {
        tensor: derive_seed(
            namespace, "fused_softmax", case_id, "validation", tensor, index
        )
        for tensor in ("x", "weight", "bias")
    }


def _write_winner_failure(
    writer: AtomicJsonl,
    manifest: dict[str, Any],
    bundle: str,
    winner: dict[str, Any],
    index: int,
    seeds: dict[str, int],
    category: str,
    error: BaseException | str,
) -> None:
    for gate_id in GATES:
        row = _base(manifest, bundle, arm="winners", gate_id=gate_id)
        row.update(
            {
                "candidate": winner["job_id"],
                "dsl": winner["dsl"],
                "job_sha256": winner["job_sha256"],
                "role": "prior_dsl_winner",
                "case_id": manifest["case"]["id"],
                "namespace": manifest["stress_split"]["namespace"],
                "seed_index": index,
                "tensor_seeds": seeds,
                "shape": manifest["shape"],
                "device": "cuda:0",
                "ok": False,
                "gate_pass": False,
                "threshold_failures": ["collection_failure"],
                "error_category": category,
                "error": str(error),
            }
        )
        writer.write(row)


def run_winners(
    manifest: dict[str, Any], gate_spec: dict[str, Any], writer: AtomicJsonl, bundle: str
) -> None:
    phase2 = REPO_ROOT / "ako_runs" / "phase2_fused_sdpa"
    if str(phase2) not in sys.path:
        sys.path.insert(0, str(phase2))
    import common2  # type: ignore  # noqa: PLC0415

    common2.ARTIFACTS_DIR = str(HERE / "build_artifacts")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_extensions")
    context = load_repository(repo_path(manifest["runner_adapter"]["path"]))
    if context.source_bundle_sha256 != manifest["runner_adapter"]["source_bundle_sha256"]:
        raise ValueError("adapter runtime source bundle changed")
    ids = [winner["job_id"] for winner in manifest["winners"]]
    jobs = select_jobs(context, ids)
    for winner, job in zip(manifest["winners"], jobs, strict=True):
        if context.adapter["grid"]["job_sha256"][job["job_id"]] != winner["job_sha256"]:
            raise ValueError(f"winner job hash changed: {job['job_id']}")
    plans = build_phase2_candidates(context, jobs)
    plan_by_job = {plan.job_id: plan for plan in plans}
    case_id = manifest["case"]["id"]
    namespace = manifest["stress_split"]["namespace"]
    previous_inputs = None
    for index in range(manifest["stress_split"]["seeds"]):
        seeds = _seed_map(namespace, case_id, index)
        try:
            inputs = make_fused_inputs(
                manifest["shape"], manifest["case"], seeds, "cuda"
            )
            # Keep the prior allocation live until the next seed has been
            # allocated; Phase-2 cached weights are keyed by tensor address.
            previous_inputs = None
            references = {
                gate_id: resolve_output(
                    gate_spec["gates"][f"fused_softmax/{gate_id}"]["reference"]["kind"],
                    "fused_softmax",
                    inputs,
                    gate_spec["gates"][f"fused_softmax/{gate_id}"]["contract"],
                )
                for gate_id in GATES
            }
        except BaseException as exc:
            for winner in manifest["winners"]:
                _write_winner_failure(
                    writer, manifest, bundle, winner, index, seeds, "setup", exc
                )
            continue

        prepared: dict[str, Any] = {}
        for winner in manifest["winners"]:
            plan = plan_by_job[winner["job_id"]]
            if plan.build_error or plan.execute is None:
                _write_winner_failure(
                    writer,
                    manifest,
                    bundle,
                    winner,
                    index,
                    seeds,
                    "build",
                    plan.build_error or "candidate has no execute function",
                )
                continue
            started = time.perf_counter()
            try:
                with torch.no_grad():
                    output = plan.execute(inputs, prepared).float()
                torch.cuda.synchronize()
                candidate_wall_s = time.perf_counter() - started
            except BaseException as exc:
                _write_winner_failure(
                    writer, manifest, bundle, winner, index, seeds, "execution", exc
                )
                continue
            for gate_id in GATES:
                gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                row = _base(manifest, bundle, arm="winners", gate_id=gate_id)
                try:
                    metrics = compute_metrics(
                        "fused_softmax", references[gate_id], output, inputs
                    )
                    failures = threshold_failures(gate, metrics)
                    raw_cutoff = (
                        gate["thresholds"]["row_sum_error_max"]["observed_anchor_max"]
                        * gate["thresholds"]["row_sum_error_max"]["safety_factor"]
                    )
                    row.update(
                        {
                            "candidate": winner["job_id"],
                            "dsl": winner["dsl"],
                            "job_sha256": winner["job_sha256"],
                            "role": "prior_dsl_winner",
                            "case_id": case_id,
                            "namespace": namespace,
                            "seed_index": index,
                            "tensor_seeds": seeds,
                            "shape": manifest["shape"],
                            "device": "cuda:0",
                            "candidate_wall_s": candidate_wall_s,
                            "build_metadata": plan.build_metadata,
                            "metrics": metrics,
                            "ok": True,
                            "threshold_failures": failures,
                            "gate_pass": not failures,
                            "raw_safety_cutoff": raw_cutoff,
                            "raw_safety_exceeded": metrics["row_sum_error_max"]
                            > raw_cutoff,
                        }
                    )
                except BaseException as exc:
                    row.update(
                        {
                            "candidate": winner["job_id"],
                            "dsl": winner["dsl"],
                            "job_sha256": winner["job_sha256"],
                            "role": "prior_dsl_winner",
                            "case_id": case_id,
                            "namespace": namespace,
                            "seed_index": index,
                            "tensor_seeds": seeds,
                            "shape": manifest["shape"],
                            "device": "cuda:0",
                            "ok": False,
                            "gate_pass": False,
                            "threshold_failures": ["metric_failure"],
                            "error_category": "metric",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                writer.write(row)
            del output
        previous_inputs = inputs
        if (index + 1) % 16 == 0:
            print(f"winners stress: {index + 1}/256 seeds", flush=True)
    del previous_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("boundary", "winners"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    manifest, gate = verify_campaign(require_freeze=not args.cpu_smoke)
    if args.cpu_smoke and args.arm != "boundary":
        parser.error("CPU smoke is available only for boundary controls")
    if args.arm == "winners" and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        parser.error("winner stress requires exactly CUDA_VISIBLE_DEVICES=1")
    output = Path(args.out).resolve()
    if not args.cpu_smoke:
        workload = next(item for item in manifest["workloads"] if item["arm"] == args.arm)
        registered = (HERE / workload["output"]).resolve()
        if output != registered:
            parser.error(f"output must be registered path {registered}")
        if not LAUNCH_PATH.is_file():
            parser.error("production work requires receipts/launch_receipt.json")
        launch = load_json(LAUNCH_PATH)
        if launch.get("freeze_receipt_sha256") != file_sha256(FREEZE_PATH):
            parser.error("launch receipt does not bind current freeze receipt")
        matches = [
            item
            for item in launch.get("workloads", [])
            if item.get("arm") == args.arm and repo_path(item["output"]) == output
        ]
        if len(matches) != 1:
            parser.error("arm/output is not uniquely preregistered in launch receipt")
        bundle = load_json(FREEZE_PATH)["source_bundle_canonical_sha256"]
    else:
        bundle = "cpu-smoke-unfrozen"
    writer = AtomicJsonl(output)
    try:
        if args.arm == "boundary":
            run_boundary(manifest, gate, writer, bundle)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            run_winners(manifest, gate, writer, bundle)
        writer.finish()
    except BaseException:
        writer.retain()
        print(f"retained {writer.count} rows at {writer.partial}", file=sys.stderr)
        raise
    print(f"wrote {writer.count} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
