"""Run the preregistered same-seed fused frontier stress."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from ako_runs.controlled_followup.fused_frontier_closure_v3 import candidates, core
from ako_runs.controlled_followup.robust_gate.distributions import make_fused_inputs
from ako_runs.controlled_followup.robust_gate.metrics import compute_metrics
from ako_runs.controlled_followup.robust_gate.oracles import resolve_output
from ako_runs.controlled_followup.robust_gate.schema import canonical_sha256
from ako_runs.controlled_followup.robust_gate.seeds import derive_seed


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OUT = HERE / "results" / "measurements.jsonl"
CAMPAIGN = REPO / "ako_runs/controlled_followup/fused_frontier_closure_v3/campaign.json"
GATE_SPEC = REPO / "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json"
PRIOR_NS = "MKB-fused-reachability-row-sum-gain16-stress-v1-20260730"
NEW_NS = "MKB-fused-same-seed-gain16-v1-20260730"
CASE = {"id": "signed_normal_gain16", "activation_distribution": "normal_zero", "weight_gain": 16.0}
GATES = ("semantic_mixed", "conformance_mixed")


def seeds(index: int) -> dict[str, int]:
    namespace = PRIOR_NS if index < 256 else NEW_NS
    return {
        tensor: derive_seed(namespace, "fused_softmax", CASE["id"], "validation", tensor, index)
        for tensor in ("x", "weight", "bias")
    }


def main() -> int:
    if OUT.exists():
        raise RuntimeError(f"refusing overwrite: {OUT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("run requires CUDA_VISIBLE_DEVICES=2")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    campaign = core.load_campaign(CAMPAIGN)
    definitions = core.candidates_by_id(campaign)
    gate_spec = json.loads(GATE_SPEC.read_text(encoding="utf-8"))
    built = {
        candidate_id: candidates.build_candidate(definition, expected_candidate_id=candidate_id)
        for candidate_id, definition in definitions.items()
    }
    selected = campaign["candidate_order"]
    shape = {"M": 1024, "K": 8192, "N": 8192}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as handle:
        for index in range(512):
            tensor_seeds = seeds(index)
            inputs = make_fused_inputs(shape, CASE, tensor_seeds, "cuda:0")
            refs = {
                gate_id: resolve_output(
                    gate_spec["gates"][f"fused_softmax/{gate_id}"]["reference"]["kind"],
                    "fused_softmax", inputs,
                    gate_spec["gates"][f"fused_softmax/{gate_id}"]["contract"],
                )
                for gate_id in GATES
            }
            for candidate_id in selected:
                candidate = built[candidate_id]
                started = time.perf_counter()
                with torch.no_grad():
                    output = candidate.run(inputs["x"], inputs["x"].half().contiguous(), inputs["weight"], inputs["bias"]).float()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                for gate_id in GATES:
                    gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                    metrics = compute_metrics("fused_softmax", refs[gate_id], output, inputs)
                    failures = [
                        f"{name}={metrics[name]:.17g}>{rule['value']:.17g}"
                        for name, rule in gate["thresholds"].items()
                        if metrics.get(name, float("inf")) > rule["value"]
                    ]
                    ratios = {
                        name: metrics[name] / rule["value"]
                        for name, rule in gate["thresholds"].items()
                        if rule["value"] > 0 and name in metrics
                    }
                    row = {
                        "schema_version": 1,
                        "record_type": "fused_same_seed_stress_measurement",
                        "campaign_id": "controlled-followup-fused-same-seed-stress-v1",
                        "candidate_id": candidate_id,
                        "candidate_definition_sha256": canonical_sha256(definitions[candidate_id]),
                        "case_id": CASE["id"],
                        "seed_index": index,
                        "seed_namespace": PRIOR_NS if index < 256 else NEW_NS,
                        "tensor_seeds": tensor_seeds,
                        "gate_id": gate_id,
                        "metrics": metrics,
                        "threshold_ratios": ratios,
                        "gate_pass": not failures,
                        "threshold_failures": failures,
                        "candidate_wall_s_diagnostic": elapsed,
                        "physical_gpu": 2,
                        "correctness_only": True,
                        "threshold_mutation_authorized": False,
                    }
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                del output, inputs, refs
            if index % 8 == 0:
                handle.flush()
                os.fsync(handle.fileno())
                print(f"seed {index}/512", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
