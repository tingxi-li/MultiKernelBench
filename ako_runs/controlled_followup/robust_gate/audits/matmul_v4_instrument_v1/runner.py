"""Run preregistered fixed-threshold checks against the immutable matmul-v4 gate.

This module intentionally contains no threshold-fitting path.  Numerical and
real-kernel outcomes are compared directly with the values in the original v4
gate; every attempted row is fsynced to a retained ``.partial`` file before the
completed file is atomically exposed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from ako_runs.controlled_followup.robust_gate.distributions import make_matmul_inputs
from ako_runs.controlled_followup.robust_gate.metrics import MetricInputError
from ako_runs.controlled_followup.robust_gate.oracles import (
    contract_reference,
    native_fp32_reference,
    native_mixed_reference,
    semantic_reference,
)
from ako_runs.controlled_followup.robust_gate.seeds import derive_seed

from .bindings import (
    HERE,
    REPO_ROOT,
    AtomicJsonl,
    audit_manifest,
    audit_manifest_hashes,
    threshold_failures,
    verify_launch_receipt,
    verify_original_v4,
)


GATE_ORDER = ("conformance_mixed", "semantic_mixed", "semantic_q32")
MIXED_CONTRACT = {
    "operand_dtype": "fp16",
    "a_dtype": "fp16",
    "b_dtype": "fp16",
    "output_dtype": "fp32",
}


def compute_matmul_metrics_precomputed(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    """Canonical matmul metrics using a once-per-input ``abs(A) @ abs(B)``."""
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise MetricInputError("reference and candidate must both be tensors")
    if reference.shape != candidate.shape:
        raise MetricInputError(
            f"shape mismatch: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        raise MetricInputError("empty outputs are not valid")
    if not torch.is_floating_point(reference) or not torch.is_floating_point(candidate):
        raise MetricInputError("reference and candidate outputs must be floating point")
    if not bool(torch.isfinite(reference).all()):
        raise MetricInputError("reference contains NaN or Inf")
    if not bool(torch.isfinite(candidate).all()):
        raise MetricInputError("candidate contains NaN or Inf")
    if not isinstance(scale, torch.Tensor) or scale.shape != reference.shape:
        raise MetricInputError("matmul backward-error scale has wrong shape")
    if not bool(torch.isfinite(scale).all()):
        raise MetricInputError("matmul backward-error scale contains NaN or Inf")

    ref = reference.double()
    got = candidate.double()
    error = got - ref
    absolute = error.abs()
    ref_norm = torch.linalg.vector_norm(ref.reshape(-1))
    error_norm = torch.linalg.vector_norm(error.reshape(-1))
    denominator = max(float(ref_norm.item()), torch.finfo(torch.float64).tiny)
    scale64 = scale.double()
    scale_norm = torch.linalg.vector_norm(scale64.reshape(-1))
    floor = max(
        float(scale64.max().item()) * torch.finfo(torch.float64).eps,
        torch.finfo(torch.float64).tiny,
    )
    nrmse = float(error_norm.item()) / denominator
    # The registered paired-cancellation reference is bitwise zero.  For a
    # deliberately nonzero wrong answer, the canonical diagnostic-only NRMSE
    # overflows (nonzero/tiny) even though every *gated* metric is finite.  Keep
    # the gate decision observable by saturating only that ungated diagnostic.
    if not math.isfinite(nrmse) and float(ref_norm.item()) == 0.0:
        nrmse = torch.finfo(torch.float64).max
    metrics = {
        "nonfinite_count": 0.0,
        "max_abs_err": float(absolute.max().item()),
        "mean_abs_err": float(absolute.mean().item()),
        "nrmse": nrmse,
        "abs_signed_bias": abs(float(error.mean().item())),
        "reference_abs_mean": float(ref.abs().mean().item()),
        "reference_abs_max": float(ref.abs().max().item()),
        "componentwise_backward_error": float((absolute / (scale64 + floor)).max().item()),
        "scaled_rmse": float(error_norm.item())
        / max(float(scale_norm.item()), torch.finfo(torch.float64).tiny),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise MetricInputError("one or more metrics are non-finite")
    return metrics


def tensor_seeds(namespace: str, case_id: str, index: int) -> dict[str, int]:
    return {
        tensor: derive_seed(namespace, "matmul", case_id, "validation", tensor, index)
        for tensor in ("a", "b")
    }


def _case_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in manifest["cases"]}


def _reference(gate_id: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if gate_id == "conformance_mixed":
        return contract_reference("matmul", inputs, MIXED_CONTRACT)
    return semantic_reference("matmul", inputs)


def _scale(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs["a"].double().abs() @ inputs["b"].double().abs()


def _base_record(
    context: dict[str, Any],
    *,
    arm: str,
    block_id: str | None,
    namespace: str,
    case_id: str,
    seed_index: int,
    seeds: dict[str, int],
    gate_id: str,
    candidate: str,
    role: str,
    expected_outcome: str,
    shape: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "record_type": "matmul_v4_fixed_threshold_audit_measurement",
        "campaign_id": context["manifest"]["campaign_id"],
        "audit_manifest_sha256": context["manifest_hashes"]["raw_sha256"],
        "audit_manifest_canonical_sha256": context["manifest_hashes"]["canonical_sha256"],
        "original_gate_sha256": context["manifest"]["original_v4"]["gate_spec"]["sha256"],
        "original_gate_canonical_sha256": context["manifest"]["original_v4"]["gate_spec"]["canonical_sha256"],
        "source_bundle_canonical_sha256": context["source_bundle"],
        "mode": context["mode"],
        "arm": arm,
        "block_id": block_id,
        "namespace": namespace,
        "case_id": case_id,
        "seed_index": seed_index,
        "tensor_seeds": seeds,
        "op": "matmul",
        "gate_id": gate_id,
        "candidate": candidate,
        "role": role,
        "expected_outcome": expected_outcome,
        "reference_kind": context["manifest"]["gate_routes"][gate_id]["reference_kind"],
        "shape": dict(shape),
        "device": context["device"],
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _finish_measurement(
    record: dict[str, Any],
    gate: dict[str, Any],
    reference: torch.Tensor | None,
    candidate: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    error: BaseException | None = None,
    error_category: str = "collection",
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if error is not None:
            raise error
        assert reference is not None and candidate is not None and scale is not None
        record["reference_exact_zero"] = bool(torch.count_nonzero(reference).item() == 0)
        record["output_dtype"] = str(candidate.dtype)
        record["output_shape"] = list(candidate.shape)
        metrics = compute_matmul_metrics_precomputed(reference, candidate, scale)
        failures = threshold_failures(gate, metrics)
        record.update(
            {
                "ok": True,
                "metrics": metrics,
                "threshold_failures": failures,
                "gate_pass": not failures,
            }
        )
    except BaseException as exc:  # every attempted outcome is evidence
        record.update(
            {
                "ok": False,
                "gate_pass": False,
                "threshold_failures": ["collection_or_preflight_failure"],
                "error_category": error_category,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    record["measurement_wall_s"] = time.perf_counter() - started
    return record


def _setup_error_records(
    context: dict[str, Any],
    *,
    arm: str,
    block_id: str | None,
    namespace: str,
    case_id: str,
    seed_index: int,
    seeds: dict[str, int],
    shape: dict[str, int],
    planned: list[tuple[str, str, str, str]],
    error: BaseException,
) -> list[dict[str, Any]]:
    rows = []
    for gate_id, candidate, role, expected in planned:
        base = _base_record(
            context,
            arm=arm,
            block_id=block_id,
            namespace=namespace,
            case_id=case_id,
            seed_index=seed_index,
            seeds=seeds,
            gate_id=gate_id,
            candidate=candidate,
            role=role,
            expected_outcome=expected,
            shape=shape,
        )
        rows.append(
            _finish_measurement(
                base,
                context["gate"]["gates"][f"matmul/{gate_id}"],
                None,
                None,
                None,
                error=RuntimeError(f"input/reference setup failed: {error}"),
                error_category="input_or_reference_setup",
            )
        )
    return rows


def _replication_plan(manifest: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return [
        (
            gate_id,
            manifest["gate_routes"][gate_id]["positive_control"],
            "positive_control",
            "pass",
        )
        for gate_id in GATE_ORDER
    ]


def run_replication(
    context: dict[str, Any], writer: AtomicJsonl, block_id: str, max_seeds: int | None
) -> None:
    manifest = context["manifest"]
    blocks = {item["block_id"]: item for item in manifest["splits"]["replication_blocks"]}
    if block_id not in blocks:
        raise ValueError(f"unknown replication block {block_id!r}")
    split = blocks[block_id]
    count = min(split["seeds_per_case"], max_seeds or split["seeds_per_case"])
    shape = context["shape"]
    plan = _replication_plan(manifest)
    for case_id, case in _case_map(manifest).items():
        for index in range(count):
            seeds = tensor_seeds(split["namespace"], case_id, index)
            try:
                with torch.no_grad():
                    inputs = make_matmul_inputs(shape, case, seeds, context["device"])
                    scale = _scale(inputs)
                    semantic = semantic_reference("matmul", inputs)
                    contract = contract_reference("matmul", inputs, MIXED_CONTRACT)
                    native_mixed = native_mixed_reference("matmul", inputs, MIXED_CONTRACT)
                    native_q32 = native_fp32_reference("matmul", inputs)
            except BaseException as exc:
                for row in _setup_error_records(
                    context,
                    arm="replication",
                    block_id=block_id,
                    namespace=split["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    shape=shape,
                    planned=plan,
                    error=exc,
                ):
                    writer.write(row)
                continue
            values = {
                "conformance_mixed": (contract, native_mixed),
                "semantic_mixed": (semantic, native_mixed),
                "semantic_q32": (semantic, native_q32),
            }
            for gate_id, candidate_name, role, expected in plan:
                reference, candidate = values[gate_id]
                base = _base_record(
                    context,
                    arm="replication",
                    block_id=block_id,
                    namespace=split["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    gate_id=gate_id,
                    candidate=candidate_name,
                    role=role,
                    expected_outcome=expected,
                    shape=shape,
                )
                writer.write(
                    _finish_measurement(
                        base,
                        context["gate"]["gates"][f"matmul/{gate_id}"],
                        reference,
                        candidate,
                        scale,
                    )
                )
            del inputs, scale, semantic, contract, native_mixed, native_q32
        print(f"replication {block_id}: completed {case_id} ({count} seeds)", flush=True)


def _synthetic_plan(
    manifest: dict[str, Any], case_id: str
) -> list[tuple[str, str, str, str]]:
    rows = []
    for control in manifest["wrong_answer_controls"]:
        expected = "pass" if case_id in control["exact_pass_cases"] else "reject"
        role = "exact_zero_exclusion" if expected == "pass" else "negative_control"
        for gate_id in control["gates"]:
            rows.append((gate_id, control["control_id"], role, expected))
    return rows


def _wrong_output(
    control_id: str,
    gate_id: str,
    reference: torch.Tensor,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    if control_id == "zeros":
        return torch.zeros_like(reference)
    if control_id == "row_roll1":
        return torch.roll(reference, 1, dims=0)
    if control_id == "column_roll1":
        return torch.roll(reference, 1, dims=1)
    if control_id == "uniform_offset_0p002":
        return reference + 0.002
    if control_id == "single_spike_0p25":
        output = reference.clone()
        output[0, 0] += 0.25
        return output
    if control_id == "forced_fp16_exact_q32":
        if gate_id != "semantic_q32":
            raise ValueError("forced_fp16_exact_q32 is registered only for semantic_q32")
        return contract_reference("matmul", inputs, MIXED_CONTRACT)

    if gate_id == "semantic_q32":
        a, b = inputs["a"].double(), inputs["b"].double()
    else:
        a = inputs["a"].to(torch.float16).double()
        b = inputs["b"].to(torch.float16).double()
    if control_id == "drop_odd_k":
        return (a[:, 0::2] @ b[0::2, :]).float()
    if control_id == "k_roll1":
        return (a @ torch.roll(b, 1, dims=0)).float()
    raise ValueError(f"unknown wrong-answer control {control_id!r}")


def run_synthetic(
    context: dict[str, Any], writer: AtomicJsonl, max_seeds: int | None
) -> None:
    manifest = context["manifest"]
    split = manifest["splits"]["synthetic"]
    count = min(split["seeds_per_case"], max_seeds or split["seeds_per_case"])
    shape = context["shape"]
    for case_id, case in _case_map(manifest).items():
        plan = _synthetic_plan(manifest, case_id)
        for index in range(count):
            seeds = tensor_seeds(split["namespace"], case_id, index)
            try:
                with torch.no_grad():
                    inputs = make_matmul_inputs(shape, case, seeds, context["device"])
                    scale = _scale(inputs)
                    references = {gate_id: _reference(gate_id, inputs) for gate_id in GATE_ORDER}
            except BaseException as exc:
                for row in _setup_error_records(
                    context,
                    arm="synthetic",
                    block_id=None,
                    namespace=split["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    shape=shape,
                    planned=plan,
                    error=exc,
                ):
                    writer.write(row)
                continue
            for gate_id, control_id, role, expected in plan:
                reference = references[gate_id]
                candidate = None
                candidate_error = None
                try:
                    with torch.no_grad():
                        candidate = _wrong_output(control_id, gate_id, reference, inputs)
                except BaseException as exc:
                    candidate_error = exc
                base = _base_record(
                    context,
                    arm="synthetic",
                    block_id=None,
                    namespace=split["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    gate_id=gate_id,
                    candidate=control_id,
                    role=role,
                    expected_outcome=expected,
                    shape=shape,
                )
                writer.write(
                    _finish_measurement(
                        base,
                        context["gate"]["gates"][f"matmul/{gate_id}"],
                        reference,
                        candidate,
                        scale,
                        error=candidate_error,
                        error_category="control_construction",
                    )
                )
            del inputs, scale, references
        print(f"synthetic: completed {case_id} ({count} seeds)", flush=True)

    structural = manifest["splits"]["structural_smoke"]
    structural_count = min(
        structural["seeds_per_case"], max_seeds or structural["seeds_per_case"]
    )
    for case_id, case in _case_map(manifest).items():
        for index in range(structural_count):
            seeds = tensor_seeds(structural["namespace"], case_id, index)
            try:
                with torch.no_grad():
                    inputs = make_matmul_inputs(shape, case, seeds, context["device"])
                    scale = _scale(inputs)
                    references = {gate_id: _reference(gate_id, inputs) for gate_id in GATE_ORDER}
            except BaseException as exc:
                planned = [
                    (gate_id, control["control_id"], "structural_control", "reject")
                    for control in manifest["structural_controls"]
                    for gate_id in GATE_ORDER
                ]
                for row in _setup_error_records(
                    context,
                    arm="structural_smoke",
                    block_id=None,
                    namespace=structural["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    shape=shape,
                    planned=planned,
                    error=exc,
                ):
                    writer.write(row)
                continue
            for control in manifest["structural_controls"]:
                for gate_id in GATE_ORDER:
                    reference = references[gate_id]
                    if control["control_id"] == "nonfinite_nan":
                        candidate = reference.clone()
                        candidate.reshape(-1)[0] = float("nan")
                    elif control["control_id"] == "wrong_shape_drop_column":
                        candidate = reference[:, :-1]
                    else:
                        raise ValueError(f"unknown structural control {control['control_id']!r}")
                    base = _base_record(
                        context,
                        arm="structural_smoke",
                        block_id=None,
                        namespace=structural["namespace"],
                        case_id=case_id,
                        seed_index=index,
                        seeds=seeds,
                        gate_id=gate_id,
                        candidate=control["control_id"],
                        role="structural_control",
                        expected_outcome="reject",
                        shape=shape,
                    )
                    writer.write(
                        _finish_measurement(
                            base,
                            context["gate"]["gates"][f"matmul/{gate_id}"],
                            reference,
                            candidate,
                            scale,
                            error_category="metric_preflight",
                        )
                    )
            del inputs, scale, references
        print(f"structural smoke: completed {case_id}", flush=True)


def _real_plan(manifest: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return [
        (gate_id, candidate["candidate_id"], "real_candidate", "pass")
        for candidate in manifest["real_candidates"]
        for gate_id in candidate["gates"]
    ]


def _build_real_candidates(
    manifest: dict[str, Any], shape: dict[str, int]
) -> tuple[dict[str, Any], dict[str, BaseException], dict[str, dict[str, Any]]]:
    phase1 = REPO_ROOT / "ako_runs" / "phase1_matmul"
    sys.path.insert(0, str(phase1))
    import common  # type: ignore  # noqa: PLC0415
    import variants  # type: ignore  # noqa: PLC0415

    common.setup_cuda_env()
    common.ARTIFACTS_DIR = str(HERE / "build_artifacts")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_extensions")
    built: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for registered in manifest["real_candidates"]:
        candidate_id = registered["candidate_id"]
        spec = registered["config"]
        try:
            cfg = common.make_config(
                "triton",
                spec["variant"],
                spec["geometry"],
                M=shape["M"],
                K=shape["K"],
                N=shape["N"],
                cast=spec["cast"],
            )
            observed = {
                "dsl": cfg.dsl,
                "variant": cfg.variant,
                "M": cfg.M,
                "K": cfg.K,
                "N": cfg.N,
                "BM": cfg.BM,
                "BN": cfg.BN,
                "BK": cfg.BK,
                "threads": cfg.threads,
                "kc": cfg.kc,
                "stages": cfg.stages,
                "arith": cfg.arith,
                "cast": cfg.cast,
            }
            expected = {
                key: spec[key]
                for key in ("dsl", "variant", "BM", "BN", "BK", "threads", "kc", "stages", "arith", "cast")
            }
            if {key: observed[key] for key in expected} != expected:
                raise ValueError(f"Phase-1 config mismatch: {observed} != {expected}")
            built[candidate_id] = variants.build(cfg)
            metadata[candidate_id] = {
                "config": observed,
                "compile_s": built[candidate_id].compile_s,
                "input_dtype": str(built[candidate_id].input_dtype),
                "notes": built[candidate_id].notes,
            }
        except BaseException as exc:
            errors[candidate_id] = exc
    return built, errors, metadata


def run_real_contact(
    context: dict[str, Any], writer: AtomicJsonl, max_seeds: int | None
) -> None:
    manifest = context["manifest"]
    split = manifest["splits"]["real_contact"]
    count = min(split["seeds_per_case"], max_seeds or split["seeds_per_case"])
    shape = context["shape"]
    built, build_errors, build_metadata = _build_real_candidates(manifest, shape)
    plan = _real_plan(manifest)
    candidate_specs = {item["candidate_id"]: item for item in manifest["real_candidates"]}
    for case_id, case in _case_map(manifest).items():
        for index in range(count):
            seeds = tensor_seeds(split["namespace"], case_id, index)
            try:
                with torch.no_grad():
                    inputs = make_matmul_inputs(shape, case, seeds, context["device"])
                    scale = _scale(inputs)
                    references = {gate_id: _reference(gate_id, inputs) for gate_id in GATE_ORDER}
            except BaseException as exc:
                for row in _setup_error_records(
                    context,
                    arm="real_contact",
                    block_id=None,
                    namespace=split["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    shape=shape,
                    planned=plan,
                    error=exc,
                ):
                    writer.write(row)
                continue

            outputs: dict[str, torch.Tensor] = {}
            output_errors: dict[str, BaseException] = dict(build_errors)
            for candidate_id, candidate in built.items():
                try:
                    input_dtype = candidate_specs[candidate_id]["config"]["arith"]
                    if input_dtype == "fp16":
                        a = inputs["a"].half().contiguous()
                        b = inputs["b"].half().contiguous()
                    else:
                        a, b = inputs["a"], inputs["b"]
                    with torch.no_grad():
                        outputs[candidate_id] = candidate.run(a, b).float()
                    del a, b
                except BaseException as exc:
                    output_errors[candidate_id] = exc
            for gate_id, candidate_id, role, expected in plan:
                base = _base_record(
                    context,
                    arm="real_contact",
                    block_id=None,
                    namespace=split["namespace"],
                    case_id=case_id,
                    seed_index=index,
                    seeds=seeds,
                    gate_id=gate_id,
                    candidate=candidate_id,
                    role=role,
                    expected_outcome=expected,
                    shape=shape,
                )
                if candidate_id in build_metadata:
                    base["candidate_build"] = build_metadata[candidate_id]
                writer.write(
                    _finish_measurement(
                        base,
                        context["gate"]["gates"][f"matmul/{gate_id}"],
                        references[gate_id],
                        outputs.get(candidate_id),
                        scale,
                        error=output_errors.get(candidate_id),
                        error_category=(
                            "candidate_build"
                            if candidate_id in build_errors
                            else "candidate_execution"
                        ),
                    )
                )
            del inputs, scale, references, outputs
        print(f"real contact: completed {case_id} ({count} seeds)", flush=True)


def _registered_output(manifest: dict[str, Any], arm: str, block_id: str | None) -> Path:
    matches = [
        workload
        for workload in manifest["workloads"]
        if workload["arm"] == arm and workload.get("block_id") == block_id
    ]
    if len(matches) != 1:
        raise ValueError("arm/block is not a unique registered workload")
    return HERE / matches[0]["output"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("replication", "synthetic", "real_contact"))
    parser.add_argument("--block", choices=("r0", "r1", "r2", "r3"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    args = parser.parse_args()

    if args.arm == "replication" and not args.block:
        parser.error("--arm replication requires --block")
    if args.arm != "replication" and args.block:
        parser.error("--block is valid only for replication")
    if args.max_seeds is not None and args.max_seeds <= 0:
        parser.error("--max-seeds must be positive")

    manifest = audit_manifest()
    gate = verify_original_v4(manifest)
    output = Path(args.out)
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    if args.cpu_smoke:
        if args.device != "cpu":
            parser.error("--cpu-smoke requires --device cpu")
        if args.arm == "real_contact":
            parser.error("real_contact has no CPU surrogate")
        source_bundle = "cpu-smoke-unfrozen"
        shape = manifest["cpu_test_shape"]
        mode = "cpu_smoke"
    else:
        if args.device != "cuda":
            parser.error("production workloads require --device cuda")
        if args.max_seeds is not None:
            parser.error("production workloads forbid --max-seeds")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            parser.error("production workloads require exactly CUDA_VISIBLE_DEVICES=1")
        registered = _registered_output(manifest, args.arm, args.block)
        if output != registered.resolve():
            parser.error(f"output must be preregistered path {registered}")
        launch = verify_launch_receipt(args.arm, args.block, output)
        source_bundle = launch["source_bundle_canonical_sha256"]
        shape = manifest["shape"]
        mode = "production"
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    context = {
        "manifest": manifest,
        "gate": gate,
        "manifest_hashes": audit_manifest_hashes(manifest),
        "source_bundle": source_bundle,
        "shape": shape,
        "device": args.device,
        "mode": mode,
    }
    writer = AtomicJsonl(output)
    try:
        if args.arm == "replication":
            run_replication(context, writer, args.block, args.max_seeds)
        elif args.arm == "synthetic":
            run_synthetic(context, writer, args.max_seeds)
        else:
            run_real_contact(context, writer, args.max_seeds)
        writer.finish()
    except BaseException:
        writer.close_partial()
        print(
            f"campaign interrupted after {writer.count} rows; retained {writer.partial}",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 2
    print(f"wrote {writer.count} retained rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
