"""Collect calibration or validation metric records.

Built-in anchors and deliberately wrong candidates make the complete pipeline
runnable on CPU.  A future GPU adapter is a normal Python callable with the
signature documented by ``--callable``; importing it is deferred until use.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from . import SCHEMA_VERSION
from .distributions import make_inputs
from .metrics import compute_metrics
from .oracles import resolve_output
from .schema import (
    canonical_sha256,
    file_sha256,
    load_json,
    validate_manifest,
    write_jsonl,
)
from .seeds import tensor_seeds


Candidate = Callable[..., torch.Tensor]


def _load_callable(spec: str) -> tuple[Candidate, str | None]:
    """Load ``package.module:function`` only when an external candidate is used."""
    if ":" not in spec:
        raise ValueError("--callable must have the form package.module:function")
    module_name, function_name = spec.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    source = inspect.getsourcefile(function)
    return function, source


def _case_by_id(op_spec: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in op_spec["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def _candidate_output(
    candidate_kind: str,
    op: str,
    inputs: dict[str, torch.Tensor],
    reference: torch.Tensor,
    case: dict[str, Any],
    contract: dict[str, Any],
    external: Candidate | None,
) -> torch.Tensor:
    if external is not None:
        return external(op=op, inputs=inputs, case=case, contract=contract)
    if candidate_kind == "exact":
        return reference.clone()
    if candidate_kind == "zeros":
        return torch.zeros_like(reference)
    if candidate_kind == "row_reverse":
        if reference.ndim < 2:
            raise ValueError("row_reverse requires at least a 2-D output")
        return reference.flip(0)
    return resolve_output(candidate_kind, op, inputs, contract)


def collect_records(
    manifest: dict[str, Any],
    *,
    op: str,
    gate_id: str,
    split: str,
    candidate: str,
    device: str = "cpu",
    cpu_smoke: bool = False,
    max_seeds: int | None = None,
    seed_start: int = 0,
    case_ids: list[str] | None = None,
    callable_spec: str = "",
    candidate_name: str = "",
) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    op_spec = manifest["operations"][op]
    if gate_id not in op_spec["gates"]:
        raise ValueError(f"unknown gate {gate_id!r} for {op}")
    gate = op_spec["gates"][gate_id]
    shape = op_spec["cpu_test_shape"] if cpu_smoke else op_spec["shape"]
    split_count = manifest["split_counts"][split]
    if seed_start < 0 or seed_start >= split_count:
        raise ValueError(f"seed_start must be in [0, {split_count})")
    seed_stop = split_count
    if max_seeds is not None:
        if max_seeds <= 0:
            raise ValueError("max_seeds must be positive")
        seed_stop = min(seed_stop, seed_start + max_seeds)

    selected_cases = case_ids or [case["id"] for case in op_spec["cases"]]
    for case_id in selected_cases:
        _case_by_id(op_spec, case_id)

    external = None
    source_path = None
    candidate_entries: list[tuple[str, str]]
    role = "candidate"
    if candidate == "anchor":
        candidate_entries = [
            (kind, f"anchor:{kind}") for kind in gate["anchors"]
        ]
        role = "anchor"
    elif candidate == "callable":
        external, source_path = _load_callable(callable_spec)
        candidate_entries = [("callable", candidate_name or callable_spec)]
    else:
        candidate_entries = [(candidate, candidate_name or candidate)]

    manifest_hash = canonical_sha256(manifest)
    builtin_source = Path(__file__).with_name("oracles.py")
    records: list[dict[str, Any]] = []
    for case_id in selected_cases:
        case = _case_by_id(op_spec, case_id)
        for seed_index in range(seed_start, seed_stop):
            seed_map = tensor_seeds(manifest, op, case_id, split, seed_index)
            setup_started = time.perf_counter()
            setup_error = None
            inputs = None
            reference = None
            try:
                with torch.no_grad():
                    inputs = make_inputs(op, shape, case, seed_map, device=device)
                    reference = resolve_output(
                        gate["reference"]["kind"],
                        op,
                        inputs,
                        gate.get("contract", {}),
                    )
            except Exception as exc:  # noqa: BLE001 - repeated into each evidence row
                setup_error = exc
                setup_traceback = traceback.format_exc()
            setup_wall_s = time.perf_counter() - setup_started
            for actual_kind, name in candidate_entries:
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "robust_gate_measurement",
                    "campaign_id": manifest["campaign_id"],
                    "manifest_sha256": manifest_hash,
                    "op": op,
                    "gate_id": gate_id,
                    "case_id": case_id,
                    "split": split,
                    "seed_index": seed_index,
                    "tensor_seeds": seed_map,
                    "candidate": name,
                    "role": role,
                    "device": device,
                    "shape": dict(shape),
                    "reference_kind": gate["reference"]["kind"],
                    "contract": dict(gate.get("contract", {})),
                    "torch_version": torch.__version__,
                    "source_sha256": file_sha256(source_path or builtin_source),
                    "shared_input_reference_wall_s": setup_wall_s,
                }
                started = time.perf_counter()
                try:
                    if setup_error is not None:
                        raise RuntimeError(
                            f"input/reference setup failed: {type(setup_error).__name__}: "
                            f"{setup_error}"
                        ) from setup_error
                    with torch.no_grad():
                        output = _candidate_output(
                            actual_kind,
                            op,
                            inputs,  # type: ignore[arg-type]
                            reference,  # type: ignore[arg-type]
                            case,
                            gate.get("contract", {}),
                            external,
                        )
                        record["output_dtype"] = str(output.dtype)
                        record["output_shape"] = list(output.shape)
                        record["metrics"] = compute_metrics(
                            op,
                            reference,  # type: ignore[arg-type]
                            output,
                            inputs,  # type: ignore[arg-type]
                        )
                    record["ok"] = True
                except Exception as exc:  # noqa: BLE001 - failure is campaign data
                    record["ok"] = False
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["traceback"] = (
                        setup_traceback if setup_error is not None else traceback.format_exc()
                    )
                record["wall_s"] = time.perf_counter() - started
                records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--op", required=True, choices=("matmul", "fused_softmax", "sdpa"))
    parser.add_argument("--gate", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=("calibration", "tuning", "validation", "performance"),
    )
    parser.add_argument(
        "--candidate",
        default="anchor",
        help="anchor, exact, zeros, row_reverse, native_fp32, native_mixed, or callable",
    )
    parser.add_argument("--candidate-name", default="")
    parser.add_argument(
        "--callable",
        default="",
        help="deferred adapter package.module:function; receives op, inputs, case, contract",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--cases", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.candidate == "callable" and not args.callable:
        parser.error("--candidate callable requires --callable")
    manifest = load_json(args.manifest)
    records = collect_records(
        manifest,
        op=args.op,
        gate_id=args.gate,
        split=args.split,
        candidate=args.candidate,
        device=args.device,
        cpu_smoke=args.cpu_smoke,
        max_seeds=args.max_seeds,
        seed_start=args.seed_start,
        case_ids=[value for value in args.cases.split(",") if value] or None,
        callable_spec=args.callable,
        candidate_name=args.candidate_name,
    )
    write_jsonl(args.out, records)
    failed = sum(not record.get("ok", False) for record in records)
    print(f"wrote {len(records)} records to {args.out}; collection failures={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
