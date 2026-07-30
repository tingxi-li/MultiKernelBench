#!/usr/bin/env python3
"""Run the complete frozen fused-v2 correctness adjudication for closure v2."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ako_runs.controlled_followup.fused_closure_v2 import candidates, core, provenance
else:  # pragma: no cover - module invocation follows this branch
    from . import candidates, core, provenance


ROBUST_MANIFEST_PATH = (
    core.REPO_ROOT / "ako_runs/controlled_followup/robust_gate/manifest.json"
)
GATE_SPEC_PATH = core.REPO_ROOT / (
    "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_fused_v2.json"
)
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _command(argv: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001 - environment evidence is best effort
        return {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}


def _environment(torch, physical_gpu: int) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "captured_utc": _utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_requested": physical_gpu,
        "logical_device": 0,
        "device_name": props.name,
        "compute_capability": [props.major, props.minor],
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "nvidia_smi": _command(["nvidia-smi", "-i", str(physical_gpu), "-q"]),
        "nvcc": _command(["/usr/local/cuda-13.1/bin/nvcc", "--version"]),
        "git_head": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
    }


def _idle_preflight(physical_gpu: int) -> dict[str, Any]:
    query = _command(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if query.get("returncode") != 0:
        raise core.ClosureError(f"GPU process preflight failed: {query}")
    if query.get("stdout", "").strip():
        raise core.ClosureError(
            "preregistered GPU is busy; refusing correctness launch: "
            + query["stdout"]
        )
    return query


def _load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = core.read_json(ROBUST_MANIFEST_PATH)
    gate_spec = core.read_json(GATE_SPEC_PATH)
    if gate_spec.get("manifest_sha256") != core.canonical_sha256(manifest):
        raise core.ClosureError("fused-v2 gate is not bound to the robust manifest")
    campaign = core.load_campaign()
    cases = [
        case["id"]
        for case in manifest["operations"]["fused_softmax"]["cases"]
    ]
    if cases != campaign["gate"]["case_ids"]:
        raise core.ClosureError("campaign cases differ from robust manifest order")
    expected_gates = {
        f"fused_softmax/{gate_id}" for gate_id in campaign["gate"]["gate_ids"]
    }
    if set(gate_spec.get("gates", {})) != expected_gates:
        raise core.ClosureError("campaign gate IDs differ from frozen gate spec")
    if manifest["split_counts"].get("validation") != 64:
        raise core.ClosureError("robust manifest validation count is not 64")
    return manifest, gate_spec


def _robust_imports():
    controlled = core.REPO_ROOT / "ako_runs/controlled_followup"
    if str(controlled) not in sys.path:
        sys.path.insert(0, str(controlled))
    import torch
    from robust_gate.distributions import make_inputs
    from robust_gate.metrics import compute_metrics
    from robust_gate.oracles import native_mixed_reference, resolve_output
    from robust_gate.seeds import tensor_seeds
    from robust_gate.validate import validate_records

    return (
        torch,
        make_inputs,
        compute_metrics,
        native_mixed_reference,
        resolve_output,
        tensor_seeds,
        validate_records,
    )


def _threshold_failures(
    gate: dict[str, Any], metrics: dict[str, float]
) -> list[str]:
    failures = []
    for metric, threshold in gate["thresholds"].items():
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            failures.append(f"{metric}=missing/nonfinite")
        elif value > threshold["value"]:
            failures.append(f"{metric}={value:.9g} > {threshold['value']:.9g}")
    return failures


def _binding(
    campaign: dict[str, Any],
    source_receipt: dict[str, Any],
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_canonical_sha256": core.canonical_sha256(campaign),
        "source_receipt_sha256": core.sha256_file(core.SOURCE_RECEIPT_PATH),
        "robust_manifest_file_sha256": core.sha256_file(ROBUST_MANIFEST_PATH),
        "robust_manifest_canonical_sha256": core.canonical_sha256(manifest),
        "gate_spec_file_sha256": core.sha256_file(GATE_SPEC_PATH),
        "gate_spec_canonical_sha256": core.canonical_sha256(gate_spec),
        "candidate_sha256": source_receipt["candidate_sha256"],
        "case_ids": campaign["gate"]["case_ids"],
        "gate_ids": campaign["gate"]["gate_ids"],
        "split": "validation",
        "seed_indices": list(range(64)),
    }


def _record_base(
    *,
    campaign: dict[str, Any],
    definition: dict[str, Any],
    binding: dict[str, Any],
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
    gate_id: str,
    case_id: str,
    seed_index: int,
    seed_map: dict[str, int],
    build_metadata: dict[str, Any],
    shape: dict[str, int],
) -> dict[str, Any]:
    gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
    return {
        "schema_version": "1.0",
        "record_type": "robust_gate_measurement",
        "campaign_id": manifest["campaign_id"],
        "closure_campaign_id": campaign["campaign_id"],
        "closure_campaign_sha256": binding["campaign_canonical_sha256"],
        "source_receipt_sha256": binding["source_receipt_sha256"],
        "manifest_sha256": binding["robust_manifest_canonical_sha256"],
        "manifest_file_sha256": binding["robust_manifest_file_sha256"],
        "gate_spec_sha256": binding["gate_spec_canonical_sha256"],
        "gate_spec_file_sha256": binding["gate_spec_file_sha256"],
        "op": "fused_softmax",
        "gate_id": gate_id,
        "case_id": case_id,
        "split": "validation",
        "seed_index": seed_index,
        "tensor_seeds": seed_map,
        "candidate": definition["candidate_id"],
        "candidate_definition_sha256": binding["candidate_sha256"][
            definition["candidate_id"]
        ],
        "candidate_definition": definition,
        "contract_adjudication": definition["contract_adjudication"],
        "structural_mismatches": definition["structural_mismatches"],
        "role": "candidate",
        "device": "cuda:0",
        "shape": shape,
        "reference_kind": gate["reference"]["kind"],
        "contract": gate["contract"],
        "build_metadata": build_metadata,
    }


def _failure_pair(
    *,
    error: str,
    trace: str,
    **base_args: Any,
) -> list[dict[str, Any]]:
    records = []
    campaign = base_args["campaign"]
    for gate_id in campaign["gate"]["gate_ids"]:
        record = _record_base(gate_id=gate_id, **base_args)
        record.update(
            {
                "ok": False,
                "gate_pass": False,
                "error": error,
                "traceback": trace,
                "threshold_failures": [error],
            }
        )
        records.append(record)
    return records


def _expected_keys(campaign: dict[str, Any], case_id: str, seed_index: int):
    return {
        (candidate_id, gate_id, case_id, seed_index)
        for candidate_id in campaign["candidate_order"]
        for gate_id in campaign["gate"]["gate_ids"]
    }


def _validate_bundle(
    path: Path,
    value: Any,
    *,
    campaign: dict[str, Any],
    binding: dict[str, Any],
    case_id: str,
    seed_index: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("record_type") != "closure_gate_bundle":
        raise core.ClosureError(f"invalid gate bundle envelope: {path}")
    if value.get("binding") != binding:
        raise core.ClosureError(f"foreign gate bundle binding: {path}")
    records = value.get("records")
    if not isinstance(records, list):
        raise core.ClosureError(f"gate bundle has no records list: {path}")
    keys = {
        (
            row.get("candidate"),
            row.get("gate_id"),
            row.get("case_id"),
            row.get("seed_index"),
        )
        for row in records
    }
    expected = _expected_keys(campaign, case_id, seed_index)
    if keys != expected or len(records) != len(expected):
        raise core.ClosureError(f"gate bundle coverage differs from plan: {path}")
    for row in records:
        candidate_id = row["candidate"]
        if row.get("source_receipt_sha256") != binding["source_receipt_sha256"]:
            raise core.ClosureError(f"gate record source binding differs: {path}")
        if row.get("candidate_definition_sha256") != binding["candidate_sha256"].get(
            candidate_id
        ):
            raise core.ClosureError(f"gate record candidate binding differs: {path}")
    if path.is_file() and path.read_bytes() != core.stable_json_bytes(value):
        raise core.ClosureError(f"gate bundle is not stable JSON: {path}")
    return records


def _cpu_smoke(
    campaign: dict[str, Any],
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
) -> int:
    (
        torch,
        make_inputs,
        compute_metrics,
        native_mixed_reference,
        resolve_output,
        tensor_seeds,
        _validate_records,
    ) = _robust_imports()
    shape = manifest["operations"]["fused_softmax"]["cpu_test_shape"]
    contract = gate_spec["gates"]["fused_softmax/conformance_mixed"]["contract"]
    rows = []
    with torch.no_grad():
        for case_id in campaign["gate"]["case_ids"]:
            seeds = tensor_seeds(manifest, "fused_softmax", case_id, "tuning", 0)
            case = next(
                case
                for case in manifest["operations"]["fused_softmax"]["cases"]
                if case["id"] == case_id
            )
            inputs = make_inputs("fused_softmax", shape, case, seeds, device="cpu")
            output = native_mixed_reference("fused_softmax", inputs, contract)
            for gate_id in campaign["gate"]["gate_ids"]:
                gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                reference = resolve_output(
                    gate["reference"]["kind"],
                    "fused_softmax",
                    inputs,
                    gate["contract"],
                )
                metrics = compute_metrics("fused_softmax", reference, output, inputs)
                failures = _threshold_failures(gate, metrics)
                rows.append(
                    {
                        "case_id": case_id,
                        "gate_id": gate_id,
                        "pass": not failures,
                        "threshold_failures": failures,
                    }
                )
    result = {
        "record_type": "fused_closure_v2_cpu_smoke",
        "shape": shape,
        "n_checks": len(rows),
        "success": all(row["pass"] for row in rows),
        "checks": rows,
    }
    print(core.stable_json_bytes(result).decode("utf-8"), end="")
    return 0 if result["success"] else 1


def _run_gpu(
    campaign: dict[str, Any],
    source_receipt: dict[str, Any],
    manifest: dict[str, Any],
    gate_spec: dict[str, Any],
    *,
    physical_gpu: int,
    tag: str,
    idle_preflight: dict[str, Any],
) -> int:
    (
        torch,
        make_inputs,
        compute_metrics,
        _native_mixed_reference,
        resolve_output,
        tensor_seeds,
        validate_records,
    ) = _robust_imports()
    if not torch.cuda.is_available():
        raise core.ClosureError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise core.ClosureError(
            "closure child must see exactly one logical GPU through CUDA_VISIBLE_DEVICES"
        )
    binding = _binding(campaign, source_receipt, manifest, gate_spec)
    result_root = core.RESULTS_ROOT / tag
    raw_root = result_root / "raw"
    launch_path = result_root / "gate_launch_receipt.json"
    summary_path = result_root / "gate_summary.json"
    planned = [
        {"case_id": case_id, "seed_index": seed_index}
        for case_id in campaign["gate"]["case_ids"]
        for seed_index in range(64)
    ]
    if not launch_path.exists() and result_root.exists() and any(result_root.iterdir()):
        raise core.ClosureError("new gate tag directory is not empty")
    if launch_path.exists():
        launch = core.read_json(launch_path)
        if (
            launch.get("binding") != binding
            or launch.get("planned_bundles") != planned
            or launch_path.read_bytes() != core.stable_json_bytes(launch)
        ):
            raise core.ClosureError("existing gate launch receipt differs from plan")
    else:
        launch = {
            "schema_version": 1,
            "record_type": "fused_closure_v2_gate_launch",
            "created_utc": _utc_now(),
            "tag": tag,
            "binding": binding,
            "planned_bundles": planned,
            "expected_bundles": len(planned),
            "expected_records": len(planned)
            * len(campaign["candidate_order"])
            * len(campaign["gate"]["gate_ids"]),
            "compute_processes_before_launch": idle_preflight,
            "environment": _environment(torch, physical_gpu),
        }
        core.atomic_json(launch_path, launch)
    launch_sha = core.sha256_file(launch_path)

    if summary_path.exists():
        summary = core.read_json(summary_path)
        if (
            summary.get("binding") != binding
            or summary.get("gate_launch_receipt_sha256") != launch_sha
            or summary_path.read_bytes() != core.stable_json_bytes(summary)
        ):
            raise core.ClosureError("existing gate summary differs from launch")
        recovered_records = []
        recovered_hashes = {}
        for item in planned:
            relative = (
                Path("raw")
                / item["case_id"]
                / f"seed{item['seed_index']:03d}.json"
            )
            path = result_root / relative
            if not path.is_file():
                raise core.ClosureError(f"gate summary raw bundle is missing: {path}")
            bundle = core.read_json(path)
            recovered_records.extend(
                _validate_bundle(
                    path,
                    bundle,
                    campaign=campaign,
                    binding=binding,
                    case_id=item["case_id"],
                    seed_index=item["seed_index"],
                )
            )
            recovered_hashes[str(relative)] = core.sha256_file(path)
        if recovered_hashes != summary.get("raw_bundle_sha256") or core.canonical_sha256(
            recovered_records
        ) != summary.get("raw_records_canonical_sha256"):
            raise core.ClosureError("gate summary no longer matches raw bundles")
        print(
            f"existing complete gate summary: {summary_path} "
            f"eligible={sum(x['same_contract_eligible'] for x in summary['adjudications'])}"
        )
        return 0

    definitions = {item["candidate_id"]: item for item in campaign["candidates"]}
    built: dict[str, candidates.BuiltCandidate] = {}
    build_errors: dict[str, tuple[str, str]] = {}
    for candidate_id in campaign["candidate_order"]:
        started = time.perf_counter()
        try:
            built[candidate_id] = candidates.build_candidate(
                definitions[candidate_id], expected_candidate_id=candidate_id
            )
            torch.cuda.synchronize()
            print(
                f"built {candidate_id} in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - retained as campaign evidence
            build_errors[candidate_id] = (
                f"BuildError: {type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
            print(f"build failed {candidate_id}: {exc}", flush=True)

    shape = dict(manifest["operations"]["fused_softmax"]["shape"])
    cases = {
        case["id"]: case
        for case in manifest["operations"]["fused_softmax"]["cases"]
    }
    previous_inputs = None
    all_records: list[dict[str, Any]] = []
    bundle_hashes: dict[str, str] = {}
    with torch.no_grad():
        for ordinal, item in enumerate(planned, 1):
            case_id, seed_index = item["case_id"], item["seed_index"]
            relative = Path("raw") / case_id / f"seed{seed_index:03d}.json"
            path = result_root / relative
            if path.exists():
                bundle = core.read_json(path)
                records = _validate_bundle(
                    path,
                    bundle,
                    campaign=campaign,
                    binding=binding,
                    case_id=case_id,
                    seed_index=seed_index,
                )
                all_records.extend(records)
                bundle_hashes[str(relative)] = core.sha256_file(path)
                print(f"gate {ordinal:03d}/{len(planned)} resume {case_id}/{seed_index}")
                continue

            seed_map = tensor_seeds(
                manifest, "fused_softmax", case_id, "validation", seed_index
            )
            records: list[dict[str, Any]] = []
            setup_error: tuple[str, str] | None = None
            current_inputs = None
            setup_started = time.perf_counter()
            try:
                # previous_inputs remains live during allocation. This prevents
                # allocator pointer reuse from defeating address-keyed W caches.
                current_inputs = make_inputs(
                    "fused_softmax",
                    shape,
                    cases[case_id],
                    seed_map,
                    device="cuda:0",
                )
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 - retained evidence
                setup_error = (
                    f"InputSetupError: {type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                )
            setup_wall_s = time.perf_counter() - setup_started

            references: dict[str, Any] = {}
            reference_errors: dict[str, tuple[str, str]] = {}
            reference_wall_s: dict[str, float] = {}
            if setup_error is None:
                assert current_inputs is not None
                for gate_id in campaign["gate"]["gate_ids"]:
                    gate = gate_spec["gates"][f"fused_softmax/{gate_id}"]
                    started = time.perf_counter()
                    try:
                        references[gate_id] = resolve_output(
                            gate["reference"]["kind"],
                            "fused_softmax",
                            current_inputs,
                            gate["contract"],
                        )
                        torch.cuda.synchronize()
                    except Exception as exc:  # noqa: BLE001 - retained evidence
                        reference_errors[gate_id] = (
                            f"ReferenceError: {type(exc).__name__}: {exc}",
                            traceback.format_exc(),
                        )
                    reference_wall_s[gate_id] = time.perf_counter() - started

            for candidate_id in campaign["candidate_order"]:
                definition = definitions[candidate_id]
                metadata = (
                    built[candidate_id].build_metadata
                    if candidate_id in built
                    else {"build_failed": True}
                )
                base_args = {
                    "campaign": campaign,
                    "definition": definition,
                    "binding": binding,
                    "manifest": manifest,
                    "gate_spec": gate_spec,
                    "case_id": case_id,
                    "seed_index": seed_index,
                    "seed_map": seed_map,
                    "build_metadata": metadata,
                    "shape": shape,
                }
                error_pair = build_errors.get(candidate_id) or setup_error
                if error_pair is not None:
                    records.extend(
                        _failure_pair(error=error_pair[0], trace=error_pair[1], **base_args)
                    )
                    continue
                assert current_inputs is not None
                started = time.perf_counter()
                try:
                    x32 = current_inputs["x"]
                    x16 = x32.half().contiguous()
                    output = built[candidate_id].run(
                        x32,
                        x16,
                        current_inputs["weight"],
                        current_inputs["bias"],
                    )
                    torch.cuda.synchronize()
                    candidate_wall_s = time.perf_counter() - started
                except Exception as exc:  # noqa: BLE001 - retained evidence
                    records.extend(
                        _failure_pair(
                            error=f"CandidateError: {type(exc).__name__}: {exc}",
                            trace=traceback.format_exc(),
                            **base_args,
                        )
                    )
                    continue

                for gate_id in campaign["gate"]["gate_ids"]:
                    record = _record_base(gate_id=gate_id, **base_args)
                    record.update(
                        {
                            "shared_input_wall_s": setup_wall_s,
                            "shared_reference_wall_s": reference_wall_s,
                            "candidate_wall_s": candidate_wall_s,
                            "output_dtype": str(output.dtype),
                            "output_shape": list(output.shape),
                        }
                    )
                    if gate_id in reference_errors:
                        error, trace = reference_errors[gate_id]
                        record.update(
                            {
                                "ok": False,
                                "gate_pass": False,
                                "error": error,
                                "traceback": trace,
                                "threshold_failures": [error],
                            }
                        )
                    else:
                        metric_started = time.perf_counter()
                        try:
                            metrics = compute_metrics(
                                "fused_softmax",
                                references[gate_id],
                                output,
                                current_inputs,
                            )
                            failures = _threshold_failures(
                                gate_spec["gates"][f"fused_softmax/{gate_id}"],
                                metrics,
                            )
                            record.update(
                                {
                                    "ok": True,
                                    "metrics": metrics,
                                    "gate_pass": not failures,
                                    "threshold_failures": failures,
                                    "metric_wall_s": time.perf_counter()
                                    - metric_started,
                                }
                            )
                        except Exception as exc:  # noqa: BLE001 - retained evidence
                            error = f"MetricError: {type(exc).__name__}: {exc}"
                            record.update(
                                {
                                    "ok": False,
                                    "gate_pass": False,
                                    "error": error,
                                    "traceback": traceback.format_exc(),
                                    "threshold_failures": [error],
                                    "metric_wall_s": time.perf_counter()
                                    - metric_started,
                                }
                            )
                    records.append(record)
                del output, x16

            bundle = {
                "schema_version": 1,
                "record_type": "closure_gate_bundle",
                "binding": binding,
                "case_id": case_id,
                "seed_index": seed_index,
                "records": records,
            }
            _validate_bundle(
                path,
                bundle,
                campaign=campaign,
                binding=binding,
                case_id=case_id,
                seed_index=seed_index,
            )
            core.atomic_json(path, bundle)
            all_records.extend(records)
            bundle_hashes[str(relative)] = core.sha256_file(path)
            if current_inputs is not None:
                previous_inputs = current_inputs
            print(
                f"gate {ordinal:03d}/{len(planned)} wrote {case_id}/{seed_index}",
                flush=True,
            )

    expected_count = (
        len(planned)
        * len(campaign["candidate_order"])
        * len(campaign["gate"]["gate_ids"])
    )
    if len(all_records) != expected_count:
        raise core.ClosureError(
            f"gate record count {len(all_records)} differs from {expected_count}"
        )
    validation = validate_records(
        manifest, gate_spec, all_records, allow_incomplete=False
    )
    groups = {
        (group["candidate"], group["gate_id"]): group
        for group in validation["groups"]
    }
    adjudications = []
    for definition in campaign["candidates"]:
        candidate_id = definition["candidate_id"]
        gate_results = {
            gate_id: groups[(candidate_id, gate_id)]
            for gate_id in campaign["gate"]["gate_ids"]
        }
        empirical = all(value["success"] for value in gate_results.values())
        structural = (
            definition["contract_adjudication"] == "fused_v2_required"
            and not definition["structural_mismatches"]
        )
        adjudications.append(
            {
                "candidate_id": candidate_id,
                "contract_adjudication": definition["contract_adjudication"],
                "structural_mismatches": definition["structural_mismatches"],
                "structurally_conforming": structural,
                "empirical_gate_success": empirical,
                "same_contract_eligible": structural and empirical,
                "gates": gate_results,
            }
        )
    summary = {
        "schema_version": 1,
        "record_type": "fused_closure_v2_gate_summary",
        "created_utc": _utc_now(),
        "tag": tag,
        "binding": binding,
        "gate_launch_receipt_sha256": launch_sha,
        "gate_launch_receipt_path": str(launch_path.relative_to(core.REPO_ROOT)),
        "coverage_complete": True,
        "expected_records": expected_count,
        "observed_records": len(all_records),
        "performance_launch_allowed": True,
        "universal_empirical_success": validation["success"],
        "adjudications": adjudications,
        "robust_validation": validation,
        "raw_bundle_sha256": bundle_hashes,
        "raw_records_canonical_sha256": core.canonical_sha256(all_records),
    }
    core.atomic_json(summary_path, summary)
    print(
        f"wrote {summary_path}; same-contract eligible "
        f"{sum(item['same_contract_eligible'] for item in adjudications)}/9",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-smoke", action="store_true")
    mode.add_argument("--gpu", type=int)
    parser.add_argument("--tag", default="gate_v1")
    args = parser.parse_args()
    if not TAG_PATTERN.fullmatch(args.tag):
        parser.error("--tag must contain only letters, digits, dot, underscore, hyphen")
    source_receipt = provenance.verify_receipt()
    campaign = core.load_campaign()
    manifest, gate_spec = _load_frozen()
    if args.cpu_smoke:
        return _cpu_smoke(campaign, manifest, gate_spec)
    if args.gpu != campaign["performance_protocol"]["physical_gpu"]:
        parser.error("--gpu must equal preregistered physical GPU 0")
    # This assignment must precede the first torch import in this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    idle_preflight = _idle_preflight(args.gpu)
    return _run_gpu(
        campaign,
        source_receipt,
        manifest,
        gate_spec,
        physical_gpu=args.gpu,
        tag=args.tag,
        idle_preflight=idle_preflight,
    )


if __name__ == "__main__":
    raise SystemExit(main())
