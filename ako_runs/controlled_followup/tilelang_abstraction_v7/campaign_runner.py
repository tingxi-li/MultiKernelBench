#!/usr/bin/env python3
"""Run hash-bound admission and fresh-process paired timing for fused-only v7."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import artifacts
from .protocol import (
    CAMPAIGN_ID,
    GPU0_UUID,
    NCU_FINAL_ELF,
    REPO,
    canonical_sha256,
    file_sha256,
    gpu_snapshot,
    load_lock,
    read_json,
    remote_lock_receipt,
    repo_path,
    result_root,
    stable_write,
    validate_timing_manifest,
)


GPU_LOCK_ID = f"multikernelbench-{GPU0_UUID}-timing"
GPU_LOCK_PATH = Path("/tmp") / f"{GPU_LOCK_ID}.lock"
GPU_LOCK_ENV = "TILELANG_ABSTRACTION_GPU0_LOCK_FD"


class JsonlWriter:
    def __init__(self, target: Path):
        self.target = target
        self.partial = target.with_name(target.name + ".partial")
        if target.exists() or self.partial.exists():
            raise FileExistsError(f"refusing existing gate evidence: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.partial.open("x", encoding="utf-8")
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1

    def finish(self) -> None:
        self.handle.close()
        os.replace(self.partial, self.target)

    def retain_partial(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


def _pair(campaign: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [pair for pair in campaign["pairs"] if pair["pair_id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"unknown pair: {pair_id}")
    return matches[0]


def _require_gpu0(gpu: int, expected: int) -> None:
    if gpu != expected or gpu != 0:
        raise RuntimeError("this campaign is bound to physical GPU 0")


def _require_admission_output(output: Path, filename: str) -> Path:
    resolved = output.resolve()
    try:
        relative = resolved.relative_to((Path(__file__).resolve().parent / "results").resolve())
    except ValueError as exc:
        raise RuntimeError("admission child output is outside the campaign result root") from exc
    if len(relative.parts) != 3 or relative.parts[1:] != ("admission", filename):
        raise RuntimeError("admission child output is outside its canonical tag position")
    result_root(relative.parts[0])
    return resolved


def _timing_tag_root(manifest: Path) -> Path:
    resolved = manifest.resolve()
    try:
        relative = resolved.relative_to((Path(__file__).resolve().parent / "results").resolve())
    except ValueError as exc:
        raise RuntimeError("timing manifest is outside the campaign result root") from exc
    if len(relative.parts) != 2 or relative.parts[1] != "timing_manifest.json":
        raise RuntimeError("timing manifest is outside its canonical tag position")
    return result_root(relative.parts[0])


def _material(materials: dict[str, Any], material_id: str) -> dict[str, Any]:
    return materials["entries"][material_id]


def _source_material_id(family: str) -> str:
    return {
        "matmul": "tilelang_matmul_abstraction",
        "fused_softmax": "tilelang_fused_abstraction",
        "sdpa": "sdpa_tilelang_abstraction",
    }[family]


def _idle_snapshot(gpu: int, phase: str) -> dict[str, Any]:
    run = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if run.returncode:
        raise RuntimeError(f"GPU idle query failed: {run.stderr.strip()}")
    processes = []
    for line in run.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2 or not fields[0].isdigit():
            raise RuntimeError(f"physical GPU {gpu} process query is malformed: {line!r}")
        processes.append({"pid": int(fields[0]), "used_memory_mib": fields[1]})
    value = {
        "phase": phase,
        "checked_at_unix": time.time(),
        "self_pid": os.getpid(),
        "compute_processes": processes,
        "stderr": run.stderr.strip(),
    }
    foreign = [row for row in processes if row["pid"] != value["self_pid"]]
    if foreign:
        raise RuntimeError(f"physical GPU {gpu} is busy: {foreign}")
    return value


def _acquire_gpu_lock():
    handle = GPU_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"host-global physical GPU 0 lock is held: {GPU_LOCK_PATH}") from None
    return handle


def _validate_inherited_gpu_lock() -> int:
    try:
        descriptor = int(os.environ.get(GPU_LOCK_ENV, ""))
        inherited = os.fstat(descriptor)
        expected = GPU_LOCK_PATH.stat()
    except (OSError, ValueError) as exc:
        raise RuntimeError("GPU child lacks the inherited physical-GPU0 lock") from exc
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("GPU child inherited another physical-GPU lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("GPU child does not share the held physical-GPU0 lock") from exc
    return descriptor


def _configure_gpu(gpu: int) -> None:
    artifacts.configure_compiler_root(gpu)
    os.environ.setdefault("MAX_JOBS", "4")


@contextmanager
def _isolated_ncu_environment():
    names = ("TMPDIR", "TMP", "TEMP")
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="tilelang-abstraction-v7-ncu-", dir="/tmp") as directory:
        os.environ.update({name: directory for name in names})
        try:
            yield Path(directory)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _parse_set(value: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in (value or "").split(","):
        if not part:
            continue
        key, raw = part.split("=", 1)
        if key.startswith("x_"):
            result.setdefault("extra", {})[key[2:]] = raw
        elif key in {"arith", "cast", "algo"}:
            result[key] = raw
        else:
            result[key] = int(raw)
    return result


def _build_arm(pair: dict[str, Any], side: str):
    arm = pair[side]
    if pair["family"] != "fused_softmax":
        raise RuntimeError("v7 is restricted to the frozen fused-softmax pair")
    artifacts.exact_import("common", "ako_runs/phase1_matmul/common.py")
    common2 = artifacts.exact_import("common2", "ako_runs/phase2_fused_sdpa/common2.py")
    variants2 = artifacts.exact_import("variants2", "ako_runs/phase2_fused_sdpa/variants2/__init__.py")
    cfg = common2.make_fused_config("tilelang_abs", arm["variant"], **_parse_set(arm["set"]))
    built = variants2.build("fused", cfg)
    artifacts.exact_import(
        "variants2.fused_tilelang_abstraction",
        "ako_runs/phase2_fused_sdpa/variants2/fused_tilelang_abstraction.py",
    )
    metadata = {
        "compile_s": float(built.compile_s),
        "config": cfg.to_dict(),
        "notes": built.notes,
        "reported_artifacts": {
            key: value for key, value in built.artifacts.items()
            if key in {"grid", "block", "n_regs", "n_spills", "shared_bytes", "backend_detail", "n_kernels", "tile"}
        },
    }
    return built, metadata


def _matmul_gate(pair: dict[str, Any], built, gate_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    import torch
    from ako_runs.controlled_followup.robust_gate.distributions import make_matmul_inputs
    from ako_runs.controlled_followup.robust_gate.oracles import contract_reference, semantic_reference
    from ako_runs.controlled_followup.robust_gate.seeds import tensor_seeds
    from ako_runs.controlled_followup.robust_gate.audits.matmul_v4_instrument_v1.runner import (
        compute_matmul_metrics_precomputed,
    )

    manifest = read_json(repo_path("ako_runs/controlled_followup/robust_gate/manifest_matmul_v4.json"))
    gate_spec = read_json(repo_path("ako_runs/controlled_followup/robust_gate/calibration/gate_spec_matmul_v4.json"))
    op = manifest["operations"]["matmul"]
    contract = gate_spec["gates"]["matmul/conformance_mixed"]["contract"]
    writer = JsonlWriter(gate_path)
    failed = 0
    maxima: dict[str, float] = {}
    try:
        for case in op["cases"]:
            for seed_index in range(manifest["split_counts"]["validation"]):
                seeds = tensor_seeds(manifest, "matmul", case["id"], "validation", seed_index)
                try:
                    inputs = make_matmul_inputs(op["shape"], case, seeds, "cuda:0")
                    with torch.no_grad():
                        output = built.run(inputs["a"].half().contiguous(), inputs["b"].half().contiguous())
                        semantic = semantic_reference("matmul", inputs)
                        conformance = contract_reference("matmul", inputs, contract)
                        scale = inputs["a"].double().abs() @ inputs["b"].double().abs()
                        torch.cuda.synchronize()
                    references = {"conformance_mixed": conformance, "semantic_mixed": semantic}
                    for gate_id, reference in references.items():
                        metrics = compute_matmul_metrics_precomputed(reference, output, scale)
                        thresholds = gate_spec["gates"][f"matmul/{gate_id}"]["thresholds"]
                        failures = [name for name, spec in thresholds.items() if metrics.get(name, math.inf) > spec["value"]]
                        failed += bool(failures)
                        for name, value in metrics.items():
                            maxima[f"{gate_id}/{name}"] = max(maxima.get(f"{gate_id}/{name}", float("-inf")), float(value))
                        writer.write({
                            **binding,
                            "record_type": "tilelang_abstraction_v7_matmul_gate",
                            "op": "matmul", "gate_id": gate_id, "case_id": case["id"],
                            "seed_index": seed_index, "tensor_seeds": seeds,
                            "metrics": metrics, "threshold_failures": failures,
                            "ok": True, "gate_pass": not failures,
                        })
                    del inputs, output, semantic, conformance, scale
                except Exception as exc:  # retain both planned gate outcomes
                    failed += 2
                    for gate_id in pair["gate"]["gate_ids"]:
                        writer.write({
                            **binding,
                            "record_type": "tilelang_abstraction_v7_matmul_gate",
                            "op": "matmul", "gate_id": gate_id, "case_id": case["id"],
                            "seed_index": seed_index, "tensor_seeds": seeds,
                            "ok": False, "gate_pass": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        })
        writer.finish()
    except BaseException:
        writer.retain_partial()
        raise
    expected = len(op["cases"]) * manifest["split_counts"]["validation"] * 2
    return {
        "complete": writer.count == expected,
        "expected_records": expected,
        "observed_records": writer.count,
        "failed_records": failed,
        "full_gate_pass": writer.count == expected and failed == 0,
        "maxima": maxima,
    }


def _fused_gate(pair: dict[str, Any], built, metadata: dict[str, Any], gate_path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    adapter = artifacts.exact_robust_adapter()
    context = adapter.load_repository()
    recovery = artifacts.exact_recovery_audit()

    def execute(inputs, prepared):
        if "x_fp16" not in prepared:
            prepared["x_fp16"] = inputs["x"].half().contiguous()
        return built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

    plan = adapter.CandidatePlan(
        candidate=f"{CAMPAIGN_ID}:{pair['pair_id']}:{binding['side']}",
        job=None,
        job_sha256=None,
        config=metadata["config"],
        build_metadata={**metadata, **binding},
        execute=execute,
    )
    rows, live = [], None
    for case_id in context.cases:
        for seed_index in range(64):
            prior = live
            evaluated, live = adapter.evaluate_case_seed(
                context, [plan], case_id=case_id, split="validation", seed_index=seed_index, device="cuda:0"
            )
            if prior is not None:
                del prior
            for row in evaluated[plan.candidate]:
                row.update(binding)
                rows.append(row)
    writer = JsonlWriter(gate_path)
    try:
        for row in rows:
            writer.write(row)
        writer.finish()
    except BaseException:
        writer.retain_partial()
        raise
    return recovery.fixed_gate_summary(context, rows)


def arm_admit(args) -> int:
    _validate_inherited_gpu_lock()
    _require_admission_output(args.output, f"{args.pair_id}__{args.side}.json")
    artifact_root = args.output.parent / (args.output.stem + "_artifact")
    if args.output.exists():
        raise FileExistsError("refusing existing arm admission output")
    artifacts.prepare_environment(artifact_root, "admit", args.gpu)
    campaign, materials, lock = load_lock(args.lock, check_gpu=True, check_remote=True)
    _require_gpu0(args.gpu, campaign["hardware"]["timing_gpu"])
    pair = _pair(campaign, args.pair_id)
    _require_admission_output(args.output, f"{pair['pair_id']}__{args.side}.json")
    if not pair["gate"]["available"]:
        raise RuntimeError("unavailable current gate must be classified before arm execution")
    idle_pre = _idle_snapshot(args.gpu, "admission_child_pre")
    _configure_gpu(args.gpu)
    base = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "gate_manifest_sha256": _material(materials, pair["gate"]["manifest_material_id"])["sha256"],
        "gate_spec_sha256": _material(materials, pair["gate"]["spec_material_id"])["sha256"],
        "pair_id": pair["pair_id"],
        "family": pair["family"],
        "side": args.side,
        "variant": pair[args.side]["variant"],
        "set": pair[args.side]["set"],
        "physical_gpu": args.gpu,
        "gpu": gpu_snapshot(args.gpu),
        "remote_authorization": remote_lock_receipt(args.lock),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_idle_preflight": idle_pre,
        "parent_pid": os.getppid(),
        "process_pid": os.getpid(),
    }
    gate_path = args.output.with_suffix(".gate.jsonl")
    started = time.time()
    try:
        built, metadata = _build_arm(pair, args.side)
        admitted_cache = artifacts.cache_snapshot(artifact_root)
        binding = {**base, "implementation_sha256": admitted_cache["files_sha256"]}
        summary = _fused_gate(pair, built, metadata, gate_path, binding)
        if artifacts.cache_snapshot(artifact_root) != admitted_cache:
            raise RuntimeError("admitted TileLang cache mutated during correctness gating")
        record = {
            **base,
            "implementation_sha256": binding["implementation_sha256"],
            "implementation_source_sha256": _material(materials, _source_material_id(pair["family"]))["sha256"],
            "admitted_cache": admitted_cache,
            "artifact_root": str(artifact_root.resolve().relative_to(REPO.resolve())),
            "metadata": metadata,
            "gate_path": str(gate_path.resolve().relative_to(REPO.resolve())),
            "gate_sha256": file_sha256(gate_path),
            "gate_summary": summary,
            "terminal_outcome": "GATE_PASSED" if summary["full_gate_pass"] else "GATE_FAILED",
        }
    except Exception as exc:
        record = {
            **base,
            "terminal_outcome": "BUILD_OR_GATE_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    record["toolchain"] = lock["toolchain"]
    try:
        record["gpu_idle_postflight"] = _idle_snapshot(args.gpu, "admission_child_post")
    except Exception as exc:
        record["terminal_outcome"] = "BUILD_OR_GATE_FAILED"
        record["gpu_idle_postflight_error"] = f"{type(exc).__name__}: {exc}"
    record["t_start"] = started
    record["t_end"] = time.time()
    record["wall_s"] = record["t_end"] - started
    stable_write(args.output, record)
    return 0 if record["terminal_outcome"] == "GATE_PASSED" else 2


def _ncu_module():
    artifacts.exact_import("common", "ako_runs/phase1_matmul/common.py")
    artifacts.exact_import("common2", "ako_runs/phase2_fused_sdpa/common2.py")
    return artifacts.exact_import("ncu_collect2", "ako_runs/phase2_fused_sdpa/ncu_collect2.py")


def _profile_command(
    pair: dict[str, Any], side: str, gpu: int, lock: Path,
    admission_receipt: Path, load_receipt: Path,
) -> list[str]:
    metrics = _ncu_module().METRICS
    return [
        str(NCU_FINAL_ELF), "--csv", "--page", "raw",
        "--target-processes", "all", "--launch-count", "400",
        "--metrics", ",".join(metrics), sys.executable, "-m",
        "ako_runs.controlled_followup.tilelang_abstraction_v7.campaign_runner",
        "profile-target", "--pair-id", pair["pair_id"], "--side", side,
        "--gpu", str(gpu), "--lock", str(lock.resolve()),
        "--admission-receipt", str(admission_receipt.resolve()), "--output", str(load_receipt.resolve()),
    ]


def _fold_profile(pair: dict[str, Any], side: str, stdout: str, iters: int = 2) -> dict[str, Any]:
    ncu = _ncu_module()
    rows = ncu.parse_csv(stdout)
    if not rows:
        return {"op": "fused", "dsl": "tilelang_abs", "variant": pair[side]["variant"],
                "set": pair[side]["set"], "ok": False, "error": "ncu emitted no metric rows"}
    by_kernel: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        name = row.get("Kernel Name") or row.get('"Kernel Name"') or "?"
        metrics: dict[str, float] = {}
        for metric in ncu.METRICS:
            raw = row.get(metric)
            if raw is None:
                raw = next((value for key, value in row.items() if key.strip('"') == metric), None)
            value = ncu.to_float(raw)
            if value is not None:
                metrics[ncu.SHORT.get(metric, metric)] = value
        if not metrics:
            continue
        short = name.split("(")[0].strip()[:70]
        key = ncu.kernel_id(short, metrics)
        entry = by_kernel.setdefault(key, {"name": short, "order": len(by_kernel), "launches": []})
        entry["launches"].append(metrics)
    kernels = []
    for key, entry in by_kernel.items():
        last = entry["launches"][-1]
        last.update({
            "kernel": entry["name"], "n_launches": len(entry["launches"]),
            "launch_order": entry["order"], "grid_block_regs": list(key[1:]),
        })
        last["is_setup"] = any(hint.lower() in entry["name"].lower() for hint in ncu.SETUP_HINTS) or len(entry["launches"]) < iters + 1
        last["dram_total_GB"] = ((last.get("dram_read_B", 0) or 0) + (last.get("dram_write_B", 0) or 0)) / 1e9
        kernels.append(last)
    kernels.sort(key=lambda value: value["launch_order"])
    algorithm = [kernel for kernel in kernels if not kernel["is_setup"]]
    return {
        "op": "fused", "dsl": "tilelang_abs", "variant": pair[side]["variant"],
        "set": pair[side]["set"], "kernels": kernels,
        "n_algo_kernels": len(algorithm),
        "algo_dram_GB": sum(kernel["dram_total_GB"] for kernel in algorithm),
        "setup_dram_GB": sum(kernel["dram_total_GB"] for kernel in kernels if kernel["is_setup"]),
        "ok": bool(algorithm),
    }


def profile_target(args) -> int:
    _validate_inherited_gpu_lock()
    _require_admission_output(args.output, f"{args.pair_id}__{args.side}.profile.load.json")
    admission = read_json(args.admission_receipt)
    artifact_root = repo_path(admission.get("artifact_root", ""))
    expected_artifact_root = args.admission_receipt.parent / (args.admission_receipt.stem + "_artifact")
    expected_output = args.admission_receipt.with_name(args.admission_receipt.stem + ".profile.load.json")
    if artifact_root.resolve() != expected_artifact_root.resolve() or args.output.resolve() != expected_output.resolve():
        raise RuntimeError("profile target paths differ from the canonical admission arm")
    artifacts.prepare_environment(artifact_root, "load_only", args.gpu)
    campaign, _materials, _lock = load_lock(args.lock)
    pair = _pair(campaign, args.pair_id)
    if (
        admission.get("campaign_lock_sha256") != file_sha256(args.lock)
        or admission.get("pair_id") != pair["pair_id"]
        or admission.get("side") != args.side
        or admission.get("terminal_outcome") != "GATE_PASSED"
    ):
        raise RuntimeError("profile target requires its matching passed admission")
    cache = admission.get("admitted_cache", {})
    before = artifacts.cache_snapshot(artifact_root)
    if before != cache:
        raise RuntimeError("profile target cache differs from admission")
    with artifacts.load_only_guards(cache) as evidence:
        built, _metadata = _build_arm(pair, args.side)
        artifacts.validate_load_evidence(evidence, cache)
        common2 = artifacts.exact_import("common2", "ako_runs/phase2_fused_sdpa/common2.py")
        x, weight, bias = common2.fused_inputs(seed=0, dist="rand")
        if str(built.x_dtype) == "torch.float16":
            x = x.half().contiguous()
        for _ in range(3):
            built.run(x, weight, bias)
        import torch
        torch.cuda.synchronize()
    after = artifacts.cache_snapshot(artifact_root)
    if after != before:
        raise RuntimeError("admitted cache mutated in profile target")
    artifacts.validate_load_evidence(evidence, cache)
    stable_write(args.output, {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "pair_id": pair["pair_id"], "side": args.side,
        "admitted_cache_sha256": cache["files_sha256"],
        "cache_before": before, "cache_after": after,
        "load_evidence": evidence,
    })
    return 0


def arm_profile(args) -> int:
    lock_fd = _validate_inherited_gpu_lock()
    _require_admission_output(args.output, f"{args.pair_id}__{args.side}.profile.json")
    expected_admission = args.output.with_name(f"{args.pair_id}__{args.side}.json")
    if args.admission_receipt.resolve() != expected_admission.resolve():
        raise RuntimeError("profile admission receipt is outside its canonical arm")
    admission = read_json(args.admission_receipt)
    artifact_root = repo_path(admission.get("artifact_root", ""))
    artifacts.prepare_environment(artifact_root, "load_only", args.gpu)
    with _isolated_ncu_environment():
        campaign, _materials, lock = load_lock(args.lock, check_gpu=True, check_remote=True)
        _require_gpu0(args.gpu, campaign["hardware"]["timing_gpu"])
        pair = _pair(campaign, args.pair_id)
        arm = pair[args.side]
        if (
            admission.get("campaign_lock_sha256") != file_sha256(args.lock)
            or admission.get("pair_id") != pair["pair_id"]
            or admission.get("side") != args.side
            or admission.get("terminal_outcome") != "GATE_PASSED"
        ):
            raise RuntimeError("profile requires the matching passed admission receipt")
        cache = admission.get("admitted_cache", {})
        expected_identity = artifacts.validate_cache_receipt(cache, artifact_root)
        idle_pre = _idle_snapshot(args.gpu, "profile_child_pre")
        _configure_gpu(args.gpu)
        raw = args.output.with_suffix(".raw.json")
        load_receipt = args.output.with_suffix(".load.json")
        if args.output.exists() or raw.exists() or load_receipt.exists():
            raise FileExistsError("refusing existing profile output")
        command = _profile_command(pair, args.side, args.gpu, args.lock, args.admission_receipt, load_receipt)
        started = time.time()
        run = subprocess.run(
            command, cwd=REPO, env=dict(os.environ), pass_fds=(lock_fd,), check=False,
            capture_output=True, text=True, timeout=7200,
        )
        profile_gpu = gpu_snapshot(args.gpu)
    profile_record = _fold_profile(pair, args.side, run.stdout)
    stable_write(raw, {"metrics_requested": _ncu_module().METRICS, "records": [profile_record]})
    load_value, artifact_error = None, None
    try:
        load_value = read_json(load_receipt)
        artifacts.validate_load_evidence(load_value.get("load_evidence"), cache)
        if load_value.get("cache_before") != cache or load_value.get("cache_after") != cache:
            raise RuntimeError("profile target cache receipt differs")
        if artifacts.cache_snapshot(artifact_root) != cache:
            raise RuntimeError("profile mutated the admitted cache")
    except Exception as exc:  # fail closed while retaining the profiler receipt
        artifact_error = f"{type(exc).__name__}: {exc}"
    artifact_match = load_value is not None and artifact_error is None
    idle_post, idle_error = None, None
    try:
        idle_post = _idle_snapshot(args.gpu, "profile_child_post")
    except Exception as exc:
        idle_error = f"{type(exc).__name__}: {exc}"
    ok = (
        run.returncode == 0
        and profile_record.get("ok") is True
        and artifact_match
        and idle_error is None
    )
    receipt = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "toolchain": lock["toolchain"],
        "pair_id": pair["pair_id"], "family": pair["family"], "side": args.side,
        "variant": arm["variant"], "set": arm["set"],
        "physical_gpu": args.gpu, "gpu": profile_gpu,
        "remote_authorization": remote_lock_receipt(args.lock),
        "command": command, "returncode": run.returncode, "ok": ok,
        "raw_path": str(raw.resolve().relative_to(REPO.resolve())) if raw.is_file() else None,
        "raw_sha256": file_sha256(raw) if raw.is_file() else None,
        "admitted_cache": cache,
        "artifact_load_path": str(load_receipt.resolve().relative_to(REPO.resolve())) if load_receipt.is_file() else None,
        "artifact_load_sha256": file_sha256(load_receipt) if load_receipt.is_file() else None,
        "expected_artifact_identity_sha256": expected_identity,
        "implementation_sha256": expected_identity if artifact_match else None,
        "artifact_identity_match": artifact_match,
        "artifact_error": artifact_error,
        "gpu_idle_preflight": idle_pre,
        "gpu_idle_postflight": idle_post,
        "gpu_idle_postflight_error": idle_error,
        "parent_pid": os.getppid(), "process_pid": os.getpid(),
        "t_start": started, "t_end": time.time(),
        "stdout_tail": run.stdout[-4000:], "stderr_tail": run.stderr[-4000:],
    }
    stable_write(args.output, receipt)
    return 0 if ok else 2


def _invoke(
    action: str,
    *,
    pair_id: str,
    side: str,
    gpu: int,
    lock: Path,
    output: Path,
    admission_receipt: Path | None = None,
    lock_fd: int,
) -> int:
    command = [
        sys.executable, "-m", "ako_runs.controlled_followup.tilelang_abstraction_v7.campaign_runner",
        action, "--pair-id", pair_id, "--side", side, "--gpu", str(gpu), "--lock", str(lock), "--output", str(output),
    ]
    if admission_receipt is not None:
        command.extend(["--admission-receipt", str(admission_receipt)])
    env = dict(os.environ)
    env[GPU_LOCK_ENV] = str(lock_fd)
    run = subprocess.run(command, cwd=REPO, env=env, pass_fds=(lock_fd,), check=False)
    return run.returncode


def _run_admission(args, gpu_lock) -> int:
    _configure_gpu(args.gpu)
    campaign, _materials, lock = load_lock(args.lock, check_gpu=True, check_remote=True)
    _require_gpu0(args.gpu, campaign["hardware"]["timing_gpu"])
    stage_idle_pre = _idle_snapshot(args.gpu, "admission_stage_pre")
    root = result_root(args.tag) / "admission"
    if root.exists():
        raise FileExistsError("admission attempt already exists; use a new result tag")
    root.mkdir(parents=True)
    stable_write(root / "launch_receipt.json", {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "toolchain": lock["toolchain"],
        "expected_pair_ids": [pair["pair_id"] for pair in campaign["pairs"]],
        "physical_gpu": args.gpu,
        "gpu": gpu_snapshot(args.gpu),
        "gpu_lock_id": GPU_LOCK_ID,
        "gpu_idle_preflight": stage_idle_pre,
        "remote_authorization": remote_lock_receipt(args.lock),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    for pair in campaign["pairs"]:
        if not pair["gate"]["available"]:
            path = root / f"{pair['pair_id']}__unavailable.json"
            stable_write(path, {
                "campaign_id": CAMPAIGN_ID, "campaign_lock_sha256": file_sha256(args.lock),
                "pair_id": pair["pair_id"], "terminal_outcome": "CURRENT_GATE_UNAVAILABLE",
                "reason": pair["gate"]["reason"], "build_attempted": False, "timing_authorized": False,
            })
            continue
        for side in ("high", "low"):
            gate = root / f"{pair['pair_id']}__{side}.json"
            profile = root / f"{pair['pair_id']}__{side}.profile.json"
            rc = _invoke(
                "arm-admit", pair_id=pair["pair_id"], side=side, gpu=args.gpu,
                lock=args.lock, output=gate, lock_fd=gpu_lock.fileno(),
            )
            if rc == 0:
                _invoke(
                    "arm-profile", pair_id=pair["pair_id"], side=side, gpu=args.gpu,
                    lock=args.lock, output=profile, admission_receipt=gate,
                    lock_fd=gpu_lock.fileno(),
                )
    from .analyze import admission_artifact_hashes, admission_summary

    hashes = admission_artifact_hashes(root, campaign)
    stable_write(root / "run_status.json", {
        "schema_version": 1,
        "record_type": "tilelang_abstraction_v7_admission_run_status",
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "complete": True,
        "artifact_sha256": hashes,
        "artifact_bundle_sha256": canonical_sha256(hashes),
        "gpu_idle_postflight": _idle_snapshot(args.gpu, "admission_stage_post"),
        "launch_receipt_sha256": file_sha256(root / "launch_receipt.json"),
    })
    summary = admission_summary(root, args.lock)
    stable_write(root / "summary.json", summary)
    print(json.dumps(summary["census"], sort_keys=True))
    return 0


def run_admission(args) -> int:
    gpu_lock = _acquire_gpu_lock()
    try:
        return _run_admission(args, gpu_lock)
    finally:
        gpu_lock.close()


def _arm_time_command(
    row: dict[str, Any], gpu: int, lock: Path, manifest: Path,
    admission: Path, output: Path,
) -> list[str]:
    return [
        sys.executable, "-m", "ako_runs.controlled_followup.tilelang_abstraction_v7.campaign_runner",
        "arm-time", "--gpu", str(gpu), "--lock", str(lock.resolve()),
        "--manifest", str(manifest.resolve()), "--admission", str(admission.resolve()),
        "--row-json", json.dumps(row, sort_keys=True), "--output", str(output.resolve()),
    ]


def arm_time(args) -> int:
    _validate_inherited_gpu_lock()
    row = json.loads(args.row_json)
    tag_root = _timing_tag_root(args.manifest)
    gate_receipt = tag_root / "admission" / f"{row['pair_id']}__{row['implementation_side']}.json"
    gate = read_json(gate_receipt)
    artifact_root = repo_path(gate.get("artifact_root", ""))
    artifacts.prepare_environment(artifact_root, "load_only", args.gpu)
    campaign, _materials, lock = load_lock(args.lock, check_gpu=True, check_remote=True)
    _require_gpu0(args.gpu, campaign["hardware"]["timing_gpu"])
    from .analyze import load_verified_admission

    if args.admission.resolve() != (tag_root / "admission" / "summary.json").resolve():
        raise RuntimeError("timing admission is outside the manifest tag")
    manifest = read_json(args.manifest)
    admission = load_verified_admission(args.admission, args.lock)
    validate_timing_manifest(manifest, campaign, file_sha256(args.lock), admission)
    if row not in manifest["rows"] or row["physical_gpu"] != args.gpu:
        raise RuntimeError("timing row is not in the frozen admission-bound manifest")
    position = manifest["rows"].index(row) + 1
    expected_output = tag_root / "timing" / "raw" / f"{position:04d}__{row['row_id']}.json"
    if args.output.resolve() != expected_output.resolve():
        raise RuntimeError("timing output is outside its canonical manifest position")
    pair = _pair(campaign, row["pair_id"])
    idle_pre = _idle_snapshot(args.gpu, "timing_child_pre")
    _configure_gpu(args.gpu)
    command = _arm_time_command(row, args.gpu, args.lock, args.manifest, args.admission, args.output)
    started = time.time()
    admission_pair = next(value for value in admission["pairs"] if value["pair_id"] == pair["pair_id"])
    expected_identity = admission_pair["artifact_identity_sha256"][row["implementation_side"]]
    cache = gate.get("admitted_cache", {})
    times: list[float] = []
    load_evidence, artifact_error = None, None
    try:
        if artifacts.validate_cache_receipt(cache, artifact_root) != expected_identity:
            raise RuntimeError("timing cache identity differs from admitted summary")
        before = artifacts.cache_snapshot(artifact_root)
        with artifacts.load_only_guards(cache) as load_evidence:
            built, _metadata = _build_arm(pair, row["implementation_side"])
            artifacts.validate_load_evidence(load_evidence, cache)
            common2 = artifacts.exact_import("common2", "ako_runs/phase2_fused_sdpa/common2.py")
            runner2 = artifacts.exact_import("runner2", "ako_runs/phase2_fused_sdpa/runner2.py")
            dist, seed = ("rand", 0) if row["distribution"] == "positive" else ("randn", 2026073101)
            x, weight, bias = common2.fused_inputs(seed=seed, dist=dist)
            if str(built.x_dtype) == "torch.float16":
                x = x.half().contiguous()
            measured, _warmups = runner2.time_kernel3(
                built.run, x, weight, bias, row["trials"], row["warmup_seconds"], True
            )
            times = [float(value) for value in measured]
        if artifacts.cache_snapshot(artifact_root) != before:
            raise RuntimeError("admitted cache mutated during timing")
        artifacts.validate_load_evidence(load_evidence, cache)
    except Exception as exc:
        artifact_error = f"{type(exc).__name__}: {exc}"
    valid = len(times) == row["trials"] and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in times)
    primary = times[campaign["inference"]["primary_trial_start"]:campaign["inference"]["primary_trial_stop"]] if valid else []
    artifact_match = load_evidence is not None and artifact_error is None
    idle_post, idle_error = None, None
    try:
        idle_post = _idle_snapshot(args.gpu, "timing_child_post")
    except Exception as exc:
        idle_error = f"{type(exc).__name__}: {exc}"
    record = {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock), "manifest_sha256": canonical_sha256(manifest),
        "manifest_row": row, "manifest_row_sha256": canonical_sha256(row),
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "toolchain": lock["toolchain"],
        "gpu": gpu_snapshot(args.gpu), "physical_gpu": args.gpu,
        "remote_authorization": remote_lock_receipt(args.lock),
        "gpu_idle_preflight": idle_pre,
        "gpu_idle_postflight": idle_post,
        "gpu_idle_postflight_error": idle_error,
        "parent_pid": os.getppid(), "process_pid": os.getpid(),
        "command": command, "cwd": str(REPO.resolve()),
        "returncode": 0 if valid and artifact_match else 2,
        "admitted_cache": cache,
        "artifact_load": load_evidence,
        "expected_artifact_identity_sha256": expected_identity,
        "implementation_sha256": expected_identity if artifact_match else None,
        "artifact_identity_match": artifact_match,
        "artifact_error": artifact_error,
        "times_ms": [float(value) for value in times] if valid else [],
        "primary_tail_median_ms": statistics.median(primary) if primary else None,
        "full_median_ms": statistics.median(times) if valid else None,
        "first_decile_median_ms": statistics.median(times[:10]) if valid else None,
        "last_decile_median_ms": statistics.median(times[-10:]) if valid else None,
        "ok": valid and artifact_match and idle_error is None,
        "stdout_tail": "", "stderr_tail": artifact_error or "",
        "t_start": started, "t_end": time.time(),
    }
    stable_write(args.output, record)
    return 0 if record["ok"] else 2


def _run_timing(args, gpu_lock) -> int:
    _configure_gpu(args.gpu)
    campaign, _materials, lock = load_lock(args.lock, check_gpu=True, check_remote=True)
    _require_gpu0(args.gpu, campaign["hardware"]["timing_gpu"])
    from .analyze import load_verified_admission

    tag_root = _timing_tag_root(args.manifest)
    if tag_root != result_root(args.tag):
        raise RuntimeError("timing tag differs from the manifest tag")
    if args.admission.resolve() != (tag_root / "admission" / "summary.json").resolve():
        raise RuntimeError("timing admission is outside the manifest tag")
    manifest = read_json(args.manifest)
    admission = load_verified_admission(args.admission, args.lock)
    validate_timing_manifest(manifest, campaign, file_sha256(args.lock), admission)
    stage_idle_pre = _idle_snapshot(args.gpu, "timing_stage_pre")
    stage_root = tag_root / "timing"
    if stage_root.exists():
        raise FileExistsError("timing attempt already exists; use a new result tag")
    root = stage_root / "raw"
    root.mkdir(parents=True)
    positions = stage_root / "position_receipts"
    positions.mkdir()
    stable_write(stage_root / "launch_receipt.json", {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "dependency_bundle_sha256": lock["dependency_bundle_sha256"],
        "toolchain": lock["toolchain"],
        "manifest_sha256": canonical_sha256(manifest),
        "manifest_file_sha256": file_sha256(args.manifest),
        "admission_summary_sha256": canonical_sha256(admission),
        "admission_summary_file_sha256": file_sha256(args.admission),
        "expected_records": len(manifest["rows"]),
        "physical_gpu": args.gpu,
        "gpu": gpu_snapshot(args.gpu),
        "gpu_lock_id": GPU_LOCK_ID,
        "gpu_idle_preflight": stage_idle_pre,
        "remote_authorization": remote_lock_receipt(args.lock),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    previous_completed = 0
    predecessor_sha = None
    for index, row in enumerate(manifest["rows"], 1):
        output = root / f"{index:04d}__{row['row_id']}.json"
        outcome = _invoke_time(row, args, output, gpu_lock.fileno())
        position_path = positions / f"{index:04d}__{row['row_id']}.json"
        idle_after, idle_error = None, None
        try:
            idle_after = _idle_snapshot(args.gpu, "timing_parent_after_child")
        except Exception as exc:
            idle_error = f"{type(exc).__name__}: {exc}"
        position = {
            "schema_version": 1,
            "record_type": "tilelang_abstraction_v7_position_receipt",
            "campaign_id": CAMPAIGN_ID,
            "manifest_sha256": canonical_sha256(manifest),
            "row_id": row["row_id"],
            "global_position": index,
            "predecessor_position_receipt_sha256": predecessor_sha,
            "child_pid": outcome["pid"],
            "child_launched_unix_ns": outcome["launched"],
            "child_completed_unix_ns": outcome["completed"],
            "returncode": outcome["returncode"],
            "raw_path": str(output.resolve().relative_to(REPO.resolve())),
            "raw_sha256": file_sha256(output) if output.is_file() else None,
            "gpu_idle_after_child": idle_after,
            "gpu_postflight_error": idle_error,
        }
        stable_write(position_path, position)
        predecessor_sha = file_sha256(position_path)
        print(
            f"[{index}/{len(manifest['rows'])}] {row['pair_id']} "
            f"{row['distribution']} b{row['block']} {row['label']} "
            f"-> {outcome['returncode']}",
            flush=True,
        )
        if outcome["returncode"] or not output.is_file() or idle_error is not None or outcome["launched"] < previous_completed:
            raise RuntimeError("timing child failed; this attempt is terminal and requires a new tag")
        previous_completed = outcome["completed"]
    from .analyze import timing_artifact_hashes, timing_summary

    hashes = timing_artifact_hashes(stage_root, manifest)
    stable_write(stage_root / "run_status.json", {
        "schema_version": 1,
        "record_type": "tilelang_abstraction_v7_timing_run_status",
        "campaign_id": CAMPAIGN_ID,
        "campaign_lock_sha256": file_sha256(args.lock),
        "complete": True,
        "expected_records": len(manifest["rows"]),
        "observed_records": len(manifest["rows"]),
        "artifact_sha256": hashes,
        "artifact_bundle_sha256": canonical_sha256(hashes),
        "gpu_idle_postflight": _idle_snapshot(args.gpu, "timing_stage_post"),
        "launch_receipt_sha256": file_sha256(stage_root / "launch_receipt.json"),
    })
    summary = timing_summary(root, args.manifest, args.admission, args.lock)
    stable_write(stage_root / "summary.json", summary)
    return 0


def run_timing(args) -> int:
    gpu_lock = _acquire_gpu_lock()
    try:
        return _run_timing(args, gpu_lock)
    finally:
        gpu_lock.close()


def _invoke_time(row: dict[str, Any], args, output: Path, lock_fd: int) -> dict[str, int]:
    command = _arm_time_command(row, args.gpu, args.lock, args.manifest, args.admission, output)
    env = dict(os.environ)
    env[GPU_LOCK_ENV] = str(lock_fd)
    launched = time.time_ns()
    process = subprocess.Popen(command, cwd=REPO, env=env, pass_fds=(lock_fd,))
    returncode = process.wait()
    return {
        "pid": process.pid,
        "launched": launched,
        "completed": time.time_ns(),
        "returncode": returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admit", "arm-admit", "arm-profile", "profile-target", "timing", "arm-time"))
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-receipt", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pair-id")
    parser.add_argument("--row-json")
    parser.add_argument("--side", choices=("high", "low"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.action == "admit":
        if not args.tag:
            parser.error("admit requires --tag")
        return run_admission(args)
    if args.action == "arm-admit":
        return arm_admit(args)
    if args.action == "arm-profile":
        if args.admission_receipt is None:
            parser.error("arm-profile requires --admission-receipt")
        return arm_profile(args)
    if args.action == "profile-target":
        if args.admission_receipt is None or args.output is None or not args.pair_id or not args.side:
            parser.error("profile-target requires --admission-receipt, --output, --pair-id, and --side")
        return profile_target(args)
    if args.action == "timing":
        if not args.tag or args.manifest is None or args.admission is None:
            parser.error("timing requires --tag, --manifest, and --admission")
        return run_timing(args)
    if args.output is None or args.manifest is None or args.admission is None or args.row_json is None:
        parser.error("arm-time requires --output, --manifest, --admission, and --row-json")
    return arm_time(args)


if __name__ == "__main__":
    raise SystemExit(main())
