#!/usr/bin/env python3
"""Measured 19-grid support probes and receipt-derived resolution."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PHASE1 = REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = REPO_ROOT / "ako_runs/phase2_fused_sdpa"
FUSED_GRID = HERE.parent / "fused_grid"
for path in (str(HERE), str(FUSED_GRID), str(PHASE2), str(PHASE1), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


PROBE_KEYS = ("cuda_noptx_register", "triton_smem")
GRID_IDS = tuple(f"g{index:02d}" for index in range(19))
TERMINAL_OUTCOMES = ("BUILD_FAILED", "LAUNCH_FAILED", "GATE_FAILED", "GATE_PASSED")


def _module(name: str):
    return importlib.import_module(f"{__package__}.{name}" if __package__ else name)


def _normalize_probe_key(probe_key: str | tuple[str, str]) -> str:
    aliases = {
        ("register_fused", "cuda_noptx"): "cuda_noptx_register",
        ("register_common_postprocess", "cuda_noptx"): "cuda_noptx_register",
        ("smem_staged", "triton"): "triton_smem",
    }
    key = aliases.get(probe_key, probe_key) if isinstance(probe_key, tuple) else probe_key
    if key not in PROBE_KEYS:
        raise KeyError(f"unknown support probe {probe_key!r}")
    return key


def config_for_probe(probe_key: str | tuple[str, str], cell: dict[str, Any]):
    key = _normalize_probe_key(probe_key)
    expected_lane = "cuda_noptx" if key == "cuda_noptx_register" else "triton"
    if cell.get("lane") != expected_lane or cell.get("grid_id") not in GRID_IDS:
        raise ValueError(f"cell does not match {key}: {cell.get('cell_id')!r}")
    if __package__:
        from .core import parse_set
    else:
        from core import parse_set
    import common2

    overrides = parse_set(cell["origin_job"]["set"])
    overrides.setdefault("extra", {})["epilogue"] = (
        "regs" if key == "cuda_noptx_register" else "smem"
    )
    overrides["extra"]["wcache"] = "cached"
    cfg = common2.make_fused_config(expected_lane, "GBGS", **overrides)
    if (cfg.M, cfg.K, cfg.N) != (1024, 8192, 8192):
        raise ValueError("support probe left the frozen fused shape")
    return cfg


def build_probe_candidate(probe_key: str | tuple[str, str], cfg):
    """Production dispatch used by both probing and resolved campaign cells."""
    key = _normalize_probe_key(probe_key)
    if key == "cuda_noptx_register":
        return _module("cuda_noptx_register").build(cfg)
    return _module("triton_smem").build(cfg)


def source_for_probe(probe_key: str | tuple[str, str], cfg) -> tuple[str, str]:
    key = _normalize_probe_key(probe_key)
    if key == "cuda_noptx_register":
        return _module("cuda_noptx_register").make_source(cfg)["source"], ".cu"
    return _module("triton_smem").make_source(cfg), ".py"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_text(path, payload)


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and len(value) > 4096:
        encoded = value.encode()
        return {"bytes": len(encoded), "sha256": _sha256_bytes(encoded)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _paths(root: Path, probe_key: str, grid_id: str, suffix: str) -> dict[str, Path]:
    base = root.resolve() / probe_key
    return {
        "source": base / "sources" / f"{grid_id}{suffix}",
        "diagnostics": base / "diagnostics" / f"{grid_id}.txt",
        "gate": base / "gate" / f"{grid_id}.jsonl",
        "receipt": base / "attempts" / f"{grid_id}.json",
        "index": base / "index.json",
    }


def _retained(
    path: Path,
    probe_key: str,
    grid_id: str,
    *,
    cell_sha256: str | None = None,
    probe_binding: dict[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("probe_key") != probe_key
        or value.get("grid_id") != grid_id
        or value.get("terminal_outcome") not in TERMINAL_OUTCOMES
        or (cell_sha256 is not None and value.get("cell_sha256") != cell_sha256)
        or (probe_binding is not None and value.get("probe_binding") != probe_binding)
        or (source_sha256 is not None and value.get("source_sha256") != source_sha256)
    ):
        raise RuntimeError(f"foreign or incomplete retained probe receipt: {path}")
    for prefix in ("source", "diagnostics"):
        artifact = Path(value[f"{prefix}_path"])
        if not artifact.is_absolute():
            artifact = REPO_ROOT / artifact
        if not artifact.is_file() or _file_sha256(artifact) != value[f"{prefix}_sha256"]:
            raise RuntimeError(f"retained {prefix} changed: {artifact}")
    if value.get("gate_attempted"):
        gate = Path(value["gate_path"])
        if not gate.is_absolute():
            gate = REPO_ROOT / gate
        if not gate.is_file() or _file_sha256(gate) != value.get("gate_sha256"):
            raise RuntimeError(f"retained gate evidence changed: {gate}")
    return value


def _plan(context, cell: dict[str, Any], cfg, built, source_sha256: str):
    import torch
    import robust_adapter as adapter

    metadata = {
        "probe_key": cell["support_probe_key"],
        "reported_compile_s": built.compile_s,
        "n_kernels": built.n_kernels,
        "notes": built.notes,
        "artifacts": _compact(built.artifacts),
        "retained_source_sha256": source_sha256,
    }

    def execute(inputs, prepared):
        if "x_fp16" not in prepared:
            prepared["x_fp16"] = inputs["x"].half().contiguous()
        return built.run(prepared["x_fp16"], inputs["weight"], inputs["bias"])

    if built.x_dtype != torch.float16 or built.n_kernels != 2:
        raise ValueError("support candidate must implement the fp16 two-kernel full operation")
    return adapter.CandidatePlan(
        candidate=f"crossed-v2-probe:{cell['support_probe_key']}:{cell['grid_id']}",
        job=cell["origin_job"],
        job_sha256=cell["origin_job_sha256"],
        config=cfg.to_dict(),
        build_metadata=metadata,
        execute=execute,
    )


def run_probe(
    cell: dict[str, Any],
    output_root: str | os.PathLike[str],
    *,
    probe_binding: dict[str, Any],
    context=None,
) -> dict[str, Any]:
    """Build and run one grid attempt through every frozen validation gate."""
    probe_key = _normalize_probe_key(cell.get("support_probe_key"))
    grid_id = cell.get("grid_id")
    cfg = config_for_probe(probe_key, cell)
    source, suffix = source_for_probe(probe_key, cfg)
    paths = _paths(Path(output_root), probe_key, grid_id, suffix)
    source_bytes = source.encode()
    source_sha256 = _sha256_bytes(source_bytes)
    cell_sha256 = _canonical_sha256(cell)
    retained = _retained(
        paths["receipt"],
        probe_key,
        grid_id,
        cell_sha256=cell_sha256,
        probe_binding=probe_binding,
        source_sha256=source_sha256,
    )
    if retained is not None:
        return retained
    _atomic_text(paths["source"], source)
    diagnostic = io.StringIO()
    base = {
        "campaign_id": "fused-epilogue-crossed-v2",
        "cell_id": cell["cell_id"],
        "cell_sha256": cell_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid_id": grid_id,
        "origin_job_sha256": cell["origin_job_sha256"],
        "probe_key": probe_key,
        "probe_binding": probe_binding,
        "schema_version": 1,
        "source_path": _relative(paths["source"]),
        "source_sha256": source_sha256,
    }
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(diagnostic), contextlib.redirect_stderr(diagnostic):
            built = build_probe_candidate(probe_key, cfg)
            ptxas_log = built.artifacts.get("ptxas_log")
            if isinstance(ptxas_log, str) and ptxas_log:
                diagnostic.write("\n--- retained ptxas diagnostics ---\n")
                diagnostic.write(ptxas_log)
                diagnostic.write("\n--- end ptxas diagnostics ---\n")
            if context is None:
                import robust_adapter as adapter

                context = adapter.load_repository()
            plan = _plan(context, cell, cfg, built, source_sha256)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        limitation = bool(getattr(exc, "capability_limitation", False))
        diagnostic.write(traceback.format_exc())
        _atomic_text(paths["diagnostics"], diagnostic.getvalue())
        receipt = {
            **base,
            "build_attempted": True,
            "build_wall_s": time.perf_counter() - started,
            "capability_limitation": limitation,
            "diagnostics_path": _relative(paths["diagnostics"]),
            "diagnostics_sha256": _file_sha256(paths["diagnostics"]),
            "failure_signature": _sha256_bytes(message.encode()),
            "gate_attempted": False,
            "terminal_outcome": "BUILD_FAILED",
            "terminal_reason": message,
        }
        _atomic_json(paths["receipt"], receipt)
        return receipt

    import robust_adapter as adapter

    rows: list[dict[str, Any]] = []
    live_inputs = None
    try:
        with contextlib.redirect_stdout(diagnostic), contextlib.redirect_stderr(diagnostic):
            for case_id in context.adapter["robust_gate"]["case_ids"]:
                for seed_index in range(64):
                    previous = live_inputs
                    evaluated, live_inputs = adapter.evaluate_case_seed(
                        context,
                        [plan],
                        case_id=case_id,
                        split="validation",
                        seed_index=seed_index,
                        device="cuda:0",
                    )
                    if previous is not None:
                        del previous
                    rows.extend(evaluated[plan.candidate])
    except Exception as exc:
        diagnostic.write(traceback.format_exc())
        rows.append({"ok": False, "gate_pass": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if live_inputs is not None:
            del live_inputs

    _atomic_jsonl(paths["gate"], rows)
    _atomic_text(paths["diagnostics"], diagnostic.getvalue())
    expected_rows = 4 * 64 * 2
    execution_failed = any(row.get("ok") is not True for row in rows)
    gate_passed = (
        len(rows) == expected_rows
        and not execution_failed
        and all(row.get("gate_pass") is True for row in rows)
    )
    outcome = (
        "LAUNCH_FAILED" if execution_failed else "GATE_PASSED" if gate_passed else "GATE_FAILED"
    )
    receipt = {
        **base,
        "build_attempted": True,
        "build_metadata": _compact(plan.build_metadata),
        "build_wall_s": time.perf_counter() - started,
        "diagnostics_path": _relative(paths["diagnostics"]),
        "diagnostics_sha256": _file_sha256(paths["diagnostics"]),
        "gate_attempted": True,
        "gate_path": _relative(paths["gate"]),
        "gate_sha256": _file_sha256(paths["gate"]),
        "gate_summary": {
            "complete": len(rows) == expected_rows,
            "expected_records": expected_rows,
            "failed_records": sum(
                row.get("ok") is not True or row.get("gate_pass") is not True for row in rows
            ),
            "observed_records": len(rows),
        },
        "terminal_outcome": outcome,
        "terminal_reason": "full frozen gate passed" if gate_passed else "build launched but gate did not pass",
    }
    _atomic_json(paths["receipt"], receipt)
    return receipt


def derive_resolution(attempts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(attempts)
    if len(rows) != 19 or {row.get("grid_id") for row in rows} != set(GRID_IDS):
        return {"status": "unresolved", "reason": "19-grid attempt set is incomplete"}
    if any(row.get("terminal_outcome") == "GATE_PASSED" for row in rows):
        return {"status": "supported", "reason": "at least one measured grid passed the full frozen gate"}
    signatures = {row.get("failure_signature") for row in rows}
    if (
        all(row.get("terminal_outcome") == "BUILD_FAILED" for row in rows)
        and all(row.get("capability_limitation") is True for row in rows)
        and None not in signatures
        and len(signatures) == 1
    ):
        return {
            "status": "unsupported",
            "reason": "all 19 builds retained one identical compiler/API limitation",
            "failure_signature": next(iter(signatures)),
        }
    return {"status": "unresolved", "reason": "no gate pass and failures do not share one limitation"}


def load_result_index(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate an index and re-derive its resolution from bound receipts."""
    index_path = Path(path).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    key = _normalize_probe_key(index.get("probe_key"))
    if __package__:
        from .core import PROBE_LOCK_PATH, file_sha256, read_json
    else:
        from core import PROBE_LOCK_PATH, file_sha256, read_json

    lock = read_json(PROBE_LOCK_PATH)
    binding = index.get("probe_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("probe_lock_sha256") != file_sha256(PROBE_LOCK_PATH)
        or binding.get("source_bundle_sha256") != lock.get("source_bundle_sha256")
        or binding.get("frozen_gate") != lock.get("frozen_gate")
    ):
        raise RuntimeError("probe index is not bound to the current probe lock")
    attempts = index.get("attempts")
    if index.get("complete") is not True or not isinstance(attempts, list) or len(attempts) != 19:
        raise RuntimeError(f"incomplete probe index: {index_path}")
    receipts = []
    for attempt in attempts:
        receipt_path = Path(attempt["receipt_path"])
        if not receipt_path.is_absolute():
            receipt_path = REPO_ROOT / receipt_path
        if not receipt_path.is_file() or _file_sha256(receipt_path) != attempt["receipt_sha256"]:
            raise RuntimeError(f"probe receipt changed: {receipt_path}")
        receipt = _retained(
            receipt_path,
            key,
            attempt["grid_id"],
            probe_binding=binding,
        )
        if receipt is None or receipt["terminal_outcome"] != attempt["terminal_outcome"]:
            raise RuntimeError(f"probe index disagrees with receipt: {receipt_path}")
        for prefix in ("source", "diagnostics"):
            if (
                receipt.get(f"{prefix}_path") != attempt.get(f"{prefix}_path")
                or receipt.get(f"{prefix}_sha256") != attempt.get(f"{prefix}_sha256")
            ):
                raise RuntimeError(f"probe index lost its {prefix} binding: {receipt_path}")
        receipts.append(receipt)
    resolution = derive_resolution(receipts)
    if resolution != index.get("resolution"):
        raise RuntimeError("probe index resolution is not receipt-derived")
    return index


def run_probe_grid(
    probe_key: str,
    cells: Iterable[dict[str, Any]],
    output_root: str | os.PathLike[str],
    *,
    probe_binding: dict[str, Any],
    context=None,
) -> dict[str, Any]:
    key = _normalize_probe_key(probe_key)
    canonical_strategy = "register_fused" if key == "cuda_noptx_register" else "smem_staged"
    selected = [
        cell
        for cell in cells
        if cell.get("support_probe_key") == key and cell.get("strategy") == canonical_strategy
    ]
    by_grid = {cell["grid_id"]: cell for cell in selected}
    if len(selected) != 19 or set(by_grid) != set(GRID_IDS):
        raise ValueError(f"{key} requires exactly one cell per frozen grid")
    paths = _paths(Path(output_root), key, "g00", ".txt")
    if paths["index"].is_file():
        return load_result_index(paths["index"])
    if context is None:
        import robust_adapter as adapter

        context = adapter.load_repository()
    attempts = [
        run_probe(
            by_grid[grid],
            output_root,
            probe_binding=probe_binding,
            context=context,
        )
        for grid in GRID_IDS
    ]
    resolution = derive_resolution(attempts)
    index = {
        "attempts": [
            {
                "diagnostics_path": row["diagnostics_path"],
                "diagnostics_sha256": row["diagnostics_sha256"],
                "grid_id": row["grid_id"],
                "receipt_path": _relative(_paths(Path(output_root), key, row["grid_id"], ".cu" if key == "cuda_noptx_register" else ".py")["receipt"]),
                "receipt_sha256": _file_sha256(_paths(Path(output_root), key, row["grid_id"], ".cu" if key == "cuda_noptx_register" else ".py")["receipt"]),
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
                "terminal_outcome": row["terminal_outcome"],
            }
            for row in attempts
        ],
        "campaign_id": "fused-epilogue-crossed-v2",
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe_key": key,
        "probe_binding": probe_binding,
        "resolution": resolution,
        "schema_version": 1,
    }
    _atomic_json(paths["index"], index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-key", choices=PROBE_KEYS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--allow-busy", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
    os.environ["PATH"] = "/usr/local/cuda-13.1/bin:" + os.environ.get("PATH", "")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(HERE / ".torch_ext" / f"gpu{args.gpu}")
    os.environ.setdefault("MAX_JOBS", "4")

    if __package__:
        from .core import PROBE_LOCK_PATH, file_sha256, load_cells, read_json, result_root
        from .validate import validate_launch_ready
    else:
        from core import PROBE_LOCK_PATH, file_sha256, load_cells, read_json, result_root
        from validate import validate_launch_ready

    ready = validate_launch_ready("probes", args.gpu, allow_busy=args.allow_busy)
    probe_lock = read_json(PROBE_LOCK_PATH)
    binding = {
        "frozen_gate": probe_lock["frozen_gate"],
        "git_commit": ready["git_commit"],
        "gpu": ready["gpu"],
        "physical_gpu": args.gpu,
        "probe_lock_sha256": file_sha256(PROBE_LOCK_PATH),
        "source_bundle_sha256": probe_lock["source_bundle_sha256"],
    }
    root = result_root(args.tag) / "support_probes"
    index = run_probe_grid(
        args.probe_key,
        load_cells(),
        root,
        probe_binding=binding,
    )
    print(json.dumps({"index": _relative(_paths(root, args.probe_key, "g00", ".txt")["index"]), **index["resolution"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
