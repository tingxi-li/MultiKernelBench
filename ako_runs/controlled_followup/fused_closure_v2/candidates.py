#!/usr/bin/env python3
"""Versioned candidate builders for fused closure v2.

GPU libraries are imported only inside :func:`build_candidate`, allowing the
campaign schema, plan, provenance, and analysis tests to run on CPU-only hosts.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import core


PHASE1 = core.REPO_ROOT / "ako_runs/phase1_matmul"
PHASE2 = core.REPO_ROOT / "ako_runs/phase2_fused_sdpa"


@dataclass
class BuiltCandidate:
    candidate_id: str
    definition: dict[str, Any]
    run: Callable[[Any, Any, Any, Any], Any]
    build_metadata: dict[str, Any]


def _imports():
    if str(PHASE2) not in sys.path:
        sys.path.insert(0, str(PHASE2))
    if str(PHASE1) not in sys.path:
        sys.path.insert(0, str(PHASE1))
    import torch
    import torch.nn.functional as functional
    import common2
    import variants2

    common2.setup_cuda_env()
    return torch, functional, common2, variants2


def _cached_half_weight(torch):
    box: dict[str, Any] = {}

    def prepare(weight):
        pointer = weight.data_ptr()
        if box.get("source_pointer") != pointer:
            box["source_pointer"] = pointer
            box["weight_kn_fp16"] = weight.half().t().contiguous()
        return box["weight_kn_fp16"]

    return prepare


def build_candidate(
    definition: dict[str, Any],
    *,
    expected_candidate_id: str | None = None,
) -> BuiltCandidate:
    """Build one frozen implementation and return a uniform four-input call."""
    candidate_id = definition["candidate_id"]
    if expected_candidate_id is not None and candidate_id != expected_candidate_id:
        raise core.ClosureError("candidate ID differs from requested binding")
    torch, functional, common2, variants2 = _imports()
    implementation = definition["implementation"]
    started = time.perf_counter()

    if implementation in {
        "historical_torch",
        "historical_torch_precast",
        "torch_contract_fp32",
    }:
        prepare_weight = _cached_half_weight(torch)
        contract_fp32 = implementation == "torch_contract_fp32"
        cast_in_region = implementation == "historical_torch"

        def run(x_fp32, x_fp16, weight, bias):
            weight_kn = prepare_weight(weight)
            if contract_fp32:
                output = torch.mm(x_fp16, weight_kn, out_dtype=torch.float32)
                output = output + bias
            else:
                activation = x_fp32.half() if cast_in_region else x_fp16
                output = activation @ weight_kn
                output = output + bias.half()
            output = functional.gelu(output, approximate="none")
            output = torch.softmax(output, dim=1)
            return output.float()

        metadata = {
            "builder": f"{Path(__file__).relative_to(core.REPO_ROOT)}:{implementation}",
            "build_wall_s": time.perf_counter() - started,
            "historical_reference_builder": (
                "ako_runs/phase2_fused_sdpa/variants2/fused_torch.py"
                if implementation != "torch_contract_fp32"
                else None
            ),
            "reported_compile_s": 0.0,
            "matmul_output_dtype": "torch.float32" if contract_fp32 else "torch.float16",
            "bias_dtype": "torch.float32" if contract_fp32 else "torch.float16",
            "probability_dtype": "torch.float32" if contract_fp32 else "torch.float16",
            "timed_input_dtype": "torch.float32" if cast_in_region else "torch.float16",
            "activation_cast_in_timed_region": cast_in_region,
        }

    elif implementation == "phase2_custom":
        overrides = core.parse_set(definition["set"])
        cfg = common2.make_fused_config(definition["dsl"], "GBGS", **overrides)
        expected = {
            "arith": "fp16",
            "cast": "precast",
            "wcache": "cached",
            "epilogue": "smem",
        }
        observed = {
            "arith": cfg.arith,
            "cast": cfg.cast,
            "wcache": cfg.extra.get("wcache"),
            "epilogue": cfg.extra.get("epilogue"),
        }
        if observed != expected:
            raise core.ClosureError(
                f"custom candidate {candidate_id} left fixed contract: {observed}"
            )
        built = variants2.build("fused", cfg)
        if built.x_dtype != torch.float16:
            raise core.ClosureError(
                f"custom candidate {candidate_id} does not accept precast fp16 x"
            )

        def run(_x_fp32, x_fp16, weight, bias):
            return built.run(x_fp16, weight, bias)

        metadata = {
            "builder": f"ako_runs/phase2_fused_sdpa/variants2/fused_{definition['dsl']}.py",
            "phase2_config": cfg.to_dict(),
            "reported_compile_s": built.compile_s,
            "build_wall_s": time.perf_counter() - started,
            "notes": built.notes,
            "n_kernels": built.n_kernels,
            "artifacts": {
                key: value
                for key, value in built.artifacts.items()
                if key
                in {
                    "backend_detail",
                    "block",
                    "grid",
                    "n_kernels",
                    "n_regs",
                    "n_spills",
                    "shared_bytes",
                    "wcache",
                }
            },
            "timed_input_dtype": "torch.float16",
        }
    else:  # validated by core, retained as a local fail-closed guard
        raise core.ClosureError(f"unsupported candidate implementation {implementation!r}")

    return BuiltCandidate(
        candidate_id=candidate_id,
        definition=definition,
        run=run,
        build_metadata=metadata,
    )
