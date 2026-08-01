#!/usr/bin/env python3
"""Freeze the 24 translator-specific, 19-attempt retune plans."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import protocol


SOURCE_GRID = protocol.REPO_ROOT / "ako_runs/phase1_matmul/jobs/native_tuned.json"
ROOT = protocol.HERE / "retune_plans"


def _parse_set(text: str) -> dict[str, int]:
    result = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        result[key] = int(value)
    return result


def native_grid(destination: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in protocol.load_json(SOURCE_GRID)
        if row.get("dsl") == destination and row.get("variant") == "D"
    ]
    if len(rows) != protocol.RETUNE_ATTEMPTS:
        raise ValueError(f"{destination}: expected 19 source grid rows, found {len(rows)}")
    configs = []
    for attempt, row in enumerate(rows):
        config = _parse_set(row["set"])
        if config.pop("kc") != 2048:
            raise ValueError(f"{destination}: historical grid KC changed")
        config["threads"] = 256
        if destination == "triton":
            config["warps"] = 8
        configs.append(
            {
                "attempt": attempt,
                "config": config,
                "kc": "from_recipe_resolution_lock",
                "failure_consumes_attempt": True,
            }
        )
    if len({protocol.canonical_sha256(row["config"]) for row in configs}) != 19:
        raise ValueError(f"{destination}: source retune grid contains duplicates")
    return configs


def plan(origin: str, destination: str, translator: str) -> dict[str, Any]:
    mapping = {
        "tilelang_phase1_confirmed": "plain_2d_n_then_m",
        "triton_grouped_autotuned": "one_dimensional_grouped_m_group8",
        "cuda_unlimited_native_d": "plain_2d_n_then_m",
    }[origin]
    return {
        "schema_version": 2,
        "campaign_id": protocol.CAMPAIGN_ID,
        "plan_kind": "recipient_native_retune",
        "recipe_origin": origin,
        "destination_dsl": destination,
        "translator": translator,
        "source_grid": {
            "path": protocol.repo_path(SOURCE_GRID),
            "sha256": protocol.file_sha256(SOURCE_GRID),
            "projection": f"the 19 Phase-1 {destination} D configurations",
        },
        "frozen_before_screening": True,
        "performance_split_visible": False,
        "preserve": [
            mapping,
            "fp16_tensor_core_operands",
            "fp32_chunk_accumulator",
            "fp32_outer_accumulator",
            "in_block_k_chunk_flush",
            "no_cross_block_split_k",
            "no_vendor_gemm",
        ],
        "selection": {
            "attempted_candidates": 19,
            "build_failures_consume_budget": True,
            "screen_repetitions": 2,
            "confirm_top_k": 2,
            "confirmation_blocks": 15,
            "gate_in_loop": "matmul_v4_all_required_cases",
            "winner_rule": "minimum_median_ms_among_all_v4_legal_candidates",
        },
        "candidates": native_grid(destination),
    }


def documents() -> dict[Path, bytes]:
    return {
        ROOT / translator / f"{origin}__to__{destination}.json": protocol.stable_json_bytes(
            plan(origin, destination, translator)
        )
        for translator in protocol.TRANSLATORS
        for origin in protocol.ORIGINS
        for destination in protocol.DESTINATIONS
    }


def write() -> None:
    for path, data in documents().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def check() -> None:
    expected = documents()
    stale = [protocol.repo_path(path) for path, data in expected.items()
             if not path.is_file() or path.read_bytes() != data]
    observed = set(ROOT.glob("*/*.json")) if ROOT.exists() else set()
    extras = [protocol.repo_path(path) for path in sorted(observed - set(expected))]
    if stale or extras:
        raise SystemExit(f"retune plans stale={stale}, extras={extras}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
