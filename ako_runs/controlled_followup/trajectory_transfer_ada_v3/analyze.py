#!/usr/bin/env python3
"""Independently validate and analyze trajectory-transfer evidence."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from . import protocol
except ImportError:  # direct script execution
    import protocol  # type: ignore


def _error(message: str) -> Exception:
    return protocol.ProtocolError(message)


def _timing(campaign: dict[str, Any]) -> tuple[int, int, int]:
    timing = campaign.get("timing", {})
    trials = int(timing.get("trials", timing.get("trials_per_record", 100)))
    tail = timing.get("controlling_trials", timing.get("primary_trials", [60, 100]))
    if isinstance(tail, dict):
        tail = [tail.get("start_inclusive"), tail.get("stop_exclusive")]
    if (
        not isinstance(tail, list)
        or len(tail) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in tail)
        or not 0 <= tail[0] < tail[1] <= trials
    ):
        raise _error("campaign has an invalid controlling-trial window")
    return trials, tail[0], tail[1]


def _times(value: Any, trials: int) -> list[float]:
    if not isinstance(value, list) or len(value) != trials:
        raise _error(f"timing record must contain exactly {trials} trials")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item <= 0 for item in result):
        raise _error("timing record contains a nonpositive or nonfinite trial")
    return result


def validate_timing_record(
    campaign: dict[str, Any], row: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Validate the immutable row binding and rederive both medians."""
    trials, tail_start, tail_stop = _timing(campaign)
    expected = {
        "campaign_id": campaign["campaign_id"],
        "primitive_map_sha256": protocol.file_sha256(protocol.PRIMITIVE_MAP_PATH),
        "record_type": "trajectory_transfer_ada_v3_timing_record",
        "row": row,
        "row_sha256": protocol.canonical_sha256(row),
        "schema_version": 1,
    }
    changed = [key for key, value in expected.items() if record.get(key) != value]
    values = _times(record.get("times_ms"), trials)
    primary = statistics.median(values[tail_start:tail_stop])
    full = statistics.median(values)
    identity = record.get("implementation_sha256")
    graph = record.get("primitive_graph_sha256")
    route = record.get("structural_route")
    expected_route = protocol.ROUTE_BY_DESTINATION.get(row.get("destination"))
    start, end = record.get("t_start_unix_ns"), record.get("t_end_unix_ns")
    if (
        changed
        or record.get("ok") is not True
        or not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or not isinstance(graph, str)
        or len(graph) != 64
        or any(character not in "0123456789abcdef" for character in graph)
        or not isinstance(record.get("coordinate_cell_id"), str)
        or not record["coordinate_cell_id"]
        or not isinstance(record.get("implementation_id"), str)
        or not record["implementation_id"]
        or route != expected_route
        or not isinstance(record.get("runtime_modules"), dict)
        or isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start <= 0
        or end < start
        or not math.isclose(float(record.get("primary_tail_median_ms", -1)), primary)
        or not math.isclose(float(record.get("full_median_ms", -1)), full)
    ):
        raise _error(f"timing record is malformed or foreign: {changed}")
    return record


def _exact_interval(values: Iterable[float]) -> dict[str, float]:
    from ako_runs.controlled_followup.fused_epilogue_crossed_v1.core import (
        exact_median_interval,
    )

    return exact_median_interval(list(values))


def _interval(log_values: list[float]) -> dict[str, Any]:
    ratio = _exact_interval(math.exp(value) for value in log_values)
    return {
        "block_log_ratios": log_values,
        "log_ratio_interval": {
            "ci_lo": math.log(ratio["ci_lo"]),
            "median": math.log(ratio["median"]),
            "ci_hi": math.log(ratio["ci_hi"]),
        },
        "ratio_interval": ratio,
    }


def _direction(effect: dict[str, Any], floor: float) -> str:
    interval = effect["log_ratio_interval"]
    if interval["ci_lo"] > floor:
        return "speedup"
    if interval["ci_hi"] < -floor:
        return "slowdown"
    return "unresolved_at_sham_floor"


def _upper_sign_p(values: list[float], threshold: float) -> float:
    """Exact one-sided sign p-value; equality is conservatively nonpositive."""
    positives = sum(value > threshold for value in values)
    n = len(values)
    return sum(math.comb(n, k) for k in range(positives, n + 1)) / (2**n)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def _factor_values(campaign: dict[str, Any], key: str, fallback: Iterable[str]) -> tuple[str, ...]:
    raw = campaign.get("factors", {}).get(key, campaign.get(key, fallback))
    if isinstance(raw, dict):
        raw = raw.get("levels", raw.get("values", ()))
    if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(x, str) for x in raw):
        raise _error(f"campaign factor is malformed: {key}")
    return tuple(raw)


def _rows(manifest: Any) -> list[dict[str, Any]]:
    rows = manifest.get("rows") if isinstance(manifest, dict) else manifest
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise _error("confirmation manifest has no row list")
    return rows


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("record_id", row.get("row_id"))
    if not isinstance(value, str) or not value:
        raise _error("confirmation row has no stable row_id")
    return value


def estimate(
    campaign: dict[str, Any], manifest: Any, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Estimate destination gains, donor gain, and gain retention."""
    inference = campaign.get("inference", {})
    if inference != protocol.INFERENCE_CONTRACT:
        raise _error("frozen inference contract is missing or malformed")
    if inference["controlling_trials"] != list(_timing(campaign)[1:]):
        raise _error("frozen inference and timing windows disagree")
    primary_estimands = inference.get("primary_estimands")
    secondary_names = inference.get("secondary_estimands")
    rows = _rows(manifest)
    expected = {_row_id(row): row for row in rows}
    observed = {_row_id(record.get("row", {})): record for record in records}
    if len(expected) != len(rows) or len(observed) != len(records) or set(observed) != set(expected):
        raise _error("timing evidence differs from the exact confirmation manifest")
    for row_id, row in expected.items():
        validate_timing_record(campaign, row, observed[row_id])

    destinations = tuple(getattr(protocol, "DESTINATIONS", ())) or _factor_values(
        campaign, "destinations", ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")
    )
    declared_adaptations = tuple(getattr(protocol, "ADAPTATIONS", ())) or _factor_values(
        campaign, "adaptations", ("donor_fixed", "bounded_retune")
    )
    distributions = _factor_values(
        campaign, "distributions", campaign.get("workload", {}).get("distributions", ("positive", "withheld_signed"))
    )
    blocks = int(campaign.get("timing", {}).get("blocks", 15))

    grouped: dict[tuple[str, str, str, str], dict[int, tuple[float, str]]] = defaultdict(dict)
    sham: dict[tuple[str, str, str], dict[int, tuple[float, str]]] = defaultdict(dict)
    for record in records:
        row = record["row"]
        block = row.get("block")
        if isinstance(block, bool) or not isinstance(block, int) or not 0 <= block < blocks:
            raise _error("timing row has an invalid block")
        destination, distribution = row.get("destination"), row.get("distribution")
        if destination not in destinations or distribution not in distributions:
            raise _error("timing row has a foreign destination/distribution")
        value = (float(record["primary_tail_median_ms"]), record["implementation_sha256"])
        kind = row.get("record_kind")
        if kind == "same_artifact_label_sham":
            label = row.get("label")
            normalized_label = {
                "same_artifact_sham_a": "sham_a",
                "same_artifact_sham_b": "sham_b",
            }.get(label)
            if normalized_label is None:
                raise _error("sham row has a foreign label")
            key = (destination, distribution, normalized_label)
            target = sham[key]
        elif kind == "candidate":
            adaptation, state = row.get("adaptation"), row.get("mechanism_state")
            if adaptation not in declared_adaptations or state not in {"off", "on"}:
                raise _error("treatment row has a foreign adaptation/state")
            key = (destination, adaptation, distribution, state)
            target = grouped[key]
        else:
            raise _error("timing row has an unknown record_kind")
        if block in target:
            raise _error(f"duplicate timing block: {key}/{block}")
        target[block] = value

    adaptations = tuple(
        adaptation
        for adaptation in declared_adaptations
        if any(
            row.get("record_kind") == "candidate" and row.get("adaptation") == adaptation
            for row in rows
        )
    )
    if not adaptations:
        raise _error("confirmation manifest contains no treatment adaptation")

    expected_blocks = set(range(blocks))

    def series(mapping: dict[int, tuple[float, str]], label: str) -> tuple[list[float], set[str]]:
        if set(mapping) != expected_blocks:
            raise _error(f"incomplete randomized blocks: {label}")
        return [mapping[index][0] for index in range(blocks)], {mapping[index][1] for index in range(blocks)}

    sham_intervals: dict[str, dict[str, dict[str, float]]] = {}
    sham_identities: dict[str, str] = {}
    floor = 0.0
    for destination in destinations:
        sham_intervals[destination] = {}
        destination_identities: set[str] = set()
        for distribution in distributions:
            left, left_ids = series(sham[(destination, distribution, "sham_a")], f"{destination}/{distribution}/sham_a")
            right, right_ids = series(sham[(destination, distribution, "sham_b")], f"{destination}/{distribution}/sham_b")
            if len(left_ids) != 1 or left_ids != right_ids:
                raise _error(f"{destination} sham labels are not byte-identical")
            destination_identities.update(left_ids)
            interval = _exact_interval(a / b for a, b in zip(left, right))
            sham_intervals[destination][distribution] = interval
            floor = max(floor, abs(math.log(interval["ci_lo"])), abs(math.log(interval["ci_hi"])))
        if len(destination_identities) != 1:
            raise _error(f"{destination} sham identity changed across distributions")
        sham_identities[destination] = next(iter(destination_identities))

    effects: dict[tuple[str, str, str], dict[str, Any]] = {}
    p_values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for destination in destinations:
        for adaptation in adaptations:
            cross_distribution_ids = {"off": set(), "on": set()}
            for distribution in distributions:
                off, off_ids = series(grouped[(destination, adaptation, distribution, "off")], f"{destination}/{adaptation}/{distribution}/off")
                on, on_ids = series(grouped[(destination, adaptation, distribution, "on")], f"{destination}/{adaptation}/{distribution}/on")
                if len(off_ids) != 1 or len(on_ids) != 1:
                    raise _error("one treatment label resolved to multiple implementations")
                cross_distribution_ids["off"].update(off_ids)
                cross_distribution_ids["on"].update(on_ids)
                logs = [math.log(before / after) for before, after in zip(off, on)]
                effect = _interval(logs)
                effect.update({
                    "direction_above_global_sham_floor": _direction(effect, floor),
                    "off_implementation_sha256": next(iter(off_ids)),
                    "on_implementation_sha256": next(iter(on_ids)),
                    "speedup_sign_p_one_sided_at_floor": _upper_sign_p(logs, floor),
                })
                effects[(destination, adaptation, distribution)] = effect
                p_values[(adaptation, distribution)][destination] = effect["speedup_sign_p_one_sided_at_floor"]
            if any(len(values) != 1 for values in cross_distribution_ids.values()):
                raise _error(
                    "selection changed across confirmation distributions; withheld_signed may not select"
                )

    holm = {key: _holm(values) for key, values in p_values.items()}
    secondary_contrasts = []
    if set(adaptations) == {"donor_fixed", "bounded_retune"}:
        definitions = (
            (
                secondary_names[0],
                ("donor_fixed", "off"),
                ("bounded_retune", "on"),
                "fixed_off_time_over_retuned_on_time",
            ),
            (
                secondary_names[1],
                ("donor_fixed", "on"),
                ("bounded_retune", "on"),
                "fixed_on_time_over_retuned_on_time",
            ),
            (
                secondary_names[2],
                ("donor_fixed", "off"),
                ("bounded_retune", "off"),
                "fixed_off_time_over_retuned_off_time",
            ),
        )
        for destination in destinations:
            for distribution in distributions:
                for name, before_key, after_key, orientation in definitions:
                    before, _before_ids = series(
                        grouped[(destination, before_key[0], distribution, before_key[1])],
                        f"{destination}/{before_key}/{distribution}",
                    )
                    after, _after_ids = series(
                        grouped[(destination, after_key[0], distribution, after_key[1])],
                        f"{destination}/{after_key}/{distribution}",
                    )
                    effect = _interval(
                        [math.log(left / right) for left, right in zip(before, after)]
                    )
                    effect["direction_above_global_sham_floor"] = _direction(effect, floor)
                    secondary_contrasts.append({
                        "contrast": name,
                        "destination": destination,
                        "distribution": distribution,
                        "effect": effect,
                        "speedup_orientation": orientation,
                    })
    donor = campaign.get("donor", {}).get(
        "destination",
        campaign.get("donor", {}).get(
            "origin_dsl", campaign.get("donor", {}).get("lane", "triton")
        ),
    )
    if donor not in destinations:
        raise _error("donor destination is outside the destination factor")
    contrasts, classifications = [], []
    for adaptation in adaptations:
        for destination in destinations:
            destination_directions: dict[str, str] = {}
            donor_directions: dict[str, str] = {}
            adjusted: dict[str, float] = {}
            for distribution in distributions:
                destination_effect = effects[(destination, adaptation, distribution)]
                donor_effect = effects[(donor, adaptation, distribution)]
                destination_directions[distribution] = destination_effect["direction_above_global_sham_floor"]
                donor_directions[distribution] = donor_effect["direction_above_global_sham_floor"]
                adjusted[distribution] = holm[(adaptation, distribution)][destination]
                contrast_logs = [
                    destination_log - donor_log
                    for destination_log, donor_log in zip(
                        destination_effect["block_log_ratios"], donor_effect["block_log_ratios"]
                    )
                ]
                contrasts.append({
                    "adaptation": adaptation,
                    "destination": destination,
                    "distribution": distribution,
                    "effect": _interval(contrast_logs),
                    "estimand": inference["destination_minus_donor_estimand"],
                })
            if destination == donor:
                classification = "fresh_donor_reference"
            elif all(value == "speedup" for value in (*destination_directions.values(), *donor_directions.values())):
                classification = "destination_gain_and_donor_gain_both_clear_sham_floor"
            elif "slowdown" in destination_directions.values():
                classification = "destination_slowdown"
            else:
                classification = "unresolved_transfer_benefit"
            classifications.append({
                "adaptation": adaptation,
                "destination": destination,
                "destination_directions": destination_directions,
                "donor_directions": donor_directions,
                "holm_adjusted_speedup_p_at_floor": adjusted,
                "classification": classification,
            })

    effect_rows = [
        {
            "destination": destination,
            "adaptation": adaptation,
            "distribution": distribution,
            "effect": effect,
            "estimand": primary_estimands[adaptation],
        }
        for (destination, adaptation, distribution), effect in effects.items()
    ]
    aggregate = all(
        row["classification"] in {
            "fresh_donor_reference", "destination_gain_and_donor_gain_both_clear_sham_floor"
        }
        and all(value <= inference["alpha"] for value in row["holm_adjusted_speedup_p_at_floor"].values())
        for row in classifications
    )
    return {
        "schema_version": 1,
        "record_type": "trajectory_transfer_ada_v3_final_analysis",
        "campaign_id": campaign["campaign_id"],
        "complete": True,
        "controlling": campaign["controlling"],
        "claim_scope": "one frozen donor mechanism implemented by each destination's preregistered direct-primitive or manual-reconstruction route on the bound Ada GPU",
        "route_by_destination": dict(protocol.ROUTE_BY_DESTINATION),
        "route_comparison_claim_authorized": not inference["route_comparison_forbidden"],
        "route_comparison_prohibition": "route is fixed by and confounded with destination programming model; no direct-versus-manual aggregate is identified",
        "inference_contract": inference,
        "primary_trials": list(_timing(campaign)[1:]),
        "timing_records": len(records),
        "global_sham_resolution_floor_log_ratio": floor,
        "sham_intervals": sham_intervals,
        "sham_implementation_sha256": sham_identities,
        "effects": effect_rows,
        "secondary_paired_contrasts": secondary_contrasts,
        "destination_minus_donor_contrasts": contrasts,
        "classifications": classifications,
        "all_destination_aggregate_claim_authorized": aggregate,
        "aggregate_rule": inference["aggregate_rule"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("final", "verify"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        from . import runner
    except ImportError:
        import runner  # type: ignore
    try:
        campaign, manifest, records, bindings = runner.load_completed(args.tag)
        value = {**estimate(campaign, manifest, records), **bindings}
        target = args.out or (runner.result_root(args.tag) / "analysis.json")
        if args.command == "verify":
            if protocol.read_json(target) != value:
                raise _error("analysis failed byte-rederivation")
            print("verified=PASS")
        else:
            runner.write_once(target, value)
            print(f"analysis={target}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError, protocol.ProtocolError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
