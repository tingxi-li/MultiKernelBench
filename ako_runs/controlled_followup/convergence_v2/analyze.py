#!/usr/bin/env python3
"""Dependency-free survival analysis primitives for convergence outcomes."""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class SurvivalObservation:
    trajectory_id: str
    group: str
    duration_s: float
    event_observed: bool

    def validate(self) -> None:
        if not self.trajectory_id or not self.group:
            raise ValueError("trajectory_id and group are required")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")


def kaplan_meier(observations: Iterable[SurvivalObservation]) -> list[dict[str, float | int]]:
    rows = list(observations)
    if not rows:
        raise ValueError("Kaplan-Meier requires observations")
    for row in rows:
        row.validate()
    survival = 1.0
    curve: list[dict[str, float | int]] = [{"time_s": 0.0, "survival": 1.0, "at_risk": len(rows), "events": 0, "censored": 0}]
    for current in sorted({row.duration_s for row in rows}):
        at_risk = sum(row.duration_s >= current for row in rows)
        events = sum(row.duration_s == current and row.event_observed for row in rows)
        censored = sum(row.duration_s == current and not row.event_observed for row in rows)
        if events:
            survival *= 1.0 - events / at_risk
        curve.append(
            {
                "time_s": current,
                "survival": survival,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
            }
        )
    return curve


def restricted_mean_survival_time(observations: Iterable[SurvivalObservation], tau_s: float) -> float:
    if not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("tau_s must be finite and positive")
    curve = kaplan_meier(observations)
    area = 0.0
    previous_time = 0.0
    previous_survival = 1.0
    for point in curve[1:]:
        current = min(float(point["time_s"]), tau_s)
        if current > previous_time:
            area += (current - previous_time) * previous_survival
        if float(point["time_s"]) >= tau_s:
            return area
        previous_time = float(point["time_s"])
        previous_survival = float(point["survival"])
    if previous_time < tau_s:
        area += (tau_s - previous_time) * previous_survival
    return area


def logrank_test(
    left: Iterable[SurvivalObservation], right: Iterable[SurvivalObservation]
) -> dict[str, float]:
    group_a = list(left)
    group_b = list(right)
    if not group_a or not group_b:
        raise ValueError("log-rank requires two nonempty groups")
    for row in group_a + group_b:
        row.validate()
    event_times = sorted({row.duration_s for row in group_a + group_b if row.event_observed})
    observed_a = expected_a = variance = 0.0
    for current in event_times:
        n_a = sum(row.duration_s >= current for row in group_a)
        n_b = sum(row.duration_s >= current for row in group_b)
        d_a = sum(row.duration_s == current and row.event_observed for row in group_a)
        d_b = sum(row.duration_s == current and row.event_observed for row in group_b)
        n = n_a + n_b
        d = d_a + d_b
        observed_a += d_a
        expected_a += d * n_a / n
        if n > 1:
            variance += n_a * n_b * d * (n - d) / (n * n * (n - 1))
    if variance == 0:
        chi_square = 0.0 if observed_a == expected_a else math.inf
    else:
        chi_square = (observed_a - expected_a) ** 2 / variance
    p_value = 0.0 if math.isinf(chi_square) else math.erfc(math.sqrt(chi_square / 2.0))
    return {
        "observed_left": observed_a,
        "expected_left": expected_a,
        "variance": variance,
        "chi_square": chi_square,
        "p_value": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    for name, value in p_values.items():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid p-value for {name}: {value}")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - index)))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def analyze(observations: list[SurvivalObservation], tau_s: float) -> dict:
    if len({row.trajectory_id for row in observations}) != len(observations):
        raise ValueError("duplicate trajectory_id in outcomes")
    groups = sorted({row.group for row in observations})
    by_group = {group: [row for row in observations if row.group == group] for group in groups}
    summaries = {
        group: {
            "n": len(rows),
            "events": sum(row.event_observed for row in rows),
            "right_censored": sum(not row.event_observed for row in rows),
            "rmst_s": restricted_mean_survival_time(rows, tau_s),
            "kaplan_meier": kaplan_meier(rows),
        }
        for group, rows in by_group.items()
    }
    tests: dict[str, dict[str, float]] = {}
    for left, right in itertools.combinations(groups, 2):
        key = f"{left}__vs__{right}"
        tests[key] = logrank_test(by_group[left], by_group[right])
    adjusted = holm_adjust({key: value["p_value"] for key, value in tests.items()})
    for key, value in adjusted.items():
        tests[key]["holm_p_value"] = value
    return {
        "schema_version": 1,
        "estimand": "time_to_first_terminal_legal_candidate_within_5pct_of_frozen_reference",
        "tau_s": tau_s,
        "groups": summaries,
        "pairwise_logrank": tests,
        "censoring_policy": "resource caps and incomplete attempts are retained as right-censored",
    }


def load_outcomes(path: Path) -> list[SurvivalObservation]:
    rows: list[SurvivalObservation] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank outcome line {line_number}")
        raw = json.loads(line)
        row = SurvivalObservation(
            trajectory_id=str(raw["trajectory_id"]),
            group=str(raw["group"]),
            duration_s=float(raw["duration_s"]),
            event_observed=bool(raw["event_observed"]),
        )
        row.validate()
        rows.append(row)
    if not rows:
        raise ValueError("outcome stream is empty; no analysis is fabricated")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--tau-s", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(load_outcomes(args.outcomes), args.tau_s)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

