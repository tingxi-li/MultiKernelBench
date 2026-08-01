"""Derive continuous threshold margins from frozen matmul-v4 records.

This is a post-campaign diagnostic. It does not change v4 acceptance, refit a
threshold, or rewrite ``margin_report_v1.json``. Version 2 validates every
required metric and gate decision fail-closed, computes one maximum threshold
utilization per real-candidate record, and reports both the nearest and the
worst failing records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
GATE = REPO / "ako_runs/controlled_followup/robust_gate/calibration/gate_spec_matmul_v4.json"
RAW = HERE / "results/raw"
OUT = HERE / "results/margin_report_v2.json"
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


class MarginReportError(RuntimeError):
    """The source evidence cannot support a valid margin report."""


@dataclass(frozen=True)
class Ratio:
    metric: str
    value: float
    threshold: float
    utilization: float


@dataclass(frozen=True)
class RecordMargin:
    source_file: str
    source_line: int
    candidate: str
    case_id: str
    gate_id: str
    seed_index: int
    gate_pass: bool
    maximum: Ratio
    failed_metrics: tuple[str, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MarginReportError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MarginReportError(f"{label} must be finite")
    if number < 0:
        raise MarginReportError(f"{label} must be nonnegative")
    return number


def gate_thresholds(spec: dict[str, Any]) -> dict[str, dict[str, float]]:
    gates: dict[str, dict[str, float]] = {}
    for spec_key, gate in spec.get("gates", {}).items():
        gate_id = gate.get("gate_id")
        if not isinstance(gate_id, str) or not gate_id:
            raise MarginReportError(f"gate {spec_key!r} has no gate_id")
        if gate_id in gates:
            raise MarginReportError(f"duplicate gate_id {gate_id!r}")
        limits: dict[str, float] = {}
        for metric, rule in gate.get("thresholds", {}).items():
            if rule.get("comparison") != "le":
                raise MarginReportError(
                    f"unsupported comparison for {gate_id}/{metric}: "
                    f"{rule.get('comparison')!r}"
                )
            limits[metric] = _finite_number(
                rule.get("value"), f"threshold {gate_id}/{metric}"
            )
        if not limits:
            raise MarginReportError(f"gate {gate_id!r} has no thresholds")
        gates[gate_id] = limits
    if not gates:
        raise MarginReportError("gate specification has no gates")
    return gates


def _utilization(value: float, threshold: float) -> float:
    if threshold == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return value / threshold


def _ratio_json(value: float) -> float | str:
    return value if math.isfinite(value) else "infinity"


def _record_json(record: RecordMargin) -> dict[str, Any]:
    return {
        "candidate": record.candidate,
        "case_id": record.case_id,
        "gate_id": record.gate_id,
        "seed_index": record.seed_index,
        "source_file": record.source_file,
        "source_line": record.source_line,
        "max_threshold_utilization": _ratio_json(
            record.maximum.utilization
        ),
        "limiting_metric": record.maximum.metric,
        "metric_value": record.maximum.value,
        "threshold": record.maximum.threshold,
        "failed_metrics": list(record.failed_metrics),
    }


def validate_record(
    row: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    *,
    source_file: str,
    source_line: int,
) -> RecordMargin:
    prefix = f"{source_file}:{source_line}"
    gate_id = row.get("gate_id")
    if gate_id not in thresholds:
        raise MarginReportError(f"{prefix}: unknown gate_id {gate_id!r}")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise MarginReportError(f"{prefix}: metrics must be an object")

    ratios: list[Ratio] = []
    for metric, threshold in thresholds[gate_id].items():
        if metric not in metrics:
            raise MarginReportError(f"{prefix}: missing required metric {metric!r}")
        value = _finite_number(metrics[metric], f"{prefix}: metric {metric}")
        ratios.append(
            Ratio(
                metric=metric,
                value=value,
                threshold=threshold,
                utilization=_utilization(value, threshold),
            )
        )

    failed_metrics = tuple(
        ratio.metric for ratio in ratios if ratio.utilization > 1.0
    )
    gate_pass = row.get("gate_pass")
    if not isinstance(gate_pass, bool):
        raise MarginReportError(f"{prefix}: gate_pass must be boolean")
    derived_pass = not failed_metrics
    if gate_pass is not derived_pass:
        raise MarginReportError(
            f"{prefix}: gate_pass={gate_pass} disagrees with metrics "
            f"(derived {derived_pass})"
        )

    recorded_failures = row.get("threshold_failures")
    if not isinstance(recorded_failures, list) or not all(
        isinstance(item, str) for item in recorded_failures
    ):
        raise MarginReportError(f"{prefix}: threshold_failures must be a string list")
    recorded_metrics = tuple(item.split("=", 1)[0] for item in recorded_failures)
    if recorded_metrics != failed_metrics:
        raise MarginReportError(
            f"{prefix}: threshold_failures disagree with metrics: "
            f"recorded={recorded_metrics!r}, derived={failed_metrics!r}"
        )

    identity: dict[str, Any] = {}
    for name, expected_type in (
        ("candidate", str),
        ("case_id", str),
        ("seed_index", int),
    ):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise MarginReportError(f"{prefix}: invalid {name}")
        identity[name] = value
    if identity["seed_index"] < 0:
        raise MarginReportError(f"{prefix}: seed_index must be nonnegative")

    maximum = max(ratios, key=lambda ratio: ratio.utilization)
    return RecordMargin(
        source_file=source_file,
        source_line=source_line,
        candidate=identity["candidate"],
        case_id=identity["case_id"],
        gate_id=gate_id,
        seed_index=identity["seed_index"],
        gate_pass=gate_pass,
        maximum=maximum,
        failed_metrics=failed_metrics,
    )


def load_real_candidate_records(
    raw_dir: Path, thresholds: dict[str, dict[str, float]]
) -> tuple[list[RecordMargin], list[dict[str, Any]]]:
    records: list[RecordMargin] = []
    sources: list[dict[str, Any]] = []
    paths = sorted(raw_dir.glob("*.jsonl"))
    if not paths:
        raise MarginReportError(f"no raw JSONL files found in {raw_dir}")
    for path in paths:
        total_rows = 0
        real_rows = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MarginReportError(
                        f"{path.name}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if row.get("role") != "real_candidate":
                    continue
                real_rows += 1
                records.append(
                    validate_record(
                        row,
                        thresholds,
                        source_file=path.name,
                        source_line=line_number,
                    )
                )
        sources.append(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "jsonl_records": total_rows,
                "real_candidate_records": real_rows,
            }
        )
    if not records:
        raise MarginReportError("raw evidence contains no real_candidate records")
    return records, sources


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise MarginReportError("cannot calculate a quantile of an empty group")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _quantile_key(probability: float) -> str:
    return f"p{round(probability * 100):02d}"


def _compact_axis(records: list[RecordMargin]) -> dict[str, Any]:
    seed_indices = [record.seed_index for record in records]
    axis: dict[str, Any]
    if seed_indices == list(range(seed_indices[0], seed_indices[0] + len(seed_indices))):
        axis = {
            "kind": "contiguous_seed_index",
            "start": seed_indices[0],
            "stop_exclusive": seed_indices[-1] + 1,
        }
    else:
        axis = {"kind": "explicit_seed_index", "values": seed_indices}
    return {
        "axis": axis,
        "values": [
            _ratio_json(record.maximum.utilization) for record in records
        ],
    }


def build_report(
    *, gate_path: Path = GATE, raw_dir: Path = RAW
) -> dict[str, Any]:
    try:
        spec = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarginReportError(f"cannot load gate specification {gate_path}: {exc}") from exc
    thresholds = gate_thresholds(spec)
    records, sources = load_real_candidate_records(raw_dir, thresholds)

    seen: set[tuple[str, str, str, int]] = set()
    for record in records:
        key = (record.candidate, record.case_id, record.gate_id, record.seed_index)
        if key in seen:
            raise MarginReportError(f"duplicate real-candidate record {key!r}")
        seen.add(key)

    grouped: dict[tuple[str, str, str], list[RecordMargin]] = defaultdict(list)
    for record in records:
        grouped[(record.candidate, record.case_id, record.gate_id)].append(record)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidate, case_id, gate_id = key
        group_records = sorted(grouped[key], key=lambda record: record.seed_index)
        utilizations = sorted(record.maximum.utilization for record in group_records)
        failures = [record for record in group_records if not record.gate_pass]
        closest = min(failures, key=lambda record: record.maximum.utilization) if failures else None
        worst_failure = max(failures, key=lambda record: record.maximum.utilization) if failures else None
        worst_record = max(group_records, key=lambda record: record.maximum.utilization)
        failure_metrics = Counter(
            metric for record in group_records for metric in record.failed_metrics
        )
        groups.append(
            {
                "candidate": candidate,
                "case_id": case_id,
                "gate_id": gate_id,
                "records": len(group_records),
                "passing_records": len(group_records) - len(failures),
                "failing_records": len(failures),
                "failure_fraction": len(failures) / len(group_records),
                "failure_metric_counts": dict(sorted(failure_metrics.items())),
                "max_threshold_utilization_quantiles": {
                    _quantile_key(probability): _ratio_json(
                        _quantile(utilizations, probability)
                    )
                    for probability in QUANTILES
                },
                "per_record_max_threshold_utilization": _compact_axis(group_records),
                "closest_failing_record": _record_json(closest) if closest else None,
                "worst_failing_record": _record_json(worst_failure) if worst_failure else None,
                "worst_record": _record_json(worst_record),
            }
        )

    failures = [record for record in records if not record.gate_pass]
    closest = min(failures, key=lambda record: record.maximum.utilization) if failures else None
    worst = max(failures, key=lambda record: record.maximum.utilization) if failures else None
    report = {
        "schema_version": "2.0",
        "record_type": "matmul_v4_margin_report",
        "source": "frozen matmul-v4 raw records; post-campaign diagnostic only",
        "threshold_mutation_authorized": False,
        "method": {
            "per_record_statistic": "max over required frozen metrics of metric_value / threshold",
            "zero_threshold_rule": "0/0 utilization is 0; a positive value over a zero threshold is encoded as 'infinity'",
            "failure_rule": "a record fails when any required metric utilization is greater than 1",
            "quantiles": "linear interpolation at rank (n - 1) * p",
            "validation": "missing, nonnumeric, negative, or nonfinite required metrics and gate-decision mismatches abort report generation",
        },
        "gate_spec": {
            "path": str(gate_path.resolve().relative_to(REPO)),
            "sha256": file_sha256(gate_path),
        },
        "raw_sources": sources,
        "real_candidate_records": len(records),
        "passing_records": len(records) - len(failures),
        "failing_records": len(failures),
        "groups": groups,
        "closest_failing_record": _record_json(closest) if closest else None,
        "worst_failing_record": _record_json(worst) if worst else None,
    }
    # Enforce strict JSON output: no NaN/Infinity may leak from diagnostics.
    json.dumps(report, allow_nan=False)
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=GATE)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(gate_path=args.gate.resolve(), raw_dir=args.raw_dir.resolve())
    write_report(report, args.output.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "groups": len(report["groups"]),
                "records": report["real_candidate_records"],
                "failing_records": report["failing_records"],
                "closest_failure": report["closest_failing_record"][
                    "max_threshold_utilization"
                ],
                "worst_failure": report["worst_failing_record"][
                    "max_threshold_utilization"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
