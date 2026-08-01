"""Analyze paired same-seed stress outcomes and effective sample size."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "measurements.jsonl"
OUT = HERE / "results" / "summary.json"


def main() -> int:
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line]
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[(row["seed_index"], row["gate_id"])].append(row)
    candidates = sorted({row["candidate_id"] for row in rows})
    groups = []
    for candidate in candidates:
        for gate_id in sorted({row["gate_id"] for row in rows}):
            subset = [row for row in rows if row["candidate_id"] == candidate and row["gate_id"] == gate_id]
            groups.append({
                "candidate_id": candidate,
                "gate_id": gate_id,
                "records": len(subset),
                "pass_records": sum(row["gate_pass"] for row in subset),
                "failure_seed_indices": [row["seed_index"] for row in subset if not row["gate_pass"]],
                "max_threshold_ratio": max((max(row["threshold_ratios"].values()) for row in subset), default=None),
                "effective_seed_count": len({row["seed_index"] for row in subset}),
            })
    summary = {
        "schema_version": 1,
        "record_type": "fused_same_seed_stress_summary",
        "campaign_id": "controlled-followup-fused-same-seed-stress-v1",
        "records": len(rows),
        "candidates": candidates,
        "shared_seed_effective_n": len({row["seed_index"] for row in rows}),
        "groups": groups,
        "threshold_mutation_authorized": False,
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(rows), "groups": len(groups), "output": str(OUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
