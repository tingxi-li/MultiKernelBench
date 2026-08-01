#!/usr/bin/env python3
"""Refuse execution of the retired convergence-v2 campaign."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    policy = json.loads(Path(__file__).with_name("retirement.json").read_text())
    if policy.get("state") != "retired_never_launched" or policy.get("launch_policy", {}).get("launch_allowed") is not False:
        raise RuntimeError("invalid convergence-v2 retirement policy")
    print("REFUSED: convergence_v2 is retired; no provider or GPU work started")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
