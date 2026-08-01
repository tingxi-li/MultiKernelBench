#!/usr/bin/env python3
"""Verify the deterministic 304-cell requested manifest."""
from __future__ import annotations

from collections import Counter

try:
    from .core import GRID_IDS, LANES, STRATEGIES, load_cells
except ImportError:  # direct script execution
    from core import GRID_IDS, LANES, STRATEGIES, load_cells


def main() -> int:
    cells = load_cells()
    assert len(cells) == 304
    assert Counter(row["strategy"] for row in cells) == {strategy: 76 for strategy in STRATEGIES}
    assert Counter(row["lane"] for row in cells) == {lane: 76 for lane in LANES}
    assert Counter(row["grid_id"] for row in cells) == {grid: 16 for grid in GRID_IDS}
    pending = sum(row["support_declared"] is None for row in cells)
    print(f"cells=304 pending_support_cells={pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
