#!/usr/bin/env python3
"""Regenerate or verify the deterministic 228-cell manifest."""
from __future__ import annotations

import argparse

from core import BASE_JOBS_PATH, CELLS_PATH, make_cells, read_json, stable_write, validate_cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = make_cells(read_json(BASE_JOBS_PATH))
    if args.write:
        stable_write(CELLS_PATH, expected)
    observed = read_json(CELLS_PATH)
    validate_cells(observed)
    print(f"cells={len(observed)} sha-ready=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
