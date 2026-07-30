"""Distribution-robust correctness-gate campaign tooling.

This package is deliberately isolated from the completed Phase 1 and Phase 2
campaigns.  It imports no candidate kernels at module import time, so its
oracles, metrics, calibration, and validation logic remain CPU-testable.
"""

SCHEMA_VERSION = "1.0"
SEED_ALGORITHM = "sha256-low63-le-v1"

