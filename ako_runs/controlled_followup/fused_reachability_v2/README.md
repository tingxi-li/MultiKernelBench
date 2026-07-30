# Fused reachability v2

Prospective, append-only test of the review's reachability interpretation. The
receipt-bound v1 CUDA files are imported as source templates but never edited.
The Phase-1 GEMM writes fp32 accumulators to its global intermediate; the
second kernel attaches fp32 bias, exact GELU, and row softmax. Therefore dynamic
shared memory is determined by A/B staging rather than a full `BM x BN` fp32
epilogue tile.

The immutable launch lock and a truthful serialized adapter manifest are
produced by `freeze.py` after all source and tests are final. Launchers refuse
source-hash drift. All results live under the git-ignored `results/` tree. After
analysis, `capture_evidence.py` validates the receipt chains and produces a
deterministic tar/index pair containing the selected result roots and frozen
execution sources.
