# Checked legacy CUDA harness contract

The original Phase-2 fused wrappers are receipt-bound historical sources and
are intentionally not rewritten.  Corrected descendants must include
`checked_cuda_launch.h` and call:

1. `checked_dynamic_smem(...)` for every `cudaFuncSetAttribute` call; and
2. `checked_kernel_launch(...)` immediately after every kernel launch.

This makes allocation and launch failures fail closed instead of allowing a
later epilogue to rank an unwritten output buffer.  The v2 reachability
candidate already follows this contract; this overlay provides the reusable
fix for future descendants.

The historical wrappers remain unchanged. Verify the overlay without a GPU by
compiling it against injected CUDA/Torch shims:

```bash
python -m unittest ako_runs.controlled_followup.legacy_cuda_harness_fix.test_launch_checks -v
```

The test exercises successful calls and injected failures from both
`cudaFuncSetAttribute` and `cudaGetLastError`.
