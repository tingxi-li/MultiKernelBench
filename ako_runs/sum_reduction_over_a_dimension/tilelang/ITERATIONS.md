# Iteration Log — convergence re-run (tilelang, GPU3, from identity)

Op: sum_reduction over dim=1 of (128,4096,4096) fp32 = read 8.59 GB once, write 2 MB
=> pure HBM streaming (memory-bound). See convergence.csv for the full logged curve.

- iter1 identity (torch.sum): 9.79 ms / 1.00x. NCU baseline: 1.03 passes, 4 launches,
  0.246 GiB of partial-sum writes (torch does a multi-pass reduction).
- Kernel: single-pass column reduction. One block per (batch, column-tile); threads map
  to the innermost (contiguous) k-axis for coalesced 128B loads, each thread walks all
  4096 rows accumulating in a register fragment, writes its output once => 1.0 HBM pass.
- Logged scan: BK=256 (9.74), BK=512 (9.74), BK=128 (9.83), BK=256+float4/thread (9.72).
  float4 (more memory-level parallelism) won marginally -> best 9.72 ms / 1.0072x.
- NCU at best: 8.005 GiB = 1.00 passes, single launch (vs torch 1.03/4 launches) =>
  ~885 GB/s ~= 92% of ~960 GB/s peak = the achievable 1-read roofline.
- STOP: within 5% of ceiling (I am the ceiling; torch ref 9.79) AND ncu confirms the
  1.00-pass HBM roofline is hit; last 2 variants <0.3% apart. Detector-clean.
- Re-reached the prior unlogged run's ceiling (also 9.72 ms / 1.0072x), independently.
