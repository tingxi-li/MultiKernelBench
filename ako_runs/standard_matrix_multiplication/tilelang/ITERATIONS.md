# Iteration Log — convergence re-run (tilelang, GPU3, from identity)

Op: C = A@B, M2048 K8192 N4096 fp32. Reference torch.matmul = cuBLAS on CUDA cores
(TF32 disabled) = 4.48 ms / ~30 TFLOP/s. Lever = fp16 tensor cores. Full curve in
convergence.csv.

- iter1 identity (cuBLAS fp32): 4.46 ms / 1.0x.
- Naive fp16 T.gemm (fp32 C fragment) FAILS the 1e-4 gate: max 0.237 > ~0.205 tol,
  with a *systematic* -0.186 bias. Diagnosed by measuring bias vs K -> scales as K^2
  => T.gemm's MMA accumulator swamps in fp16 despite the fp32 fragment (order-invariant
  across all tilings). tf32 tiles were worse (-1.54).
- FIX (independently found) = in-block split-K flush: T.gemm accumulates a short KC-length
  chunk into a fragment, then that partial is added into a TRUE fp32 accumulator fragment;
  bias drops to ~ K*KC (KC=2048 -> -0.046, max 0.092, passes 2x). No atomics, no grid-z.
- Logged sweep (all correct): KC1024/BN128 3.94x; KC2048/BN128 3.97x; KC2048/BN256 4.19x
  (BEST); t512 3.82x; fp32-in cast-on-load 3.83x (slower: GEMM re-reads tiles ~16x so
  fp16 tiles halve 3GB->1.5GB, beating the 2 cast launches); BK64/st2 4.06x; BM64 3.72x.
- NCU: GEMM ~128 TFLOP/s, L2 91% hit; the 2 .half() casts are separate small kernels but
  net-win because fp16 tiles halve the GEMM's re-read traffic. Dual-accumulator register
  pressure caps occupancy (~17%), so bigger M-reuse (BM128,BN256) beats more blocks.
- STOP: 4 consecutive levers below the 4.19x best (plateau); far past cuBLAS external ref.
- BEST 4.19x (1.07 ms) — EXCEEDS the prior unlogged run (3.40x split-K-atomic) and the
  finding's 3.86x, because the in-block fp32 flush avoids grid-z atomic contention.
