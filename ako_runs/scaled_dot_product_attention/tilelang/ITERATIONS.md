# Iteration Log — convergence re-run (tilelang, GPU3, from identity)

Op: O = softmax(scale·Q@K^T)@V, Q,K,V(32,32,512,1024), head_dim D=1024. Reference torch
SDPA exceeds flash's head-dim cap -> slow fp32 math path ~59 ms. Full curve in
convergence.csv.

- iter1 identity (torch SDPA): 57.7 ms / 1.0x.
- 2 batched fp16 tensor-core kernels (batch = B*H = 1024 heads). K1 = QK^T (K-dim=D)
  with all S keys in one block so the softmax row lives in an fp32 fragment (never a
  fp16 score roundtrip, which exp would amplify) + scale + row-softmax -> P fp16.
  K2 = P@V -> O fp32. Scratch max-diff vs torch = 4.4e-5 (passes 1e-4 gate).
- Key levers (logged): fp16-in .half() -> 2.80x; fp32-in cast-on-load 3.04x (removes the
  Q/K/V .half() kernels = ~9 GB of extra HBM on 2 GB tensors); K2 no split-K flush 3.05x
  (softmax-normalized P sums to 1 => no fp16 swamping, verified); K2 BN256 3.21x (BEST).
- Non-winners: K1 BM32 2.54x / BM128 2.44x (tall-skinny or spilling GEMM); K1 BK16 deep
  pipe 2.92x; 3-kernel split with fp32 scores 2.78x (the extra 2 GB score roundtrip
  outweighs the occupancy gain).
- NCU: 2 launches, 4.48 passes, occ ~17% (K1's (BM,S) score fragment is the register
  bound). The residual gap to the compute-ideal ~9 ms is this occupancy + the P roundtrip.
- STOP: 3.2x past the external torch ceiling (stop-rule met many times over); last 3
  levers below the 3.21x best.
- BEST 3.21x (18.4 ms). Slightly under the prior unlogged 3-kernel run (3.44x) and the
  finding's 3.41x — my 3-kernel retry needed fp32 scores for correctness, which cost more
  HBM than it saved; a true fused-flash kernel is blocked by the D=1024 O-accumulator.
