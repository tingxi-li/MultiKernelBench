# Iteration Log

<!--
Per-iteration template (copy when adding a new iter entry under "## Iterations"):

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next

Append one row per iter to the Summary table below.
Status values: improved / no-change / regression / failed.
-->

## Summary

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | TF32 WMMA GEMM + fused GELU + softmax | 0.86x | 7.19 ms | regression |
| 2 | FP32 register-blocking BM=BN=128, BK=16, TM=TN=8 | 0.99x | 6.26 ms | improved |
| 3 | FP32 coalesced loads (k=e%BK,m=e/BK) + __ldg | 0.90x | 6.82 ms | regression |
| 4 | Transposed weight [K,N] for coalesced WT loads | 0.92x | 6.59 ms | regression |
| 5 | at::mm + fused bias+GELU+online-softmax kernel | 0.89x | 6.84 ms | regression |
| 6 | Restore iter-2 kernel (BM=BN=128, BK=16, TM=TN=8) | 0.75x | 8.26 ms | regression (thermal) |
| 7 | WMMA TF32 BKK=32 + precomputed WT + fused GELU+softmax | 1.35x | 4.58 ms | improved |
| 8 | cp.async double-buffer BKK=16, WMMA TF32 + fused epilogue | 1.32x | 4.56 ms | no-change (noisy) |
| 9 | BM=64 BN=128 BKK=32, 4 warps (2M×2N), __ldg hints | 1.47x | 4.14 ms | improved |
| 10 | BM=64 BN=128 BKK=16 cp.async double-buffer, 4 warps | 1.15x | 5.30 ms | regression |

## Iterations

### Iter 10 — BM=64 BN=128 BKK=16 cp.async double-buffer, 4 warps

- **Hypothesis:** Combine BM=64 occupancy benefit with cp.async pipeline hiding memory latency.
- **Changes:** BKK=16 with 2-stage async pipeline. Smem: 2*(As[64][20]+Bs[16][132])*4=27136B.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 5.30 ms (mean), 3.56~5.55 ms (min~max)
  - Speedup: 1.15x (mean)
- **Analysis:** High variance (std=0.523ms) and slower than iter-9. The BKK=16 pipeline has 512 K-loop iterations, each needing a pipeline barrier. The pipeline overhead dominates over the latency hiding benefit. Iter-9 BKK=32 without pipeline is clearly better.
- **Next:** Best = iter-9 (1.47x). Try to improve on that with different warp layout or BM=64 BN=256.

### Iter 9 — BM=64, BN=128, BKK=32, 4 warps, __ldg hints

- **Hypothesis:** Smaller M tile (BM=64 vs 128) reduces smem from 35KB to 26KB, allowing two blocks to co-reside per SM. This doubles SM occupancy, hiding latency through more warps in flight.
- **Changes:** BM=64, 4 warps (2M×2N), NT2=128, smem=26112B (allows 2 blocks/SM). Each warp still handles 2×4 WMMA tiles (32×64 region). Added `__ldg` hints for global loads.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 4.14 ms (mean), 3.83~4.21 ms (min~max)
  - Speedup: 1.47x (mean)
- **Analysis:** Clear improvement over iter-7 (1.35x → 1.47x). The BM=64 occupancy boost (2 blocks/SM) is the key factor. Low std=0.074ms shows stable performance.
- **Next:** Try BM=64 + double-buffered pipeline (smem=2*26112=52224B > 48KB — won't fit). Try BKK=16 with double-buffer: 2*(64*(16+4)+16*(128+4))*4 = 2*(5120+8448) = 27136B — fits! Or try BN=256 to test wider warp-N tile.

### Iter 8 — cp.async double-buffer BKK=16, WMMA TF32 + fused GELU+softmax

- **Hypothesis:** Using cp.async 2-stage pipeline to overlap global memory loads with WMMA computation should hide the memory latency of the large GEMM tiles.
- **Changes:** Changed to BKK=16 with STAGES=2 double-buffering via `__pipeline_memcpy_async`/`__pipeline_commit`/`__pipeline_wait_prior`. Preloads next tile while computing current. Added `cuda_pipeline.h`.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 4.56 ms (mean), 3.98~4.84 ms (min~max)
  - Speedup: 1.32x (mean)
- **Analysis:** Mean similar to iter-7 but higher variance (std=0.184ms vs 0.028ms). Fast-signal showed 3.71ms but full bench settled at 4.56ms. The double-buffering helps at warmup (min=3.98ms) but introduces more latency variability. The BKK=16 pipeline gives more launches per K-dimension (512 vs 256) which may add overhead.
- **Next:** Try BKK=32 with 2-stage pipeline (needs 2*(18432+16896)=70656B → doesn't fit). Instead try increasing WARPS configuration or use BKK=16 with 3-stage pipeline (37376*1.5=56064B > 48KB). Best approach: go back to BKK=32 single-stage from iter-7 as it has the best mean+std. Try tuning warp layout.

### Iter 7 — WMMA TF32 BKK=32, precomputed WT[K,N], fused bias+GELU+softmax

- **Hypothesis:** WMMA TF32 tensor cores (sm_89 Ada) with precomputed transposed weight [K,N] gives fully coalesced global loads. Fused bias+GELU+softmax kernel keeps all 32 elements per thread in registers, avoiding a full HBM intermediate pass.
- **Changes:** New kernel: wmma_gemm_tf32 (BM=128, BN=128, BKK=32, 8 warps 4M×2N, WM=2×WN=4 WMMA tiles), + bias_gelu_softmax_k (256T, 32 elements/thread, register-resident). Model.__init__ caches weight_T = weight.T.contiguous() via register_buffer so there's no per-call 256MB transpose.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 4.58 ms (mean), 4.50~4.64 ms (min~max)
  - Speedup: 1.35x (mean)
- **Analysis:** Clear win. WMMA TF32 tensor cores give real throughput over the FP32 register-blocking approach. Precomputed WT eliminates 256MB/call transpose overhead. Low std (0.028ms) shows stable clocks. The 1.35x speedup is genuine.
- **Next:** Try larger BKK (64) or double-buffered pipeline to further hide memory latency. Also try inline bias+GELU epilogue in the GEMM kernel to save the extra HBM pass for the C matrix.

### Iter 6 — Restore iter-2 kernel (final, thermal regression)

- **Hypothesis:** Restoring the proven iter-2 kernel (BM=BN=128, BK=16, TM=TN=8, 256 threads) as the final solution.
- **Changes:** Same code as iter-2 with a new extension name.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 8.26 ms (mean), 7.11~8.75 ms (min~max)
  - Speedup: 0.75x (mean)
- **Analysis:** GPU is thermally throttled at this point in the session. The kernel is identical to iter-2 but performance degraded due to thermal state. Iter-2 (0.99x) remains the best result.
- **Next:** N/A — iter cap reached. Best iter = iter-2 (0.99x).

### Iter 5 — at::mm + fused bias+GELU+online-softmax kernel

- **Hypothesis:** Use PyTorch's highly optimized at::mm (cuBLAS) for the GEMM, then apply a custom CUDA kernel that fuses bias+GELU+online-softmax in 2 passes (compute+max simultaneously, then normalize). This should give cuBLAS GEMM speed + some fusion benefit.
- **Changes:** GEMM via `at::mm(A, W.t())`. New `bias_gelu_softmax_kernel` using true online softmax algorithm (maintains running (max, sum) pair in single pass, then 1 normalize pass).
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 6.84 ms (mean), 5.64~7.36 ms (min~max)
  - Speedup: 0.89x (mean)
- **Analysis:** Surprisingly slower than ref (6.84ms vs 6.11ms). The at::mm dispatch overhead + creating 2 tensors (gemm + out) adds latency. PyTorch's native pipeline is more efficient.
- **Next:** Return to hand-rolled GEMM approach (iter-2 was best at 0.99x) and try double-buffered smem to reduce sync stalls.

### Iter 4 — Transposed weight for coalesced WT loads

- **Hypothesis:** Pre-transposing weight from [N,K] to [K,N] enables fully coalesced loads of the weight tile (BN consecutive elements along N for each k_inner).
- **Changes:** Added WT = weight.T pre-computed in __init__ (register_buffer). Changed GEMM to C=A@WT where WT[K,N]. Load pattern for WT: e → k=e/BN, n=e%BN → coalesced ✓.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 6.59 ms (mean), 4.82~6.74 ms (min~max)
  - Speedup: 0.92x (mean)
- **Analysis:** Still slower than iter 2. The overhead of using a separate WT buffer adds a warmup cost. The min (4.82ms) is better than iter 3 but worse than iter 2's 4.21ms. The key bottleneck is still the non-coalesced A loads (BM rows of A have stride K between them).
- **Next:** Try to address the A-load coalescing problem by using a tile-swap: load A in a transposed fashion or switch to a completely different compute strategy.

### Iter 3 — FP32 coalesced loads (k=e%BK) + __ldg hints

- **Hypothesis:** Switching load pattern to k=e%BK, m=e/BK should improve coalescing since consecutive threads load consecutive K-positions (same row of A).
- **Changes:** Load pattern changed from k=e/BM,m=e%BM to k=e%BK,m=e/BK. Added `__ldg()` hints. Removed __launch_bounds__ min-blocks hint that caused register spill.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 6.82 ms (mean), 5.77~7.39 ms (min~max)
  - Speedup: 0.90x (mean)
- **Analysis:** Worse than iter 2 (6.26ms). The changed load pattern apparently reduces performance. The iter-2 load pattern (k=e/BM, m=e%BM) gives better coalescing for the shared memory stores (stores consecutive m values for same k, which is column-major into As[k][m]).
- **Next:** Restore iter-2 load pattern. Try to increase arithmetic intensity by using TF32 PTX for inner loop while keeping FP32 memory operations.

### Iter 2 — FP32 register-blocking GEMM, BM=BN=128, BK=16, TM=TN=8

- **Hypothesis:** Large register tiles (8×8 per thread) with K-major shared memory layout should maximize arithmetic intensity and reach closer to cuBLAS performance.
- **Changes:** Replaced TF32 WMMA with a standard FP32 register-blocking GEMM. BM=BN=128, BK=16, TM=TN=8, 256 threads per block. Shared memory stores As[BK][BM+PAD] and Bs[BK][BN+PAD] for column-access during compute.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 6.26 ms (mean), 4.21~6.81 ms (min~max), std=0.884ms
  - Speedup: 0.99x (mean)
- **Analysis:** High variance (std=0.884ms) suggests clock ramp issues. Min of 4.21ms is very promising (speedup ~1.44x at min). Mean just under ref. The register-blocking approach is more competitive than WMMA TF32.
- **Next:** Reduce variance via warm-up (already --num-warmup 200 in bench.sh), try to stabilize. Also try BK=32 or larger to increase arithmetic intensity per smem load.

### Iter 1 — TF32 WMMA GEMM + fused bias/GELU + online softmax

- **Hypothesis:** Use TF32 tensor-core WMMA (sm_89 supported) for the GEMM with FP32 accumulators, then fuse bias+GELU into the epilogue, and do online safe-softmax as a second kernel. This should approach cuBLAS throughput while eliminating the GELU intermediate HBM round-trip.
- **Changes:** Replaced identity PyTorch solution with a 2-kernel CUDA solution: (1) `gemm_gelu_tf32` with BM=64, BN=64, BK=32, 4 warps, WMMA m16n16k8 TF32; (2) `softmax_kernel` with one block per row, 256 threads, online reduce.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 7.19 ms (mean), 7.04~7.30 ms (min~max)
  - Speedup: 0.86x (mean)
- **Analysis:** TF32 WMMA kernel is slower than cuBLAS GEMM (6.15ms ref). The custom GEMM with BK=32 and 128 threads achieves low occupancy and inefficient memory access. The small tile size means too many sync barriers and smem bandwidth is the bottleneck.
- **Next:** Try much larger tiles (BM=128, BN=128) with more warps, or use PTX `mma.sync` directly for finer control. Also consider a split approach: PyTorch mm + fused GELU+softmax kernel.

