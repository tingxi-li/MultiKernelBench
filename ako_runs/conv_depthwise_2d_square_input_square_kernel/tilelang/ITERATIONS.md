# Iteration Log — convergence re-run (tilelang, GPU3, from identity)

Op: depthwise 3x3 conv, stride1 pad0, X(16,64,512,512) -> Out(16,64,510,510). Memory-bound
(~4.8 GFLOP over ~2.1 GB). Reference cuDNN depthwise = 3.55-3.91 ms on GPU3. Full curve in
convergence.csv.

- iter1 identity (cuDNN): 3.39 ms / 1.0x.
- Kernel: direct coalesced 9-tap. Threads map to the innermost width axis; each thread
  computes one output pixel from 9 global reads. Adjacent block-rows share 2 of 3 input
  rows -> the halo is absorbed by L2. Output-size arithmetic (H-2) lives in the builder,
  NOT forward (forward has no BinOp; detector-clean, conv2d weight read not called).
- Logged sweep: direct TW256 2.66 ms/1.47x (BEST); TW128 2.68; TW512 4.09 (too few
  blocks); shared-tiled BH8/BW64 2.67 (equal, no gain).
- NCU at best: DRAM read = 1.000 GiB = 1.00x the input tensor (L2 fully absorbs the 3x3
  halo -> input read exactly once), write 0.953 GiB = 1.95 passes total = the 2-pass HBM
  roofline. 733 GB/s effective (~81% of peak), occ 82%, L2 76% hit. Shared tiling can't
  beat this because DRAM is already 1.0x; the bound is pure HBM bandwidth.
- STOP: at the 1.95-pass roofline (ncu-confirmed) + last 2 levers didn't beat 2.66 ms.
- BEST 1.47x (2.66 ms). Matches the prior tie runtime (~2.58 ms); the prior's "2.44x" used
  the inflated GPU0/1/2 cuDNN ref (findings caveat C4) — on the honest GPU3 ref it's ~1.5x.
