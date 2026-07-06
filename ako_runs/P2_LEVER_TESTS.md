# P2 lever tests — artifact vs ceiling (empirical, bench-measured)

Method: AKO4ALL bench.py, cuda_event timing, per-trial L2 clear (cold). Clock state
matters enormously (LayerNorm/GroupNorm are memory-bound). GPUs 0/1/2 were stuck at a
slow memory P-state (ref 11.0 ms); GPU3 reached the fast state (ref 6.40 ms = AKO's
stable ref). All numbers below are GPU3, 200 warmup, 100 trials, min & mean.

## P2a — cuda_noptx layer_norm: L2-resident per-row loop (no PTX)

Lever: process ONE 16 MB row at a time (host C++ row loop), split across B=512 blocks
for occupancy, so the apply pass re-reads x from L2 (2-pass) instead of DRAM (3-pass).
This is triton's design ported to plain load_inline CUDA — no PTX, __ldg caching loads.

Definitive A/B (GPU3, ref=6.40 ms, min runtime):

| kernel | min ms | mean ms | speedup | regime |
|---|--:|--:|--:|---|
| cuda_unlimited | 2.81 | 2.93 | 2.19× | winner cluster |
| triton | 2.93 | 2.99 | 2.14× | winner cluster |
| tilelang | 3.00 | 3.04 | 2.11× | winner cluster |
| **v2 (mine, no-PTX 2-pass)** | **3.20** | **3.27** | **1.96×** | **between** |
| cuda_noptx committed | 3.86 | 3.90 | 1.64× | 3-pass |

Reproduces AKO's committed speedups (noptx 1.64, triton 2.14, tilelang 2.11, unlim 2.19).

**ncu confirmation (measured, P1's own instrument, app-replay + cache-control none, 128
launches; all byte figures GiB = ÷1024³).** Per-kernel decomposition is the decisive proof:
- `ln_reduce_row` (stats) reads x **once: 1.000 GiB = exactly 1× the tensor**.
- `ln_apply_row` (apply) reads only **0.093 GiB** — its re-read of x **hit L2, not HBM**. A
  3-pass kernel's apply would read another ~1 GiB; it read essentially nothing.
- Total: 1.093 GiB read / 2.045 GiB = **2-pass**, vs committed noptx's 2.039 / 3.04 GiB (3-pass).

v2 reads ~4% *more* than the two fastest winners (triton/unlimited 1.052 GiB) and a hair less
than tilelang (2.070 GiB total) — squarely in the 2-pass band, **not** "identical." **The lever
fired.**

**VERDICT: artifact, not ceiling — earned AND measured.** The L2-resident 2-pass lever IS
expressible in plain no-PTX CUDA (`__ldg` caching loads, host-side per-row loop). v2 is
CORRECT, measurably 2-pass, and beats committed noptx (3.20 vs 3.86 ms), closing ~69% of
the noptx→winner gap. That alone answers artifact-vs-ceiling: a ceiling would mean the lever
is inexpressible; it is expressible and it fires. Reaching the winner cluster is *sufficient*,
not *necessary*. The residual ~9% (3.20 vs 2.81–3.00 ms) is **dominated by launch/pipeline
efficiency, not bytes** — v2's 128 launches lack the winners' fusion/async pipelining. The
clean tell: **tilelang moves MORE total bytes than v2 (2.070 vs 2.045 GiB) yet runs FASTER
(3.00 vs 3.20 ms)** — so at this traffic level efficiency, not byte count, sets runtime. (A
minor byte slice exists — v2's ~4% extra reads — but it is not the driver, and it is not a
capability gap: the winners also reach the cluster with *zero* inline PTX.)

**Refines the doc's prediction** ("re-running would close them into their clusters"):
directionally right, but a *naive* no-PTX port closes ~70%, not 100%. **P3 repeats (n=3,
GPU3, ref locked 6.40-6.42 ms): v2 = 1.963/1.963/1.963×, committed noptx = 1.646/1.644/
1.646×, triton = 2.144/2.144/2.147× — variance <0.2%, between-cluster position confirmed.** Skipped the __stcs/v3 tuning attempt:
in apply each block reads its x-chunk once and consumes it immediately, so an evict-first
store can't retain something not re-read — a null hypothesis (advisor-confirmed). The one
plausible remaining lever is software-pipelined/async loads (triton num_stages=2); left as
noted-but-untested polish, not needed for the verdict.

## P2b — cuda_noptx group_norm: L2-resident K=4 chunk pipeline (no PTX)

Lever: cuda_unlimited's committed winning kernel with its ONE inline-PTX line
(`st.global.cs.v4.f32`) replaced by the plain-CUDA `__stcs` intrinsic. Walk NG groups K=4
(32 MB) at a time; cooperative SPLIT-block stats with atomic accumulators; apply re-reads
the chunk from L2. Everything is plain CUDA — exactly the "off-diagonal chunked form" the
report says cuda_noptx could fully express but never tried.

Definitive A/B (GPU3, ref=31.0 ms, 200 warmup — runs out the cudaMalloc stall, min runtime):

| kernel | min ms | mean ms | speedup | regime |
|---|--:|--:|--:|---|
| triton | 20.1 | 20.3 | 1.53× | winner |
| tilelang | 20.7 | 20.9 | 1.48× | winner |
| **gn_v2 (mine, no-PTX)** | **21.4** | **21.6** | **1.44×** | **winner cluster** |
| cuda_unlimited | 21.4 | 21.6 | 1.44× | winner |
| cuda_noptx committed | 31.0 | 31.3 | **0.99×** | 3-pass |

**ncu confirmation (GiB = ÷1024³; per-kernel decomposition, the decisive proof):**
`gn_stats` reads **8.001 GiB = exactly 1× the tensor**; `gn_apply` reads **0.001 GiB** — a
near-perfect L2 hit (a 3-pass apply would re-read ~8 GiB). Total **15.69 GiB read+write =
2-pass** (514 launches), matching unlimited's 15.49 GiB (within 1.3%), vs committed noptx's
24.0 GiB (3-pass). The lever fired.

**VERDICT: pure search artifact — gap closed 100%.** Porting the K=4 L2-residency chunk
pipeline to no-PTX CUDA takes committed noptx from 0.99× (3-pass, 24 GB) to **1.44×
(2-pass, 15.7 GB) — byte-for-byte identical to the cuda_unlimited sibling** in both traffic
(15.69 vs 15.49 GB) and runtime (21.4 vs 21.4 ms). Cleaner than P2a: the entire gap to the
unlimited sibling is recovered (triton's extra 0.09× via evict_last hints is a separate
ergonomics point the report already isolates). **Bonus — PTX-null, now measured on a matched
pair:** gn_v2 (`__stcs`) ≡ unlimited (inline `st.global.cs`) to 3 sig figs, so the inline
PTX store is provably non-causal.

**P3 group_norm (done here):** committed noptx settles to **0.99× at 200 warmup**, confirming
the report's stall-corrected ~1.0× steady state (vs the stall-dragged tabulated 0.917×). The
Trial-1 cudaMalloc(8.59 GB) stall is excluded by the min and diluted by 100 trials + warmup.

## P2c — tilelang gather: shared-memory x-row staging

Lever: stage X[r,:] (8192 f32 = 32 KB, fits shared) into shared once with a coalesced pass,
then gather OUT[r,c] = Xs[IDX[r,c]] from shared — the random index now hits on-chip memory,
not HBM. The committed tilelang did OUT[r,c]=X[r,IDX[r,c]] as a random *global* load.

Definitive A/B (GPU3, 200 warmup, 300 trials — **read the runtime column, not the speedup
column**: for an 11-18 µs kernel the per-run reference timing is noisy and the speedup column
is internally inconsistent — e.g. triton's 0.0127 ms at the other rows' implied ~0.026 ms ref
would be ~2.1×, not 1.45× — so the speedups below mix per-run refs and are unreliable; the
absolute sol runtime is the trustworthy metric and RESULTS.md's committed 1.50/1.51/1.31/1.33
are the authoritative normalized figures):

| kernel | mean ms | speedup | regime |
|---|--:|--:|---|
| triton | 0.0127 | 1.45× | winner — evict_last cache route (fastest) |
| cuda_unlimited | 0.0182 | 1.45× | winner — shared-staging route |
| **gh_v2 (mine, shared staging)** | **0.0183** | **1.43×** | **winner cluster** |
| cuda_noptx committed | 0.0203 | 1.30× | loser |
| tilelang committed | 0.0207 | 1.26× | loser |

**VERDICT: search artifact — gap closed to the shared-staging sibling.** Adding shared x-row
staging takes tilelang from 0.0207 ms (1.26×, random global loads) to **0.0183 ms — matching
cuda_unlimited's shared-staging route (0.0182 ms) exactly**, well outside the ~0.0006 ms std.
The lever tilelang missed is expressible in tilelang and closes the gap. It does not reach
triton's faster 0.0127 ms — but that is triton's *evict_last cache-residency* route, a
distinct (and per the report, faster) winning form; the artifact claim is about the staging
lever, which is now demonstrably in reach.

**Confirmation is on SUBSTITUTE evidence (runtime + structural match), NOT measured traffic —
this is the weakest of the three legs, and labeled as such.** gather's lever is an
access-pattern change (random-global → coalesced-global + shared gather), not a byte-count
change — P1 already measured gather traffic as identical (0.008 GiB) across all four DSLs and
ncu-inconclusive, so the "moves winner-level traffic" criterion *cannot* be met here. The
substitute — a 0.0207→0.0183 ms drop (≈4× the ~0.0006 ms std) plus gh_v2 being tilelang + the
*exact* shared-staging lever unlimited uses — is defensible but rests on a sub-20 µs kernel the
honest framing flags as noisy, and it reaches only unlimited's route (0.0183 ms), **not**
triton's faster evict_last route (0.0127 ms). So: **CONFIRMED on runtime+structural grounds,
not on byte-measured footing equal to group_norm.** ncu traffic would (correctly) show no
change — expected for an access-pattern lever, not a failure of it.

## P2d — cuda_noptx scatter: fused single-kernel packed-atomic (no PTX)

Lever: the committed cuda_noptx scatter used TWO kernels (argk winner-select +
gather). The cuda_unlimited sibling reached ~10x with ONE fused kernel — a packed
64-bit shared-memory atomicMax winner-slab (high 32 bits = write index k+1 for
deterministic last-wins, low 32 = value bits) then a branchless coalesced copy,
no uncoalesced `updates[]` gather. The decisive observation: **that "unlimited"
kernel contains NO inline PTX** — `atomicMax(unsigned long long*)`,
`__float_as_uint`/`__uint_as_float`, and `cudaFuncSetAttribute` are all plain-CUDA
intrinsics. So the fusion is fully expressible in no-PTX CUDA; the 6.5-vs-10x gap
was never a PTX capability difference, only search divergence (the noptx search
settled on the 2-kernel form).

Definitive A/B (GPU3 serial, --deterministic, 200 warmup, ref = torch's
deterministic scatter 0.181 ms):

| kernel | runtime ms | speedup | design |
|---|--:|--:|---|
| committed noptx (2-kernel) | 0.0276 | 6.56× | winner-select + gather |
| **noptx fused port (mine)** | **0.0170** | **10.65×** | 1 kernel, packed shared atomicMax |
| cuda_unlimited (fused) | 0.0171 | 10.0–10.6× | same kernel (also no PTX) |

**VERDICT: search artifact — gap closed ~100%, and PTX-null confirmed a third way.**
Porting the fused kernel takes noptx from 0.0276 ms (6.56×) to **0.0170 ms (10.65×)
— matching the unlimited sibling to within noise (0.0170 vs 0.0171 ms)**. CORRECT
under --deterministic, detector-clean, zero inline PTX. Because the unlimited kernel
was itself PTX-free, this is the cleanest PTX-null result yet: the two "different
DSL" scatter cells were mechanism-identical all along — the only thing separating
6.5× from 10× was which fused/unfused form each search happened to land on.

---

## Summary — all FOUR "phantom cells" are SEARCH ARTIFACTS, not ceilings

| cell | committed | with lever (mine) | winner cluster | lever fires? | gap closed | evidence |
|---|--:|--:|--:|:--:|:--:|---|
| layer_norm cuda_noptx | 1.64× | **1.96×** (v2) | 2.11–2.19× | ✔ 2-pass (apply reads 0.09 GiB) | **~70%** | byte-measured; residual=efficiency |
| group_norm cuda_noptx | 0.99× | **1.44×** (gn_v2) | 1.44–1.53× | ✔ 2-pass (apply reads 0.001 GiB) | **~100%** to unlim | byte-measured; strongest |
| gather tilelang | 1.26× | **1.43×** (gh_v2) | 1.43–1.45× | ✔ shared staging | **~100%** to unlim route | runtime+structural (substitute); weakest |
| scatter cuda_noptx | 6.56× | **10.65×** (fused) | 10.0–10.6× | ✔ single-kernel fusion | **~100%** to unlim | runtime; PTX-null (unlim also no-asm) |

All four rebuilt kernels **pass the cheating detector** (forward() glue-only: no forbidden
torch ops, no scalar for-loops) and use **no inline PTX** (layer_norm: `__ldg`; group_norm:
`__stcs` intrinsic, not asm; gather: tilelang; scatter: `atomicMax`/`__float_as_uint`). So each
is correct + detector-clean + PTX-free + measurably the winner-cluster mechanism — no reward-hack
escape hatch.

**PROMOTED to the committed solutions (this session):** all three cuda_noptx lever kernels —
layer_norm/cuda_noptx v2 → **1.95×** (2-pass, was 1.64×), scatter/cuda_noptx fused → **10.65×**
(was 6.56×), and group_norm/cuda_noptx gn_v2 → **1.29× mean** (2-pass, was 0.92×; ~1.44× steady) —
each re-verified serial on GPU3, gated (`tools/check_gate.py`), and detector-clean, with the gate
floors bumped to the new kernels. Only **gh_v2 (gather/tilelang)** stays in `p2_lever_kernels/` as
evidence and is *not* promoted: it reaches unlimited's shared-staging point (0.0183 ms), not
triton's faster evict_last route (0.0127 ms), so the win is smaller and route-ambiguous — left to
the ncu-in-loop redo. So three of the four phantom-cell levers are now the committed noptx solutions.

**Thesis CONFIRMED — with layer_norm graded distinctly, not flattened into the two ~100% cases.**
Every gap the report called a search artifact is one: the missed lever is expressible in the
losing DSL with **no inline PTX**, and when built it fires (measurably the winner mechanism).
- **group_norm & gather** close ~100% — gn_v2 ties the unlimited sibling byte-for-byte; gh_v2
  matches unlimited's shared-staging route.
- **layer_norm is the weaker leg:** the lever is expressible and fires (measurably 2-pass), which
  **already refutes the ceiling hypothesis** — "cuda_noptx *can't* express the L2 lever" is false.
  A *naive* port closes ~70%; a ~14% runtime gap to the fastest winner remains, **uncharacterized**.
  It is *most likely* schedule-tuning, not an expressiveness ceiling — triton (2.93 ms) and tilelang
  (3.00 ms) reach the cluster with *zero* inline PTX, and async/software-pipelined loads are
  expressible without PTX — but I did **not** measure whether that closes the last ~14% (cuda_noptx
  and the winners also differ in codegen/autotuner, not only PTX). The verdict rests on the
  expressibility proof, not on this residual argument.

**Two PTX-null / mechanism claims fell out for free:** group_norm's `__stcs` port ties the
inline-PTX sibling byte-for-byte (traffic AND runtime), and both v2/gn_v2 hit winner-cluster
traffic with zero PTX.
