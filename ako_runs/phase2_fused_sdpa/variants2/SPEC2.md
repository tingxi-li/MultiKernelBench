# Phase-2 variant specification

What each arm is, what is held fixed, and — where an arm could not be built as
specified — what was done instead and why. Anything a reader would have to take
on trust belongs here rather than in the report's prose.

## 0. What carries over from Phase 1 unchanged

The measurement protocol is imported, not re-derived: `common2.py` pulls
`time_kernel`, `gate_stats`, `median_ci`, `setup_cuda_env`, the gate constants
and the distribution definitions straight out of `phase1_matmul/common.py`. A
Phase-2 millisecond and a Phase-1 millisecond are therefore the same kind of
object: median of per-process medians over 5 independent processes, fixed
**warm-up time** (2.0 s, not a fixed iteration count — the card thermally soaks
and has no steady state), L2 flushed between trials, randomized process order
with a fixed seed, one GPU, nothing else on it.

The gate is the harness's own:

    |ref - got| <= 1e-4 + 1e-4*|ref|   elementwise, against an fp32 reference

## 1. Fused op — `matmul_gelu_softmax`

Reference: `nn.Linear(8192, 8192)` on `(1024, 8192)`, then `F.gelu`
(`approximate='none'`, the **erf** form), then `softmax(dim=1)`.

`2*1024*8192*8192 == 2*2048*8192*4096`, so the fused GEMM has **exactly** the
arithmetic volume of Phase 1's GEMM. Arm G is therefore directly comparable to
Phase-1 variant D, which is the transfer check.

### 1.1 The ladder

One feature added per rung; everything else identical.

| arm | bias | GELU | softmax | what it isolates |
|---|---|---|---|---|
| `G`    | – | – | – | the matched GEMM, transplanted |
| `GB`   | ✓ | – | – | cost of a column-indexed bias in the epilogue |
| `GBG`  | ✓ | ✓ | – | cost of fusing a transcendental (`erf`) |
| `GBGS` | ✓ | ✓ | ✓ | the full op; softmax is a **second kernel** |

Softmax cannot be fused into the GEMM epilogue: the row is 8192 wide and a
`BN=128` block owns 1/64th of it, so normalizing requires cross-block
communication. Every lane pays the same second kernel. This is not a limitation
of any DSL; it is the shape.

### 1.2 Matched configuration

Phase 1's primary geometry and variant-D schedule, transplanted verbatim:
`BM=128 BN=128 BK=32 threads=256`, `arith=fp16`, `kc=2048`, `stages=3`,
`cast=precast`. Shared memory `(128*32 + 32*128)*2*3 = 49152 B`, inside
sm_89's 101376 B. Nothing here is autotuned; the shipped triton incumbent for
this cell carries 13 `@triton.autotune` configs and that search is a cost the
published runtime does not account for, so no autotuner is used anywhere.

### 1.3 The weight factor — three levels, not two

The study asks for two. A third is carried because without it the "uncached"
level is a straw man.

| level | what `run` does per call | kernel |
|---|---|---|
| `cached`   | reuse an fp16 `(K,N)` copy built once | fp16 `(K,N)` B |
| `uncached` | `W.t().contiguous().half()` every call | same kernel |
| `native`   | nothing — pass `W` as stored | fp32 `(N,K)` B, transposed, converted on the global→shared path |

`cached` and `uncached` compile to the **identical kernel** and differ only in
host-side work, which is what makes them a clean two-way factor. `native` needs
a different kernel body, so it is reported apart from the two-way comparison.

Implemented once, in `common2.weight_fn`, and shared by every lane — so the
factor cannot become a per-DSL implementation difference.

**`native` is not implemented in the two hand-written CUDA lanes.** A `(N,K)`
B operand needs a transposed WMMA fragment load / a different `ldmatrix`
staging, which is a rewrite rather than a factor level; building it would have
confounded the factor with a kernel change. `tilelang`, `triton` and `torch`
express it as a config change and carry the level.

### 1.4 Epilogue asymmetry between the two CUDA lanes

A WMMA accumulator's `frag.x[e] -> (row, col)` mapping is **not part of the API
contract**, so `cuda_noptx` cannot apply a column-indexed bias in registers; it
must stage the tile through shared memory and walk it with explicit indices.
`cuda_unlimited` wrote its own `ldmatrix`/`mma.sync` pairing and therefore knows
the mapping (`gr = m0+wm+i*16+(lane>>2)`, `gc = n0+wn+j*8+((lane&3)<<1)`), so it
can add the bias straight to the accumulator registers.

That is a real capability difference produced by abstraction level, so both are
measured:

* `epilogue=smem` — matched to what the WMMA lane is forced into. **This is the
  point used in the matched table.**
* `epilogue=regs` — this lane's native form, reported separately.

The shared-memory staging is applied to **every arm including `G`**, so the
ladder's increments are not contaminated by a staging change appearing at `GB`.
TileLang's `T.copy(Cacc, C[...])` and Triton's block `tl.store` also stage
through shared memory, so this is matched behaviour rather than a handicap.

### 1.5 Softmax kernel, held fixed across the cross-DSL ladder

One row per block, 256 threads, 32 elements per thread held in registers so the
exponentials are computed **once**, warp-shuffle reduction into a shared-memory
tree. Identical algorithm in all four lanes. Varying it is the separate F1–F4
study, not this one.

Triton is the one lane that cannot write this literally — it has no warp-shuffle
primitive. There the same algorithm falls out of a single `BLOCK_N == N` tile
(`tl.max`/`tl.sum` over one tile, exponentials live in registers between the sum
and the divide). Writing a fake shuffle tree in Triton would have measured the
fake, not the language.

## 2. Fused abstraction study — TileLang only, softmax reduction level

GEMM held fixed at the matched kernel with a `T.Parallel` bias + erf-GELU
epilogue (i.e. arm `GBG`'s kernel, unchanged). Only the reduction changes.

| arm | reduction |
|---|---|
| `F1` | `T.reduce_max` / `T.reduce_sum` over a `(1,N)` fragment |
| `F2` | manual shared-memory tree, **no** warp shuffles anywhere |
| `F3` | manual `T.shfl_down` warp reduction + smem tree; `exp` **recomputed** |
| `F4` | `F3` + per-thread local buffer caching the exponentials |

`F3 -> F4` is deliberately **not** an abstraction step — it is one algorithmic
idea (trade 32 registers per thread for one pass of transcendentals). It is
carried inside the same ladder so the report can state what fraction of the
`F1 -> F4` distance is abstraction and what fraction is that idea. `F4` is the
shipped incumbent's kernel.

The weight factor is run as a separate two-way factor, per the study spec.

## 3. SDPA

Reference: B=32, H=32, S=512, D=1024, no mask, no dropout, no scale override.

`sdpa_reference_audit.py` settled what the benchmark's denominator actually
runs: at D=1024 FlashAttention is unavailable (head-dim cap 256) but the
**mem-efficient backend is selected** — a real tiled kernel, not the naive
`math` fallback. The widely-repeated "the SDPA reference falls back to math" is
false as stated, and `TORCH_MATH` is carried as an explicit arm so the claim can
be checked rather than repeated.

### 3.1 Two algorithms, identical semantics in every lane

| algo | kernels | S materialized? |
|---|---|---|
| `K3`    | 3: `QK^T` → softmax → `PV` | **yes**, to global memory |
| `FLASH` | 1: tiled online softmax | **no** |

`K2` (fused `QK^T`+softmax, then `PV`) is **absent from the cross-DSL job list**
— the study puts the two-kernel decomposition on the TileLang algorithm axis
(`S2`), because the spec asks the cross-DSL comparison to hold two algorithms
fixed, not three. It *is* implemented as a first-class algorithm in the tilelang
lane (that is what `S2` delegates to), so it can be run cross-DSL without a code
change if that comparison is ever wanted; only the job list omits it.

`K2` reuses `K3`'s `PV` kernel builder unchanged, at `K3`'s tile, so a `K2`
vs `K3` comparison isolates exactly one thing: whether `S` is written to global.
Independently confirmed by a memory census: at `d=128` the transient peak is
768 MiB for `K2` (`P` + `O`) against 1536 MiB for `K3` (`S` + `P`) — the 1024 MiB
`(BH,S,S)` fp32 score buffer never exists in `K2` — and the two lanes' `PV`
kernels are md5-identical.

`K2` and `K3` do **not** always report the same `max_abs_err` at `(fp32,fp32)`:
5.6028e-6 vs 5.6624e-6 at `d=128`, identical at `d=1024`. This is **summation
order**, not an extra rounding step — with `sdtype=fp32` the store to `Sc` is an
exact fp32 store and rounds nothing. `K3` reduces one `(8,512)` fragment while
`K2` accumulates eight partial sums. The identical `d=1024` values rule out a
systematic extra rounding, which would have to show up at both head dims.

### 3.2 The two dtypes, fixed independently

| `sdtype` | dtype the score tensor is kept in (in `K3`, also its dtype in global memory) |
| `pdtype` | dtype the probabilities are in when they feed the `PV` matmul |

Pairs run: `(fp32,fp32)`, `(fp32,fp16)`, `(fp16,fp16)`. The mma accumulator is
always fp32. `pdtype=fp32` means `PV` cannot use fp16 tensor cores; on sm_89 an
fp32 operand into a tensor-core op becomes **tf32**, and any lane where that
happens records it explicitly rather than letting it pass as "fp32".

The tile is held **identical across the three dtype pairs** for a given
`(algo, d)`. Otherwise the dtype factor would be confounded with a tile change.

### 3.3 Head dims

`d ∈ {128, 256, 1024}`. The two smaller dims are the point: they are where a
real flash backend is available to the reference, so they show what the
reference could have been, and they are where an output accumulator of
`(block_M, d)` still fits in registers.

### 3.4 SDPA abstraction study — TWO AXES, KEPT APART

The spec is explicit that these must not be merged, and the file is arranged to
make merging them awkward.

**Within-kernel abstraction axis.** Same algorithm, same tile, only how the
kernel is written changes. This is the axis on which "abstraction costs X%" is
meaningful.

| arm | form |
|---|---|
| `S3-H`  | `T.Pipelined` KV loop, `T.gemm`, `T.reduce_*`, **one** output accumulator |
| `S3-M`  | regular KV loop, one accumulator **per d-tile**, explicit fp32→smem→fp16 layout bridge |
| `S3-MP` | `S3-M` + `T.Pipelined` over the same manual structure |
| `S3-L`  | `S3-M` + manual `T.shfl_down` warp reductions for the online softmax |

At `d=1024` a `(block_M, d)` accumulator does not fit in registers, so `S3-H`'s
single-accumulator form is forced into an **outer loop over d-tiles that redoes
the whole `QK^T` for every d-tile**. That is not a strawman — it is what the
natural high-level formulation degenerates into once `d` exceeds one
accumulator. At `d=128` there is one d-tile and the arm is a clean single pass.
The head-dim sweep is what separates "abstraction is expensive" from
"abstraction is expensive *at this shape*".

**Algorithmic decomposition axis.** `S1` (three kernels) vs `S2` (two kernels)
vs the best `S3`. Different kernel counts, materialization, V re-read counts and
occupancy — **not** an abstraction result, and the report must not label it as
one. `S1`/`S2` delegate to the cross-DSL TileLang lane's `build()`, so `S1` here
and `K3` there are the same object code and a number can be carried between the
two tables.

Tile fixed at `block_M=64, block_N=64, D_TILE=128, threads=256` for **every**
arm at **every** head dim. Shared-memory use is read off the lowered TIR
(`dyn_shared_memory_buf`) and recorded in `artifacts["shared_bytes"]`, not
computed by hand — an earlier hand-count of "about 98 KB, fits" is exactly how
the first version shipped a tile that could not be pipelined.

The tile was `block_N=128` (the shipped incumbent's shape) and had to be halved:
at 128 the un-pipelined library arm already used 100352 B of sm_89's 101376 B, so
neither `T.Pipelined`'s second buffer (133120 B) nor `S3-L`'s fp32 staging tile
(131328 B) could exist, and two of the four arms could not be built at all. The
shrink is **not free** — measured back to back in one process on an identical
body, `S3-M` at `d=128` runs 2.4023 ms at `block_N=128` against 2.6757 ms at
`block_N=64, +11.4%` — but all four arms pay it equally, which is what the axis
requires.

| arm | smem at `block_N=128` | smem at `block_N=64` |
|---|---|---|
| `S3-M` | 100352 B | 59392 B |
| `S3-H` (`d=128` / `d≥256`) | 133120 / 135168 B — will not launch | 75776 / 77824 B |
| `S3-MP` (`d=128`) | 133120 B — will not launch | 75776 B |
| `S3-L` | 131328 B — will not launch | 73984 B |

**`S3-MP` exists only at `d=128`,** and that is a measured structural result
rather than a gap. TileLang's pipeline planner requires each shared buffer to be
written by at most one statement in the loop body; the manual multi-accumulator
structure feeds `n_d_tiles` accumulators from one `(block_N, D_TILE)` V buffer
re-loaded per d-tile, which is one write at `n_d_tiles=1` and `n_d_tiles` writes
above it (`Multiple writes to overlapping buffer regions ... buffer 'V_s'`). The
arrangement the planner wants — one V buffer per d-tile — was built and compiled
to 108544 B at `d=256` (over budget by 7168 B) and 305152 B at `d=1024` (3.0×
over). The V term is `block_N · d · 2` bytes however `d` is tiled, so `D_TILE`
cannot help and only `block_N` can; the budget solves to `block_N ≤ 18`, i.e. 16,
a quarter of the KV tile the other three arms use. Shrinking the tile for one arm
would make the axis measure the tile, so it was not done. **Any statement about
`S3-MP` in the report is a statement about `d=128` alone.**

## 4. Deviations and things that are not measured

* `wcache=native` — absent from both hand-written CUDA lanes (§1.3).
* `K2` — implemented in the tilelang lane, omitted from the cross-DSL *job list*
  by design (§3.1); measured as `S2` on the algorithm axis.
* **Operand dtype must be converted on the global→shared path, not on the host.**
  The tilelang SDPA lane originally materialized fp16 copies of Q/K/V inside the
  timed `run()`, which the other three lanes do not do; at `d=128` that was up to
  50% of the measured millisecond. Q/K/V are now declared `float32` kernel
  parameters in every lane and `T.copy` (or the equivalent load) does the
  conversion. Note this is *not* free and did not simply make the lane faster:
  the kernel now reads fp32 operands from global, i.e. twice the bytes, which at
  `K3`/`d=1024` costs more than the removed host cast saved. It is done because
  cross-DSL comparability requires all four lanes to measure the same work, not
  because it is the faster arrangement.
* **A cfg-declared tile is not necessarily the compiled tile.**
  `make_sdpa_config` writes `SDPA_TILES` into `cfg.BM/BN/threads`, but each lane
  chooses its own per-`(algo, d)` tile and the tilelang FLASH `d=1024` kernel
  really runs `Br=64/threads=256`. Any tile reported in this study comes from the
  lane's own `artifacts["tile"]`, which `runner2.py` now records, never from the
  config.
* The shipped incumbent artifacts have **diverged from their own convergence
  logs** (the triton fused solution on disk is a 13-config autotuned fp16 kernel
  while its log's last kept iteration is a tf32 one at 2.91 ms). Published
  numbers are therefore re-measured rather than quoted;
  `fused_incumbent_check.py` and `sdpa_incumbent_check.py` do that in one
  process against both denominators.
* Two harness bugs were found and fixed during development, both of which
  produce clean-looking wrong numbers and are worth naming so a reader can check
  their own harness for them:
  1. `build()`'s warm-up call primed the cached-weight box with a zero dummy —
     every timed call then multiplied by zeros and timed beautifully.
  2. `box.get("src") is not W.data_ptr()` compares two large Python ints by
     **identity**; CPython does not intern those, so the test was always true
     and the `cached` arm was silently the `uncached` arm.
