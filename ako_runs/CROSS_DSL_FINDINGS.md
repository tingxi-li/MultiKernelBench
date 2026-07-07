# Cross-DSL findings — ceiling, trajectory, transferability

Answers three questions about optimizing the **same 12 ops across 4 DSLs**
(triton, cuda_noptx = plain CUDA no inline PTX, cuda_unlimited = CUDA + inline
PTX, tilelang) on RTX 6000 Ada, benched against the same PyTorch golden.

## Scope & method (read first)

- **Data = the committed serial-GPU3 speedup floors** in `tools/committed_baseline.csv`
  (`committed_speedup` column). Speedup is a ref/solution ratio measured in one bench;
  the memory-bound ops were re-benched **serial on GPU3** because 4 concurrent
  memory-bound benches destabilize this host's memory P-state and contaminate the
  ratio (see `tools/NCU_RUNBOOK.md` Discipline 4). Mechanism claims (N-pass,
  L2-residency, PTX-null) are **ncu-measured** (`NCU_VALIDATION.md`, `P2_LEVER_TESTS.md`).
- **Every op here is memory-bound, index, elementwise, or cuDNN-closed.** The 28
  compute-bound tensor-core ops (matmul/attention/conv) are **not yet optimized** —
  that is the frontier where a genuine ceiling could still appear (§Open frontier).

## The committed floor table (winner per op in **bold**)

| op | triton | cuda_noptx | cuda_unlimited | tilelang | best | real gap (>3%) |
|---|---|---|---|---|---|---|
| relu | 1.00 | 0.99 | 1.00 | **1.03** | tilelang | none (copy floor) |
| sigmoid | **1.01** | 1.01 | 1.01 | 1.00 | ~tie | none |
| hardsigmoid | **1.01** | 0.99 | 1.00 | 0.99 | ~tie | none |
| elu | 0.99 | 1.00 | 1.00 | **1.03** | tilelang | none |
| gelu | 1.01 | 1.00 | 1.00 | **1.03** | tilelang | none |
| swish | 2.46 | 2.45 | **2.53** | 2.43 | unlim | ≤4% (tie) |
| layer_norm | 2.10 | 1.95 | **2.16** | 2.10 | unlim | **noptx −10%** |
| group_norm | **1.33** | 1.29 | 1.28 | 1.32 | triton | ≤4% (tie) |
| gather | **1.50** | 1.33 | 1.48 | 1.31 | triton | **noptx/tilelang −12%** |
| scatter | 6.79 | **10.65** | 10.0–10.6 | 7.63 | noptx≈unlim | **triton/tilelang −28%** |
| cumsum | 1.23 | 1.20 | **1.24** | 1.19 | unlim | ≤4% (tie) |
| lstm | **1.00** | 1.00 | 0.97 | 0.98 | ~tie | none (cuDNN) |

Winner tally: triton 4–5, tilelang 3, cuda_unlimited 3, cuda_noptx 1 — **every DSL
is the sole top performer on ≥1 op.**

## Q1 — Is any DSL's ceiling higher? **No universal ceiling — demonstrated.**

No DSL is universally ≥ another. Stronger: the three post-hoc **promotions**
(layer_norm/scatter/group_norm cuda_noptx, from the P2 lever tests) flattened the
apparent noptx-laggard hierarchy *in the committed artifact*, not just in argument:

- **scatter**: noptx 6.5→**10.65** — the most-constrained DSL now *ties the top* (unlim
  ~10.6) and both hand-CUDA variants beat triton/tilelang by ~28%.
- **group_norm**: noptx 0.92→**1.29** — ties unlimited (1.28), within 3% of triton (1.33).
- **layer_norm**: noptx 1.64→**1.95** — the one gap that survives (~10% below unlim 2.16).

So the DSL with the *least* freedom (no PTX) reaches parity-or-winner on every
previously-gapped op except layer_norm. The three residual gaps are all **programming-model
fit, not hardware/PTX ceilings**, and they cut *both* directions:

- **scatter (−28% for triton/tilelang):** an arbitrary 64-bit shared-memory `atomicMax`
  winner-slab is natural in CUDA, awkward in the tile model → here the *compiler* DSLs lag.
- **gather (−12% for noptx/tilelang):** triton's `evict_last` cache-hint route is fastest
  and has no clean equivalent elsewhere → an **ergonomic** edge (a primitive triton exposes).
- **layer_norm (−10% for noptx):** the winners' async/software-pipelined schedule. triton and
  tilelang (both **no-PTX**) reach 2.10, so it is *not* a PTX ceiling — their compilers
  auto-generate the pipeline while hand-CUDA must code it and the port didn't.

None is a hardware capability wall. The hardware roofline (HBM 2-pass traffic) is
DSL-agnostic and ncu-confirmed identical across the winners.

## Q2 — Trajectory differences / DSL-unique levers

- **cuda_unlimited's defining feature (inline PTX) produced ZERO unique results** —
  triply confirmed: layer_norm matches via `__ldg`; group_norm's `__stcs` intrinsic ties
  unlimited's inline `st.global.cs.v4.f32` byte-for-byte (a matched pair); scatter's
  "winning PTX kernel" contains **no PTX at all**. The PTX was a red herring on every op.
- **triton `evict_last`** — the one genuinely-unique lever that yields a unique *result*
  (fastest gather; group_norm's write also cached → sub-2-pass at 1.5 passes vs the others' ~1.8).
- **tilelang single cooperative grid-sync kernel** — a unique *mechanism* reaching the same
  result (layer_norm 2-pass in 1 launch vs unlimited's 130); also unique *pitfalls* (fp64 at
  1/64 rate; small-grid SM starvation).
- **cuda_noptx = the manual-expression capability-floor probe** — it can reach every lever
  but by hand (host row-loops, manual chunk pipelines, `__stcs`) where triton/tilelang get it
  from the compiler. If noptx can do it, no PTX or compiler magic was required — which is why
  it is the right instrument for the ceiling question.

## Q3 — Are trajectories transferable? **Yes, demonstrated — with a clean rule.**

Three winning levers were transferred from a sibling DSL into noptx and *committed* at
matching performance:

| lever | transfer | closure |
|---|---|---|
| group_norm K=4 chunk pipeline | unlimited → noptx | **~100%** (1.29 ≈ 1.28) |
| scatter fused packed-atomic | unlimited → noptx | **~100%** (10.65 ≈ 10.6) |
| layer_norm L2-resident 2-pass | triton → noptx | **~90%** (1.95 vs 2.10–2.16; residual = schedule) |
| gather `evict_last` route | triton → tilelang | **did not transfer** (reached staging route only) |

**Rule the data supports:**
> **Algorithmic levers** (traffic-reducing structure: L2-residency, chunk-pipelining, fusion)
> transfer ~100% across all four DSLs. **Schedule-level tuning** transfers partially and needs
> DSL-native re-tuning. **DSL-native primitives** (triton `evict_last`, CUDA arbitrary shared
> atomics) don't transfer — they define the single op where that DSL wins.

## What this dataset CANNOT answer — convergence rate

**"Which DSL converges faster?" is a meaningful question — but the existing 12-op history
cannot answer it, because that run was not a controlled convergence experiment.** The
trajectories differ in iteration count for reasons that are *not* the DSL:

- **Variable, uncontrolled effort.** Each cell ran a different number of iterations, stopped at
  a different point, and was driven by analytical reasoning (no ncu in the loop), sometimes
  interrupted by session limits. Iteration count measures agent effort + search luck, not a
  DSL-intrinsic rate.
- **The stop points were search artifacts, now proven.** layer_norm/group_norm noptx were
  declared "at a physical floor" and stopped — *wrong* decisions (the three promotions lifted
  exactly those cells). So "noptx converged to 1.64 in N iters" measures where the search gave
  up, not where the DSL converges. Measuring "iterations to converge" from that history would
  make noptx look like it "converged fast" — precisely backwards, since it stopped early.
- **No uniform decision procedure.** Convergence rate only becomes a DSL property if every cell
  is driven by the *same* driver, *same* effort budget, and *same* stopping rule.

**To make it answerable**, the ncu-in-loop redo would need to be run as a controlled experiment:
identical driver + fixed effort budget + uniform stopping rule + same optimizer (model/prompt)
for every cell, from the same identity baseline, recording **speedup-vs-iteration curves** and
"iterations to within X% of final" per DSL, averaged over ops.

**Testable hypothesis** (consistent with, but not proven by, the residuals above): *same ceiling,
different convergence rate — the compiler DSLs (triton/tilelang) converge in fewer iterations
because one autotune step sweeps a large schedule space, while hand-CUDA must code each variant
by hand.* The gather (`evict_last`) and layer_norm (pipelining) residuals — the two places
noptx/tilelang lagged — are the schedule/primitive levers a compiler explores cheaply, which is
exactly what this hypothesis predicts. The redo is what would confirm or refute it.

## Open frontier — compute-bound ops

All findings above are for memory-bound / index / elementwise / cuDNN ops, where the winning
lever is algorithmic and expressible everywhere. The 28 compute-bound tensor-core ops
(matmul/attention/conv) are unoptimized. Their winning levers — tensor-core MMA, `wgmma`,
async copy, warp specialization — have genuinely different expressibility across the DSLs, so
that is where a **real capability ceiling** (not an ergonomic one) is most likely to appear, and
where both the ceiling and the convergence-rate questions have the highest information value.
