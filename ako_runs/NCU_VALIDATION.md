# ncu validation of the DSL trajectory analysis — measured, not inferred

**Goal.** The trajectory analysis' single biggest caveat was *"these rooflines are reasoned models, not `ncu`-measured."* Nsight Compute is now enabled (`/usr/local/cuda-13.1/bin/ncu`, `RmProfilingAdminOnly: 0`). This profiles the **48 already-committed kernels** (12 ops × 4 DSLs) plus a warm-L2 scatter counterfactual — no search re-run, just measurement — and adjudicates every "at floor / N-pass / L2-resident / bandwidth-bound" claim against hardware counters.

## Method (and its limits)

- **Driver** (`ncu_driver.py`) reproduces exactly one `bench.py` trial: warmup → clear 256 MB L2 (mirrors `clear_l2_cache`) → `cudaProfilerStart` → one `forward()` → `cudaProfilerStop`. `--profile-from-start off` makes ncu profile **only** the op's kernels.
- **layer_norm / group_norm** use **application replay + `--cache-control none`**: their win is a *cross-launch* L2 reuse (stats kernel leaves the row/chunk in L2, apply re-reads it), which per-kernel flushing would destroy. App-replay preserves it; the driver's own L2 clear sets the cold initial state.
- **All other ops** use **kernel replay + `--cache-control all`** (cold before every pass = the bench's per-trial clear; intra-kernel reuse survives, the flush is only pre-kernel).
- **THE ANCHOR IS DRAM BYTES.** `dram__bytes_read/write.sum` are replay- and clock-invariant — they measure exactly how many bytes crossed HBM, which is what every "N-pass / L2-residency / fusion" claim reduces to. This is the bulletproof spine.
- **Units:** all byte figures here are **GiB (÷1024³)** — ncu reports raw bytes; a re-sum in decimal GB (÷10⁹) reads ~7.4% higher (same numbers, different base). Ratios (2-pass vs 3-pass, etc.) are base-independent.
- **`%peak` is qualitative only.** ncu locks clocks to base and serializes replays, which *depresses* `dram__throughput.%peak` for **many-launch** winners (layer_norm 128–130, group_norm 514 launches) and **sub-µs** kernels (gather 11 µs, scatter). Read it as "saturated vs not," never as a number to do arithmetic with.
- **ncu wall-time is NOT a speed proxy** (locked base clock, no launch overlap, serialized). Speedups below are the committed `RESULTS.md` cuda-event numbers; ncu supplies *mechanism*, not runtime.

## Master traffic table (bytes = truth; %peak = qualitative)

Tensor sizes: elementwise/swish ≈ 5.98 GB; layer_norm x = 1.07 GB; group_norm x = 8.59 GB; cumsum ≈ 4 GB; gather/scatter = a few MB. "passes" = total DRAM / tensor.

| op | dsl | speedup | read GB | total GB | passes | DRAM% | launches | verdict |
|---|---|--:|--:|--:|--:|--:|--:|---|
| **elementwise** (relu/sig/hsig/elu/gelu) | all 4 | 0.99–1.03 | 6.00 (1×) | 11.96 | **2** (1r+1w) | 88–92 | 1 | copy floor, converged |
| **swish** | all 4 | 2.43–2.53 | 6.00 (1×) | 11.96 | **2 = FUSED** (not ~5) | 89–93 | 1 | fusion confirmed |
| layer_norm | cuda_noptx | 1.637 | **2.04 (2×)** | **3.04** | **3-pass** | 84 | 2 | **3-pass (loser)** |
| layer_norm | triton | 2.095 | **1.05 (1×)** | 1.96 | **2-pass** | 54 | 128 | 2-pass L2-resident |
| layer_norm | cuda_unlimited | 2.162 | **1.05 (1×)** | 1.95 | **2-pass** | 62 | 130 | 2-pass (multi-launch) |
| layer_norm | tilelang | 2.101 | **1.05 (1×)** | 2.07 | **2-pass** | 57 | **1** | 2-pass (cooperative 1-kernel) |
| group_norm | cuda_noptx | 0.917 | **16.0 (2×)** | **24.0** | **3-pass** | 89 | 2 | **3-pass (loser)** |
| group_norm | triton | 1.354 | 8.01 (1×) | 12.85 | ~1.5 | 82 | 514 | sub-2-pass (write also cached) |
| group_norm | cuda_unlimited | 1.281 | 8.00 (1×) | 15.49 | ~1.8 | 63 | 514 | ~2-pass |
| group_norm | tilelang | 1.323 | 8.00 (1×) | 15.49 | ~1.8 | 80 | 514 | ~2-pass |
| gather | all 4 | 1.31–1.51 | 0.008 | 0.008 | (tiny) | 67–69 | 1 | **ncu INCONCLUSIVE** |
| scatter (cold) | all 4 | 6.56–10.0 | 0.005–0.009 | — | (tiny) | **40–47** | 1–3 | not BW-bound |
| cumsum | all 4 | 1.19–1.24 | 4.00 (1×) | 7.96 | **2** (copy floor) | 88–91* | 1 | copy floor |
| lstm | all 4 | 0.97–1.0 | — | — | — | — | identical cuDNN | cuDNN closed-box |

\* tilelang cumsum 63% (its `T.Pipelined` codegen, noted in the report).

## Claim-by-claim verdicts

### CONFIRMED by DRAM bytes (the bulletproof spine)

1. **Elementwise = DSL-agnostic copy floor.** All 4 DSLs move exactly 2× the tensor (1 read + 1 write, 11.96 GB) at 88–92% peak. Spread ≤ 4% = the noise band. ✔
2. **PTX streaming stores are null — anchored on BYTES, not %peak.** On relu, cuda_unlimited (inline `st.global.cs`) reads **6.008 GiB — the *most* of the three** (cuda_noptx 6.001, triton 6.000): its signature streaming-store lever moves **no fewer bytes**, and speedup ties (~1.00×). Its %peak (88.7%) is *not above* triton's (89.5%); it does edge cuda_noptx's (88.2%), but that is inside the noise band *and* while moving more bytes, so %peak carries no signal here — the byte anchor is what makes PTX-null decisive. ✔ (The report's load-bearing "PTX is non-causal" claim — measured. Independently re-confirmed by P2b: the group_norm `__stcs` port ties inline-PTX unlimited byte-for-byte.)
3. **swish fusion is real.** All 4 move 11.96 GB = one fused pass. Un-fused `x*sigmoid(x)` would be ~5 tensor-passes (~30 GB). ✔
4. **layer_norm: cuda_noptx does 3-pass, the three winners do 2-pass.** cuda_noptx reads x **twice** (2.04 GB) → 3.04 GB total; triton/unlimited/tilelang read x **once** (1.05 GB) → ~2.0 GB total. The 1.55× traffic reduction is the L2-residency lever, made concrete. And **tilelang does it in ONE cooperative `T.sync_grid` launch** while unlimited does it in 130 multi-launches — *same lever, opposite mechanism, both 2-pass* (report claim, measured). ✔
5. **The L2-residency mechanism itself (held open to refutation, passed).** triton reads x only 1× from DRAM *despite reading it twice logically* — the apply pass's re-read hit L2, not HBM. If the mechanism were fake, triton would read x 2×; it does not. ✔
6. **group_norm: cuda_noptx 3-pass (24.0 GB) vs winners ~1.5–1.8-pass (12.85–15.49 GB).** cuda_noptx alone kept the 3× traffic model; the three siblings cut it via the L2-resident chunk pipeline. ✔
7. **lstm = cuDNN closed-box.** All 4 DSLs launch the **byte-for-byte identical** cuDNN kernel set: `gemmSN_TN_kernel` (64.5%), `elemWiseRNNcell` (31.5%), `cutlass::Kernel2` (2.4%). The 1.0× floor is cuDNN, not any DSL. ✔
8. **cumsum = copy floor.** All move 2× the tensor (7.96 GB) — the scan itself is ~free; the ~1.2× is the fixed copy ceiling. ✔

### Traffic ratio is a *ceiling*, not a runtime predictor — do NOT over-model

The layer_norm traffic ratio (noptx 3-pass / winner 2-pass = **1.55×**) is the *ceiling* on the achievable speedup gap. The bench realizes **1.28×** of it (2.16/1.64). The shortfall is exactly the serialization-depressed `%peak` of the 128-launch winners: they move 1.55× fewer bytes but at lower measured bandwidth efficiency, so runtime gains less than the byte ratio. **Traffic = qualitative ceiling; ncu %peak cannot predict the runtime** (locked clocks). An earlier attempt to reconcile speedup = efficiency×traffic over-corrects to parity and is wrong — dropped.

### Where ncu CANNOT confirm the report

9. **gather — INCONCLUSIVE.** All four DSLs show *identical* traffic (0.008 GB) and %peak (67–69%); neither traffic, occupancy, nor L2-hit separates the 1.50/1.51 winners (triton/unlimited) from the 1.31/1.33 losers (noptx/tilelang) — triton's occ is 31% but co-winner unlimited's is 15%, same as the losers. **ncu cannot see this divergence at all.** The report's specific *"triton gather hits ~97% of HBM peak via evict_last"* is **unsupported** — measured 68.6%, and the kernel is 11 µs, far too short for `%peak` to be meaningful. The coalescing story may be true, but this profiling neither confirms nor explains it. *(A dedicated micro-measurement — effective GB/s = bytes/runtime on the isolated kernel — would be needed; %peak is the wrong instrument here.)*

10. **scatter — the two sub-claims land OPPOSITELY; report the split.**
    - *"Not bandwidth-bound"* → **CONFIRMED.** Cold (bench-faithful), all DSLs sit at **40–47% peak**, not the ~89% of a saturated op. Scatter is latency/occupancy-bound, not HBM-bound.
    - *"L2-resident"* (unlimited's self-justification for full fusion) → **REFUTED under the bench.** Cold L2-hit for the fused kernel is only **31%**; the ~101% hit appears *only* in the warm counterfactual (cache-control none). Because `bench.py` clears L2 every trial, the data is **not** L2-resident when the kernel runs — so the fused kernel is fast because it's traffic-light + latency-light, **not** because of L2 residency.
    - **Honest tension the report under-weighted:** the fused kernel (unlimited) moves the **least** traffic cold (0.005 GB, 1 launch) and **still wins cold** (10× vs tilelang's partial-fusion 7.63×). So the cold data mildly *favors full fusion*, not the critics' skepticism — full fusion is legitimately best, it was just justified by the wrong reason (traffic-light, not L2-resident). The "regime-contested, lean-to-critics" hedge should soften to "full fusion wins for a different reason than claimed."

## Scope: what P1 proves and what it doesn't

ncu confirms **mechanism** — cuda_noptx *does* run 3-pass; the winners *do* run 2-pass; the levers that separate them are traffic-volume, not PTX. It does **not** prove cuda_noptx *could* have run 2-pass — that is the "artifact, not ceiling" thesis, and only **P2** (add the L2-resident lever to cuda_noptx layer_norm/group_norm and show the gap closes) tests it. P1 removes the "inferred roofline" caveat from the *mechanism* claims; P2 is what closes the central argument.

## Corrections the ncu data forces on the trajectory doc

- Drop the gather **"~97% of HBM peak"** figure — unsupported (measured 68.6%; 11 µs kernel). Mark gather's divergence **ncu-invisible**.
- Reframe scatter: **"not bandwidth-bound" confirmed, "L2-resident" refuted**; full fusion wins cold on least-traffic grounds, so soften the "regime-contested" hedge.
- Keep every traffic/pass/fusion/cuDNN claim as-is — all measured and confirmed.
- Add: **PTX-null is now measured** (unlimited elementwise %peak ≤ siblings), upgrading it from "argued from the AKO A/Bs" to "counter-confirmed."
