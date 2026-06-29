#!/usr/bin/env python3
"""Scaffold AKO4ALL workspaces for the cross-DSL kernel port.

For every op in ako_runs/ (the 12 NPUKernelBench-matched ops already done in
Triton) this creates THREE sibling workspaces — one per target DSL:

    ako_dsl_runs/<op>/cuda_noptx/      # plain CUDA C++ (no inline PTX asm)
    ako_dsl_runs/<op>/cuda_unlimited/  # CUDA + inline PTX / every trick allowed
    ako_dsl_runs/<op>/tilelang/        # TileLang DSL

Each workspace mirrors the existing ako_runs/<op>/ layout:
    solution/<op>.py        -> identity baseline (ref copy); a generator overwrites it
    scripts/bench.sh        -> GPU-pinned wrapper around AKO4ALL's bench.py
    HINTS.md, ITERATIONS.md -> AKO scaffold + per-(op,dsl) directives
    trajectory/             -> per-iteration code + bench output

Reuses the same bench.py / reference goldens as the Triton run, so speedups are
directly comparable. bench.sh's GPU pin is OVERRIDABLE (CUDA_VISIBLE_DEVICES is
honored if already set) so the orchestrator can fan benches across the 4 GPUs.
"""
import os, stat, textwrap

ROOT = "/home/lxt230026/MultiKernelBench"
AKO = f"{ROOT}/AKO4ALL"
BENCH = f"{AKO}/bench/kernelbench/bench.py"
RUNS = f"{ROOT}/ako_dsl_runs"
CUDA_HOME = "/usr/local/cuda-13.1"   # nvcc 13.1; validated against torch cu128
ARCH = "8.9"                          # RTX 6000 Ada

DSLS = ["cuda_noptx", "cuda_unlimited", "tilelang"]

# op -> (category, tier, deterministic, note)
OPS = [
    ("relu",        "activation",    "floor", False, "Elementwise max(x,0); HBM-bandwidth bound -> ~1x roofline."),
    ("sigmoid",     "activation",    "floor", False, "Elementwise 1/(1+exp(-x)); HBM bound -> ~1x."),
    ("hardsigmoid", "activation",    "floor", False, "Elementwise clamp(x/6+0.5,0,1); HBM bound -> ~1x."),
    ("elu",         "activation",    "floor", False, "Elementwise ELU, alpha=1.0 (init arg); HBM bound -> ~1x."),
    ("gelu",        "activation",    "floor", False, "Exact GELU via erf (tanh fails 1e-4); HBM bound -> ~1x."),
    ("swish",       "activation",    "win",   False, "x*sigmoid(x): eager = 2 passes; fuse to 1 -> ~2.5x."),
    ("layer_norm",  "normalization", "win",   False, "LayerNorm last 3 dims, affine. Fused reduction."),
    ("group_norm",  "normalization", "win",   False, "GroupNorm 8 groups, affine. Per-(batch,group) reduction (8.6GB tensors)."),
    ("gather",      "index",         "win",   False, "gather dim=1; idx int64. Indexed load."),
    ("scatter",     "index",         "risk",  True,  "scatter-overwrite dim=1, dup indices -> deterministic last-wins (atomicMax). Score --deterministic."),
    ("cumsum",      "math",          "win",   False, "cumsum dim=1, rows of 32768. Chunked scan with carry."),
    ("lstm",        "arch",          "floor", False, "6-layer nn.LSTM (cuDNN floor) + projection GEMM ported to the DSL."),
]

CAP = {"floor": 2, "win": 6, "risk": 4}

DSL_NOTE = {
    "cuda_noptx":     "Plain CUDA C++ via torch.utils.cpp_extension.load_inline. "
                      "NO inline PTX `asm(...)`. Intrinsics (__expf/erff/float4/__ldg/__shfl) OK.",
    "cuda_unlimited": "CUDA via load_inline, EVERYTHING allowed: inline PTX `asm volatile`, "
                      "cp.async, vectorized ld/st, warp intrinsics, cache hints. No cuBLAS/cuDNN library offload.",
    "tilelang":       "TileLang DSL (import tilelang). JIT-compiled tile kernels.",
}


def write(path, content, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def bench_sh(op, dsl, ref, gpu, deterministic):
    is_cuda = dsl.startswith("cuda")
    det = " --deterministic" if deterministic else ""
    cuda_env = ""
    if is_cuda:
        cuda_env = textwrap.dedent(f"""\
            export CUDA_HOME={CUDA_HOME}
            export PATH="{CUDA_HOME}/bin:$PATH"
            export TORCH_CUDA_ARCH_LIST="{ARCH}"
            # per-workspace build dir (script already cd'd into the workspace) so the
            # 24 load_inline extensions never collide on build lock under parallel benches
            export TORCH_EXTENSIONS_DIR="$(pwd)/.torch_ext"
            """)
    return f"""#!/bin/bash
# AKO4ALL bench wrapper — op={op} dsl={dsl}
set -eo pipefail
cd "$(dirname "$0")/.."
# GPU pin is OVERRIDABLE: orchestrator may pre-set CUDA_VISIBLE_DEVICES to fan
# benches across GPUs; fall back to the per-workspace default {gpu} otherwise.
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{gpu}}}"
{cuda_env}
LABEL="${{1:-}}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

set +e
python {BENCH} --ref {ref} --solution solution/{op}.py --num-warmup 200{det} --verbose 2>&1 | tee _bench_output.txt
BENCH_EXIT=$?
set -e

if [ -n "$LABEL" ]; then TRAJ_DIR="trajectory/${{TIMESTAMP}}_${{LABEL}}"; else TRAJ_DIR="trajectory/${{TIMESTAMP}}"; fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"
exit $BENCH_EXIT
"""


def hints(op, dsl, ref, gpu, tier, note, deterministic):
    base = open(f"{AKO}/HINTS.md").read().rstrip()
    det = ("\n- Score with bench `--deterministic` (already in bench.sh): the reference is "
           "order-nondeterministic at duplicate indices; both sides compute last-index-wins.") if deterministic else ""
    extra = textwrap.dedent(f"""

    ## Workspace directives (op: {op}, dsl: {dsl})
    - **Target DSL: {dsl}.** {DSL_NOTE[dsl]}
    - This is a cross-DSL port of the already-optimized Triton kernel at
      `{ROOT}/ako_runs/{op}/solution/{op}.py` — use it as the correctness oracle
      and its `ako_runs/RESULTS.md` speedup as the perf bar / stop criterion.
    - `forward()` must stay glue-only (allocate/reshape/launch); all compute in the
      kernel. Verify with `utils/cheating_detection.py` (bench.py does NOT run it).
    - `ncu` unavailable -> proceed analytically from runtime stats.
    - GPU **{gpu}** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES).
    - Reference: `{ref}`. Device memory 47GB.
    - Tier: **{tier}**. Iteration cap: **{CAP[tier]}**. {note}{det}
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).
    """)
    return base + extra


def main():
    n = 0
    gpu_idx = 0
    for (op, cat, tier, det, note) in OPS:
        ref = f"{ROOT}/reference/{cat}/{op}.py"
        assert os.path.isfile(ref), ref
        for dsl in DSLS:
            gpu = gpu_idx % 4
            gpu_idx += 1
            ws = f"{RUNS}/{op}/{dsl}"
            # identity baseline solution = ref copy (real kernel overwrites it).
            # Guard: never clobber a real solution already written into a workspace.
            sol = f"{ws}/solution/{op}.py"
            if not os.path.exists(sol):
                write(sol, open(ref).read())
            os.makedirs(f"{ws}/trajectory", exist_ok=True)
            write(f"{ws}/scripts/bench.sh", bench_sh(op, dsl, ref, gpu, det), executable=True)
            write(f"{ws}/HINTS.md", hints(op, dsl, ref, gpu, tier, note, det))
            write(f"{ws}/ITERATIONS.md", open(f"{AKO}/ITERATIONS.md").read())
            n += 1
        print(f"  {op:12s} cat={cat:13s} tier={tier:5s} -> 3 DSL workspaces")
    print(f"\nScaffolded {n} workspaces under ako_dsl_runs/ ({len(OPS)} ops x {len(DSLS)} DSLs)")


if __name__ == "__main__":
    main()
