#!/usr/bin/env python3
"""Scaffold AKO4ALL workspaces for the 28 NEW kernels (30-list minus LN/GN),
across ALL 4 DSLs, under ako_runs/<op>/<dsl>/.

Driven by ako_runs/kernels_28_tiers.csv (the reviewed tier classification):
    number,op,category,tier,cap,deterministic,mem_watch_gb,note

Mirrors the live layout of the 12 already-optimized kernels:
    ako_runs/<op>/{cuda_noptx,cuda_unlimited,tilelang,triton}/
        solution/<op>.py   -> identity baseline = PyTorch reference copy (NO-CLOBBER guard)
        scripts/bench.sh    -> GPU-pinned bench.py wrapper (--num-warmup 200; +--deterministic if flagged)
        HINTS.md, ITERATIONS.md
        trajectory/         (gitignored)

Deliberately different from the original scaffold_dsl_runs.py:
  * RUNS -> ako_runs/ (not the defunct ako_dsl_runs/); DSLS adds `triton` (4 total).
  * OPS come from the CSV, not a hardcoded 12-op list.
  * ORACLE FIX: the PyTorch reference is the SOLE correctness oracle. The new ops
    have NO pre-existing Triton solution, so the old "port the optimized Triton
    kernel" directive would be a dangling pointer — removed.
  * Param-bearing ops (conv/linear/MHA/batchnorm/parameter...) get an explicit
    seeded-weight-init directive: ModelNew must rebuild the same nn.* layers in the
    same order so seeded init matches the reference, else correctness fails looking
    like a kernel bug.
  * No per-workspace git (these workspaces are tracked in the MAIN repo).
LN/GN are hard-asserted out so the already-optimized kernels can never be clobbered.
"""
import os, re, csv, stat, textwrap

ROOT = "/home/lxt230026/MultiKernelBench"
AKO = f"{ROOT}/AKO4ALL"
BENCH = f"{AKO}/bench/kernelbench/bench.py"
RUNS = f"{ROOT}/ako_runs"
TIERS_CSV = f"{RUNS}/kernels_28_tiers.csv"
CUDA_HOME = "/usr/local/cuda-13.1"   # nvcc 13.1; validated against torch cu128
ARCH = "8.9"                          # RTX 6000 Ada

DSLS = ["cuda_noptx", "cuda_unlimited", "tilelang", "triton"]
CAP = {"floor": 2, "win": 6, "risk": 4}
ALREADY_OPTIMIZED = {"layer_norm", "group_norm", "relu", "sigmoid", "hardsigmoid",
                     "elu", "gelu", "swish", "gather", "scatter", "cumsum", "lstm"}

# ops carrying learnable parameters must reproduce the reference's seeded init
PARAM_RE = re.compile(r"nn\.(Linear|Conv\w*|MultiheadAttention|BatchNorm\w*|"
                      r"Parameter|LSTM|GRU|RNN|Embedding|InstanceNorm\w*|"
                      r"LayerNorm|GroupNorm)")

DSL_NOTE = {
    "cuda_noptx":     "Plain CUDA C++ via torch.utils.cpp_extension.load_inline. "
                      "NO inline PTX `asm(...)`. Intrinsics (__expf/erff/float4/__ldg/__shfl) OK.",
    "cuda_unlimited": "CUDA via load_inline, EVERYTHING allowed: inline PTX `asm volatile`, "
                      "cp.async, vectorized ld/st, warp intrinsics, cache hints. No cuBLAS/cuDNN library offload.",
    "tilelang":       "TileLang DSL (import tilelang). JIT-compiled tile kernels.",
    "triton":         "Triton DSL (import triton, triton.language as tl). @triton.jit kernels; "
                      "triton.autotune allowed. No torch-op offload in the hot path.",
}


def write(path, content, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def bench_sh(op, dsl, ref, gpu, deterministic):
    """Byte-for-byte the live convention (matches the existing 12 kernels' bench.sh)."""
    is_cuda = dsl.startswith("cuda")
    det = " --deterministic" if deterministic else ""
    cuda_env = ""
    if is_cuda:
        cuda_env = textwrap.dedent(f"""\
            export CUDA_HOME={CUDA_HOME}
            export PATH="{CUDA_HOME}/bin:$PATH"
            export TORCH_CUDA_ARCH_LIST="{ARCH}"
            # per-workspace build dir (script already cd'd into the workspace) so the
            # load_inline extensions never collide on build lock under parallel benches
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


def hints(op, dsl, ref, gpu, tier, note, deterministic, mem_gb, has_params):
    base = open(f"{AKO}/HINTS.md").read().rstrip()
    cap = CAP[tier]
    floor_caveat = ""
    if tier == "floor":
        floor_caveat = ("\n- **FLOOR op:** eager already dispatches to cuBLAS/cuDNN (library-optimal). "
                        "~1x is the physical ceiling. Confirm the floor within the cap and STOP; do not "
                        "chase a speedup that isn't there.")
    det = ("\n- Score with bench `--deterministic` (already in bench.sh): reference is "
           "order-nondeterministic at duplicate indices; both sides compute last-index-wins.") if deterministic else ""
    param = ("\n- **Param-bearing op:** the reference builds learnable `nn.*` layers with SEEDED init. "
             "`ModelNew.__init__` MUST construct the same layers (same types, same args, same order) so "
             "the seeded weights match — otherwise correctness fails and looks like a kernel bug. Keep the "
             "layers as attributes; do compute in the kernel, but read weights from those layers.") if has_params else ""
    mem = ""
    try:
        if float(mem_gb) >= 4.0:
            mem = (f"\n- **Memory watch (~{mem_gb}GB):** device has 49GB but bench holds ref+solution inputs "
                   "resident simultaneously. The identity baseline MUST bench green (no OOM) before optimizing; "
                   "if it OOMs, trim the reference input shape and note it.")
    except ValueError:
        pass
    extra = textwrap.dedent(f"""

    ## Workspace directives (op: {op}, dsl: {dsl})
    - **Target DSL: {dsl}.** {DSL_NOTE[dsl]}
    - **Correctness oracle:** the PyTorch reference `{ref}` is the SOLE oracle — this op has no
      prior DSL solution. bench.py renames your `Model`->`ModelNew` and checks output vs the
      reference within tolerance.
    - **Perf bar / stop criterion:** beat PyTorch eager. Tier **{tier}**, iteration cap **{cap}**. {note}{floor_caveat}
    - `forward()` must stay glue-only (allocate/reshape/launch); all compute in the kernel.
      Verify with `utils/cheating_detection.py` (bench.py does NOT run it).{param}{mem}
    - `ncu` unavailable -> proceed analytically from runtime stats.
    - GPU **{gpu}** default in `scripts/bench.sh` (overridable via CUDA_VISIBLE_DEVICES). Device memory 49GB.
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): `--num-warmup 200` (in bench.sh).{det}
    """)
    return base + extra


def main():
    rows = list(csv.DictReader(open(TIERS_CSV)))
    n = 0
    gpu_idx = 0
    for row in rows:
        op = row["op"].strip()
        cat = row["category"].strip()
        tier = row["tier"].strip()
        note = row["note"].strip()
        mem_gb = row["mem_watch_gb"].strip()
        det = row["deterministic"].strip().lower() in ("yes", "true", "1")
        assert op not in ALREADY_OPTIMIZED, f"REFUSING to scaffold already-optimized op: {op}"
        assert tier in CAP, f"unknown tier {tier!r} for {op}"
        ref = f"{ROOT}/reference/{cat}/{op}.py"
        assert os.path.isfile(ref), f"missing reference: {ref}"
        has_params = bool(PARAM_RE.search(open(ref).read()))
        for dsl in DSLS:
            gpu = gpu_idx % 4
            gpu_idx += 1
            ws = f"{RUNS}/{op}/{dsl}"
            sol = f"{ws}/solution/{op}.py"
            if not os.path.exists(sol):                    # NO-CLOBBER guard
                write(sol, open(ref).read())
            os.makedirs(f"{ws}/trajectory", exist_ok=True)
            write(f"{ws}/scripts/bench.sh", bench_sh(op, dsl, ref, gpu, det), executable=True)
            write(f"{ws}/HINTS.md", hints(op, dsl, ref, gpu, tier, note, det, mem_gb, has_params))
            write(f"{ws}/ITERATIONS.md", open(f"{AKO}/ITERATIONS.md").read())
            n += 1
        print(f"  {op:48s} cat={cat:12s} tier={tier:5s} cap={CAP[tier]} "
              f"det={'Y' if det else '-'} params={'Y' if has_params else '-'} -> 4 DSL ws")
    print(f"\nScaffolded {n} workspaces under ako_runs/ ({len(rows)} ops x {len(DSLS)} DSLs)")


if __name__ == "__main__":
    main()
