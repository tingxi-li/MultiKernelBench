#!/usr/bin/env python3
"""Scaffold one AKO4ALL workspace per target op (reusing AKO4ALL scaffold).

Each workspace:  ako_runs/<op>/
  solution/<op>.py   -> identity copy of the reference (baseline; optimized later)
  scripts/bench.sh   -> GPU-pinned wrapper calling AKO4ALL's bench.py against the ref
  HINTS.md, ITERATIONS.md -> copied from AKO4ALL scaffold (+ per-op hints)
  .git               -> isolated history (AKO protocol: one workspace per kernel)
"""
import os, shutil, subprocess, stat, textwrap

ROOT = "/home/lxt230026/MultiKernelBench"
AKO = f"{ROOT}/AKO4ALL"
BENCH = f"{AKO}/bench/kernelbench/bench.py"
RUNS = f"{ROOT}/ako_runs"

# op -> (category, headroom-tier, per-op note)
OPS = [
    ("relu",        "activation",    "floor",  "Pure elementwise max(x,0); HBM-bandwidth bound -> ~1x is the roofline."),
    ("sigmoid",     "activation",    "floor",  "Pure elementwise; HBM-bandwidth bound -> ~1x is the roofline."),
    ("hardsigmoid", "activation",    "floor",  "Pure elementwise clamp; HBM-bandwidth bound -> ~1x is the roofline."),
    ("elu",         "activation",    "floor",  "Elementwise, alpha=1.0 (init arg); HBM-bandwidth bound -> ~1x."),
    ("gelu",        "activation",    "floor",  "Exact GELU via erf (NOT tanh approx; tanh fails 1e-4). HBM bound -> ~1x."),
    ("swish",       "activation",    "win",    "x*sigmoid(x): eager runs sigmoid+mul as TWO passes (32GB traffic). Fuse -> ~2x."),
    ("layer_norm",  "normalization", "win",    "nn.LayerNorm over last 3 dims, affine weight+bias. Fuse reduction -> headroom."),
    ("group_norm",  "normalization", "win",    "nn.GroupNorm 8 groups, affine. Per-(batch,group) reduction over 8x512x512."),
    ("gather",      "index",         "win",    "torch.gather dim=1. idx is int64 (bench keeps int dtype after fix)."),
    ("scatter",     "index",         "risk",   "scatter-overwrite dim=1; random dup indices -> order-nondeterministic in torch."),
    ("cumsum",      "math",          "win",    "torch.cumsum dim=1, rows of length 32768. Multi-block scan with carry."),
    ("lstm",        "arch",          "floor",  "6-layer nn.LSTM (cuDNN) + Linear. ModelNew MUST build nn.LSTM+nn.Linear in same order (weights match via seeded init)."),
]

CAP = {"floor": 2, "win": 8, "risk": 3}

def write(path, content, executable=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout

wrapper = open(f"{AKO}/bench-wrapper.sh").read()

for i, (op, cat, tier, note) in enumerate(OPS):
    gpu = i % 3
    ws = f"{RUNS}/{op}"
    ref = f"{ROOT}/reference/{cat}/{op}.py"
    assert os.path.isfile(ref), ref
    # identity baseline solution = copy of reference
    write(f"{ws}/solution/{op}.py", open(ref).read())
    os.makedirs(f"{ws}/trajectory", exist_ok=True)

    # bench.sh: pin GPU, full verdict command (agents add --no-ref/--num-perf-trials for fast signal)
    bench_cmd = (f"python {BENCH} --ref {ref} "
                 f"--solution solution/{op}.py --verbose 2>&1 | tee _bench_output.txt")
    bsh = wrapper.replace("{{BENCH_COMMAND}}", bench_cmd)
    bsh = bsh.replace('cd "$(dirname "$0")/.."\n',
                      'cd "$(dirname "$0")/.."\nexport CUDA_VISIBLE_DEVICES=%d  # pinned; do not change\n' % gpu)
    write(f"{ws}/scripts/bench.sh", bsh, executable=True)

    # scaffold docs
    hints = open(f"{AKO}/HINTS.md").read().rstrip() + textwrap.dedent(f"""

    ## Workspace directives (op: {op})
    - Prefer **Triton** (no nvcc/toolkit-version risk; CUDA toolkit is 13.1 but torch is cu128).
    - `ncu` is **unavailable** on this host -> proceed analytically from runtime stats.
    - GPU **{gpu}** is pinned in `scripts/bench.sh` via `CUDA_VISIBLE_DEVICES`; do not change it.
    - Device memory 49GB. Reference: `{ref}`.
    - Tier: **{tier}**. Iteration cap: **{CAP[tier]}**. {note}
    - Fast signal: `--no-ref --num-perf-trials 20`; full verdict (real SPEEDUP): no flags.
    """)
    write(f"{ws}/HINTS.md", hints)
    write(f"{ws}/ITERATIONS.md", open(f"{AKO}/ITERATIONS.md").read())
    write(f"{ws}/.gitignore", open(f"{AKO}/workspace.gitignore").read())

    # isolated git repo + baseline commit
    if not os.path.isdir(f"{ws}/.git"):
        run(["git", "init", "-q"], ws)
        run(["git", "config", "user.email", "ako@local"], ws)
        run(["git", "config", "user.name", "AKO4ALL"], ws)
    run(["git", "add", "-A"], ws)
    run(["git", "commit", "-q", "-m", f"[scaffold] {op}: identity baseline + GPU{gpu} bench"], ws)
    print(f"  ako_runs/{op:12s} cat={cat:13s} gpu={gpu} tier={tier:5s} cap={CAP[tier]}")

print("\nScaffolded", len(OPS), "workspaces under ako_runs/")
