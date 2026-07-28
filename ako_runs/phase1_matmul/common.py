"""Phase-1 matched-GEMM study — shared infrastructure.

Every variant in every DSL plugs into the API defined here, so that the only
thing differing between two measured numbers is the thing under test.

Shape is the benchmark's own:  C = A @ B,  A:(M,K) fp32, B:(K,N) fp32,
M=2048, K=8192, N=4096  (reference/matmul/standard_matrix_multiplication.py).

Measurement discipline implemented here:
  * absolute runtime is primary; speedup is derived, never measured directly
  * identical *saved* input tensors for every variant (inputs/*.pt)
  * cuda-event timing, L2 thrashed before every trial (harness-faithful)
  * compile time measured and reported separately from execution time
  * one variant per process; the driver randomizes order and repeats processes
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(HERE, "inputs")
RESULTS_DIR = os.path.join(HERE, "results")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")

# ---------------------------------------------------------------- problem ---
M, K, N = 2048, 8192, 4096
FLOPS = 2.0 * M * N * K  # 137.44 GFLOP

# The harness oracle: utils/correctness.py:34 / AKO bench.py:501 both use
#   |ref - new| <= atol + rtol*|ref|   with atol == rtol == 1e-4 for fp32.
GATE_ATOL = 1e-4
GATE_RTOL = 1e-4

# torch.rand is U[0,1): E[x^2] = 1/3, so RMS = 1/sqrt(3). The zero-mean control
# distribution is randn scaled to that same RMS, per the precision-control spec.
RAND_RMS = 1.0 / math.sqrt(3.0)

DISTRIBUTIONS = ("rand", "randn")

# ------------------------------------------------------------- geometries ---
# Primary matched geometry is the one specified for the study. The secondary
# point is the geometry the incumbent tilelang winner actually used, carried so
# that no conclusion is hostage to a single tile shape.
GEOM_PRIMARY = dict(BM=128, BN=128, BK=32, threads=256)
# Wider N tile, so each block streams (128+256)*K instead of (128+128)*K and the
# grid is 256 blocks instead of 512 -- ~1.6 GB through L2 per launch instead of
# ~2.15 GB. BK stays 32: at BK=64 a 3-stage pipeline needs
# (128*64 + 64*256)*2*3 = 147456 B of shared memory and sm_89 allows 101376 B,
# so BK=64 cannot host the matched stages=3 point at all.
GEOM_SECONDARY = dict(BM=128, BN=256, BK=32, threads=256)
# The incumbent tilelang winner's exact shape. It only fits at stages<=2
# ((128*64 + 64*256)*2*2 = 98304 B), so it is NOT a matched point for variant D
# and is carried separately rather than in the matched table.
GEOM_INCUMBENT = dict(BM=128, BN=256, BK=64, threads=256)
GEOMS = {"primary": GEOM_PRIMARY, "secondary": GEOM_SECONDARY,
         "incumbent": GEOM_INCUMBENT}


def smem_bytes(BM, BN, BK, stages, elem_bytes=2):
    """Shared memory a stages-deep double buffer needs. sm_89 allows 101376 B."""
    return (BM * BK + BK * BN) * elem_bytes * stages


SM89_MAX_SMEM_PER_BLOCK = 101376

# ---------------------------------------------------------------- variants ---
# A  fp32, no tensor cores, full-K accumulation, DSL-native pipelining
# B  fp16 tensor cores -> fp32 accumulator, full-K chain, pipeline off (1 stage)
# C  B + KC=2048 chunk flush into a second fp32 accumulator
# D  C + 2-3 stage software pipeline
VARIANT_SPECS = {
    "A": dict(arith="fp32", kc=0, stages=1, label="fp32 / full-K / native pipe"),
    "B": dict(arith="fp16", kc=0, stages=1, label="fp16 TC / full-K / no pipe"),
    "C": dict(arith="fp16", kc=2048, stages=1, label="fp16 TC / KC=2048 / no pipe"),
    "D": dict(arith="fp16", kc=2048, stages=3, label="fp16 TC / KC=2048 / 3-stage pipe"),
}
DSLS = ("tilelang", "triton", "cuda_noptx", "cuda_unlimited")

# --- TileLang-only abstraction study (reported separately from the cross-DSL
# transfer study, because abstraction level, algorithm and hardware instruction
# path are otherwise confounded).
#
# Everything below is held constant: fp16 operands, fp32 output, KC=2048,
# BM=128 BN=128 BK=32, 256 threads, pre-cast inputs. Only the abstraction level
# of the inner-K implementation changes.
#
#   H1 vs H2  isolates the compiler's software pipeline
#   H2 vs M1  measures high-level scheduling overhead with pipelining disabled
#   H1 vs M2  compiler-generated vs manually expressed pipeline
#   M1 vs S1  the tensor-core contribution -- a HARDWARE control, not an
#             abstraction measurement, since S1 removes tensor cores entirely
ABSTRACTION_SPECS = {
    "H1": dict(arith="fp16", kc=2048, stages=3, level="TL-H",
               label="T.Pipelined(num_stages=3) + T.copy + T.gemm"),
    "H2": dict(arith="fp16", kc=2048, stages=1, level="TL-H",
               label="same, num_stages=1"),
    "M1": dict(arith="fp16", kc=2048, stages=1, level="TL-M",
               label="regular K loop, explicit T.copy, barriers, T.gemm"),
    "M2": dict(arith="fp16", kc=2048, stages=2, level="TL-M",
               label="explicit double-buffered shared storage + sync around T.gemm"),
    "S1": dict(arith="fp32", kc=2048, stages=1, level="TL-SIMT",
               label="same blocking and fp32 chunking, scalar/thread-level FMA"),
}

# kc == 0 means "one accumulator across the whole K extent" (no chunk flush).

CAST_MODES = ("precast", "in_region", "on_load")
# precast    fp16 operands materialized outside the timed region
# in_region  fp32 operands, .half() executed inside the timed region
# on_load    fp32 operands in global memory, converted during the smem load


@dataclass
class Config:
    dsl: str
    variant: str
    M: int = M
    N: int = N
    K: int = K
    BM: int = 128
    BN: int = 128
    BK: int = 32
    threads: int = 256
    kc: int = 0
    stages: int = 1
    arith: str = "fp16"
    cast: str = "precast"
    extra: dict = field(default_factory=dict)

    @property
    def input_dtype(self) -> torch.dtype:
        if self.arith == "fp32":
            return torch.float32
        return torch.float16 if self.cast == "precast" else torch.float32

    def key(self) -> str:
        base = f"{self.dsl}.{self.variant}.{self.BM}x{self.BN}x{self.BK}.kc{self.kc}.s{self.stages}.{self.arith}.{self.cast}"
        if self.extra:
            base += "." + ".".join(f"{k}{v}" for k, v in sorted(self.extra.items()))
        return base

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["input_dtype"] = str(self.input_dtype)
        return d


@dataclass
class Built:
    """What a DSL module returns from build()."""
    run: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    compile_s: float
    input_dtype: torch.dtype
    artifacts: dict = field(default_factory=dict)
    notes: str = ""


def make_config(dsl: str, variant: str, geom: str = "primary", **over) -> Config:
    spec = VARIANT_SPECS.get(variant) or ABSTRACTION_SPECS[variant]
    g = dict(GEOMS[geom])
    cfg = Config(dsl=dsl, variant=variant, arith=spec["arith"], kc=spec["kc"],
                 stages=spec["stages"], **g)
    for k, v in over.items():
        if k == "extra":
            cfg.extra.update(v)
        else:
            setattr(cfg, k, v)
    return cfg


# ------------------------------------------------------------------ inputs ---
def _gen(dist: str, seed: int, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    if dist == "rand":
        A = torch.rand(M, K, generator=g, device=device)
        B = torch.rand(K, N, generator=g, device=device)
    elif dist == "randn":
        # zero-mean, scaled to the same RMS as torch.rand
        A = torch.randn(M, K, generator=g, device=device) * RAND_RMS
        B = torch.randn(K, N, generator=g, device=device) * RAND_RMS
    else:
        raise ValueError(dist)
    return A, B


def input_path(dist: str, seed: int) -> str:
    return os.path.join(INPUTS_DIR, f"{dist}_seed{seed}.pt")


def save_inputs(dist: str, seed: int) -> str:
    os.makedirs(INPUTS_DIR, exist_ok=True)
    p = input_path(dist, seed)
    if not os.path.exists(p):
        A, B = _gen(dist, seed)
        torch.save({"A": A, "B": B, "dist": dist, "seed": seed}, p)
    return p


def load_inputs(dist: str, seed: int, device="cuda"):
    """Byte-identical operands for every variant.

    Uses the saved tensor file when it exists (the timing seeds), otherwise
    regenerates deterministically from the seed (the accuracy campaign, where
    40 saved pairs would be 8 GB of disk for no added rigour -- generation is
    bit-reproducible from the seed on this host).
    """
    p = input_path(dist, seed)
    if os.path.exists(p):
        d = torch.load(p, map_location="cpu")
        A, B = d["A"], d["B"]
    else:
        A, B = _gen(dist, seed)
    return A.to(device, non_blocking=True), B.to(device, non_blocking=True)


def reference_fp32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """The benchmark's own golden: torch.matmul on fp32 CUDA tensors."""
    return torch.matmul(A, B)


def reference_fp64(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Ground truth. Lets us separate 'kernel error' from 'the oracle's own error'."""
    return torch.matmul(A.double(), B.double())


# ------------------------------------------------------------------ timing ---
def clear_l2(device=None):
    """Same L2 thrash the AKO harness uses (bench.py:67) so numbers stay comparable."""
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device or "cuda")
    dummy.fill_(42)
    del dummy


def time_kernel(fn, A, B, num_warmup=200, num_trials=100, discard_first=1,
                device=None, flush_l2=True, warmup_s=0.0) -> list[float]:
    """cuda-event timing, one fresh event pair per trial, L2 thrashed before each.

    Warmup: these cards idle at 210 MHz with persistence off, boost to ~3.1 GHz,
    then throttle back as they heat. Measured on this host for a ~1 ms GEMM, the
    per-process median rises monotonically with warmup depth
    (50/200/500/1000 iters -> 0.886/0.923/0.939/1.023 ms) while the spread across
    independent processes falls (27.6/10.2/4.9/2.4%). There is no true steady
    state; there is a thermal soak.

    `warmup_s` (preferred) warms for a wall-clock budget instead of a fixed
    iteration count. That matters because variants differ ~4.5x in runtime here,
    so a fixed iteration count delivers ~4.5x more heat before the slow variant
    is measured than before the fast one -- which would bias the comparison in
    favour of whichever variant is already fast. Equal time ~ equal energy ~
    comparable thermal state.
    """
    if device is None:
        device = torch.cuda.current_device()
    out = []
    with torch.cuda.device(device), torch.no_grad():
        if warmup_s > 0:
            t0 = time.perf_counter()
            n = 0
            while True:
                fn(A, B)
                n += 1
                if n % 8 == 0:
                    torch.cuda.synchronize(device=device)
                    if time.perf_counter() - t0 >= warmup_s:
                        break
            out_warm_iters = n
        else:
            for _ in range(num_warmup):
                fn(A, B)
            out_warm_iters = num_warmup
        torch.cuda.synchronize(device=device)
        time_kernel.last_warmup_iters = out_warm_iters
        torch.cuda.empty_cache()
        for t in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device)
            if flush_l2:
                clear_l2(device)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(A, B)
            e.record()
            torch.cuda.synchronize(device=device)
            if t >= discard_first:
                out.append(s.elapsed_time(e))
    return out


def summarize(times: list[float]) -> dict:
    ts = sorted(times)
    n = len(ts)

    def q(p):
        if n == 1:
            return ts[0]
        i = p * (n - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return ts[lo] + (ts[hi] - ts[lo]) * (i - lo)

    med = q(0.5)
    return {
        "n": n,
        "median_ms": med,
        "mean_ms": statistics.fmean(ts),
        "std_ms": statistics.stdev(ts) if n > 1 else 0.0,
        "min_ms": ts[0],
        "max_ms": ts[-1],
        "p05_ms": q(0.05),
        "p25_ms": q(0.25),
        "p75_ms": q(0.75),
        "p95_ms": q(0.95),
        "tflops_at_median": FLOPS / (med * 1e-3) / 1e12 if med > 0 else 0.0,
    }


def median_ci(values: list[float], conf=0.95) -> dict:
    """Non-parametric CI on the median of independent per-process medians.

    With n=5 processes the order-statistic interval is [min, max] at ~93.75%
    two-sided, so we report the full range alongside a t-interval on the mean of
    the medians. Both are reported; neither is claimed to be more than it is.
    """
    v = sorted(values)
    n = len(v)
    med = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    out = {"n_processes": n, "median_of_medians_ms": med,
           "min_ms": v[0], "max_ms": v[-1], "process_medians_ms": v}
    if n > 1:
        m = statistics.fmean(v)
        sd = statistics.stdev(v)
        # t_{0.975} for small n
        tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
        half = tcrit * sd / math.sqrt(n)
        out.update({"mean_of_medians_ms": m, "sd_ms": sd,
                    "ci95_lo_ms": m - half, "ci95_hi_ms": m + half,
                    "rel_spread_pct": 100.0 * (v[-1] - v[0]) / med if med else 0.0})
    return out


# ------------------------------------------------------------------- error ---
def gate_stats(ref: torch.Tensor, got: torch.Tensor, truth: Optional[torch.Tensor] = None) -> dict:
    """Error of `got` against the harness oracle `ref`, plus optional fp64 truth.

    Returns absolute-error moments/quantiles and the fraction of elements that
    violate the harness gate |ref-got| <= atol + rtol*|ref|.
    """
    r = ref.double().reshape(-1)
    g = got.double().reshape(-1)
    d = (r - g).abs()
    budget = GATE_ATOL + GATE_RTOL * r.abs()
    fail = (d > budget)
    qs = torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64, device=d.device)
    quant = torch.quantile(d.float(), qs.float()).double() if d.numel() < 2**24 else None
    if quant is None:
        # torch.quantile caps at 2^24 elements; subsample deterministically.
        idx = torch.arange(0, d.numel(), max(1, d.numel() // (2**23)), device=d.device)
        quant = torch.quantile(d[idx].float(), qs.float()).double()
    out = {
        "max_abs_err": d.max().item(),
        "mean_abs_err": d.mean().item(),
        "signed_mean_err": (g - r).mean().item(),   # bias of kernel vs oracle
        "err_q50": quant[0].item(),
        "err_q90": quant[1].item(),
        "err_q99": quant[2].item(),
        "err_q999": quant[3].item(),
        "err_q100": quant[4].item(),
        "pct_elems_failing_gate": 100.0 * fail.sum().item() / d.numel(),
        "gate_pass": bool(fail.sum().item() == 0),
        "ref_abs_mean": r.abs().mean().item(),
        "budget_mean": budget.mean().item(),
    }
    if truth is not None:
        t = truth.double().reshape(-1)
        out["oracle_max_abs_err_vs_fp64"] = (t - r).abs().max().item()
        out["kernel_max_abs_err_vs_fp64"] = (t - g).abs().max().item()
        out["oracle_signed_mean_err_vs_fp64"] = (r - t).mean().item()
        out["kernel_signed_mean_err_vs_fp64"] = (g - t).mean().item()
    return out


# -------------------------------------------------------------------- misc ---
def env_fingerprint() -> dict:
    import subprocess
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    try:
        sm_clk = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader", "-i", str(dev)],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        sm_clk = "n/a"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "total_mem_gb": round(props.total_memory / 2**30, 1),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "clocks_sm_max_temp": sm_clk,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def setup_cuda_env():
    """nvcc is not on PATH on this host; the CUDA lanes need it for load_inline."""
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.1")
    binp = "/usr/local/cuda-13.1/bin"
    if binp not in os.environ.get("PATH", ""):
        os.environ["PATH"] = binp + ":" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(HERE, ".torch_ext"))


def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
