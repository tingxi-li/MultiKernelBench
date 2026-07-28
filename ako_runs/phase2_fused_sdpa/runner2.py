#!/usr/bin/env python3
"""Time ONE Phase-2 variant in a fresh process and emit JSON on stdout.

Same contract as Phase 1's runner.py -- one variant per process, identical
record schema so `phase1_matmul/analyze.py` aggregates Phase-2 raw records
without modification -- extended with `--op {fused,sdpa}`.

usage:
  python runner2.py --op fused --dsl tilelang --variant GBGS \
      [--set x_wcache=uncached] [--dist rand] [--seed 0] [--rep 0] \
      [--trials 100] [--warmup-s 2.0] [--out results/<tag>/raw/<name>.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common2  # noqa: E402
common2.setup_cuda_env()

import torch  # noqa: E402
import common  # noqa: E402  (Phase-1 protocol, via common2's sys.path insert)
import variants2  # noqa: E402


def parse_set(s: str) -> dict:
    out = {}
    for part in (s or "").split(","):
        if not part.strip():
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in ("cast", "arith", "algo"):
            out[k] = v
        elif k.startswith("x_"):
            out.setdefault("extra", {})[k[2:]] = v
        else:
            out[k] = int(v)
    return out


def op_flops(args, cfg):
    """FLOPs of the region actually timed, or 0 when a rate is meaningless.

    Returning 0 is deliberate for the softmax-only arms: the timer brackets one
    small kernel while the op's FLOP count belongs to the GEMM, so any rate
    computed from the pair is fiction (it reads ~1560 TFLOP/s on a 142-SM card).
    """
    e = (cfg.extra or {}) if hasattr(cfg, "extra") else {}
    if args.op == "sdpa":
        d = int(e.get("d", 0) or 0)
        if not d:
            return 0.0
        B, H, S = common2.S_B, common2.S_H, common2.S_S
        # QK^T is S*S*d MACs and PV is S*d*S MACs, per (batch, head).
        return 2.0 * B * H * (S * S * d + S * d * S)
    if str(e.get("soft_only", "0")) in ("1", "True", "true"):
        return 0.0
    return common2.F_FLOPS_GEMM


def time_kernel3(fn, a, b, c, num_trials, warmup_s, flush_l2=True):
    """Phase-1 `time_kernel`, arity 3. Identical discipline: fixed WARMUP TIME
    (not iterations -- the card thermally soaks and has no steady state, so a
    fixed count systematically favours whichever variant is already fast), an
    L2 flush between trials, cuda-event timing, all trials returned."""
    t0, n = time.perf_counter(), 0
    while True:
        fn(a, b, c)
        n += 1
        if n % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() - t0 >= warmup_s:
                break
    torch.cuda.synchronize()
    flusher = torch.empty(int(128e6 // 4), dtype=torch.float32, device="cuda") \
        if flush_l2 else None
    ts = []
    for _ in range(num_trials):
        if flusher is not None:
            flusher.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn(a, b, c)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    del flusher
    return ts, n


def build_fused(args, over, rec=None):
    cfg = common2.make_fused_config(args.dsl, args.variant, **over)
    # Record the config BEFORE building. A build failure otherwise leaves
    # `cfg: None`, and the record no longer says what it was trying to do --
    # which matters most for the arms that fail structurally (`S3-MP` above
    # d=128), because the head dim IS the result.
    if rec is not None:
        rec["cfg"] = cfg.to_dict()
        rec["key"] = cfg.key()
    x, W, b = common2.fused_inputs(seed=args.seed, dist=args.dist)
    built = variants2.build("fused", cfg)
    ref = None
    if not args.time_only:
        # The gate is always against the fp32 truth built from the fp32 inputs,
        # whatever the kernel's operand dtype -- the same oracle Phase 1 used.
        with torch.no_grad():
            ref = common2.fused_reference(
                x, W, b, arm=common2.reference_arm(args.variant),
                dtype=torch.float32)
    # The activation cast lives outside the timed region only for cast=precast.
    if built.x_dtype == torch.float16:
        x = x.half().contiguous()
    return cfg, built, (x, W, b), ref


def build_sdpa(args, over, rec=None):
    cfg = common2.make_sdpa_config(args.dsl, args.variant, **over)
    if rec is not None:                      # see build_fused: record before building
        rec["cfg"] = cfg.to_dict()
        rec["key"] = cfg.key()
    d = cfg.extra["d"]
    q, k, v = common2.sdpa_inputs(d, seed=args.seed, dist=args.dist)
    built = variants2.build("sdpa", cfg)
    ref = None
    if not args.time_only:
        with torch.no_grad():
            ref = common2.sdpa_reference(q, k, v, torch.float32)
    return cfg, built, (q, k, v), ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=("fused", "sdpa"))
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--geom", default="fused")   # carried for schema parity
    ap.add_argument("--dist", default="rand", choices=common.DISTRIBUTIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--warmup-s", dest="warmup_s", type=float, default=2.0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-flush-l2", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    args = ap.parse_args()

    over = parse_set(args.setstr)
    rec = {"op": args.op, "dsl": args.dsl, "variant": args.variant,
           "geom": args.geom, "dist": args.dist, "seed": args.seed,
           "rep": args.rep, "trials": args.trials, "warmup": args.warmup,
           "warmup_s": args.warmup_s, "flush_l2": not args.no_flush_l2,
           "pid": os.getpid(), "t_start": time.time()}

    try:
        fn = build_fused if args.op == "fused" else build_sdpa
        cfg, built, operands, ref = fn(args, over, rec)
        rec["cfg"] = cfg.to_dict()
        rec["key"] = cfg.key()
        rec["compile_s"] = built.compile_s
        rec["notes"] = built.notes
        rec["n_kernels"] = built.n_kernels
        # `tile` is kept because the config's tile fields and the tile a lane
        # actually compiles can disagree -- `make_sdpa_config` writes SDPA_TILES
        # into cfg.BM/BN/threads, but a lane is free to pick its own per-(algo,d)
        # tile, and the tilelang FLASH d=1024 kernel does. Recording what the
        # lane reports, not what the config assumed, is the only way a tile
        # column in the report can be trusted.
        rec["artifacts"] = {k: v for k, v in built.artifacts.items()
                            if k in ("n_regs", "n_spills", "shared_bytes",
                                     "grid", "block", "backend_detail",
                                     "wcache", "n_kernels", "algo", "tile",
                                     "score_dtype", "prob_dtype")}
        torch.cuda.synchronize()

        if ref is not None:
            with torch.no_grad():
                got = built.run(*operands)
                torch.cuda.synchronize()
            assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
            rec["out_dtype"] = str(got.dtype)
            rec["error"] = common.gate_stats(ref, got.float())
            del ref, got
            torch.cuda.empty_cache()

        times, warm_n = time_kernel3(built.run, *operands,
                                     num_trials=args.trials,
                                     warmup_s=args.warmup_s,
                                     flush_l2=not args.no_flush_l2)
        rec["warmup_iters_actual"] = warm_n
        rec["timing"] = common.summarize(times)
        # `common.summarize` divides by Phase 1's fixed GEMM FLOP count, which is
        # right for the fused op and wrong for everything else: at SDPA d=1024 it
        # reads ~6 TFLOP/s instead of ~47, and for a softmax-only run it reads
        # ~1560. Recompute it from the op actually timed, or drop it when the
        # timed region is not a whole op, so nobody reads a fabricated number.
        flops = op_flops(args, cfg)
        med = rec["timing"].get("median_ms") or 0.0
        if flops and med > 0:
            rec["timing"]["tflops_at_median"] = flops / (med * 1e-3) / 1e12
        else:
            rec["timing"].pop("tflops_at_median", None)
        rec["times_ms"] = [round(t, 5) for t in times]
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["ok"] = False
        rec["error_msg"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()

    rec["env"] = common.env_fingerprint()
    rec["t_end"] = time.time()
    if args.out:
        common.write_json(args.out, rec)
    print("###JSON###")
    print(json.dumps(rec, default=str))
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
