#!/usr/bin/env python3
"""Time ONE variant in a fresh process and emit JSON on stdout.

One variant per process is deliberate: JIT caches, cuBLAS workspaces, allocator
state and clock history all leak between variants inside a process, and that
leakage is exactly the kind of thing that manufactures a 10% "compiler
advantage" out of nothing.

usage:
  python runner.py --dsl tilelang --variant D [--geom primary] [--dist rand]
                   [--seed 0] [--trials 100] [--warmup 200] [--rep 0]
                   [--set BM=128,BN=256,BK=64,kc=1024,stages=2,cast=in_region]
                   [--out results/raw/<name>.json] [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
common.setup_cuda_env()

import torch  # noqa: E402
import variants  # noqa: E402


def parse_set(s: str) -> dict:
    out = {}
    if not s:
        return out
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k in ("cast", "arith"):
            out[k] = v
        elif k.startswith("x_"):          # x_foo=bar -> cfg.extra['foo']
            out.setdefault("extra", {})[k[2:]] = v
        else:
            out[k] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--dist", default="rand", choices=common.DISTRIBUTIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--warmup-s", dest="warmup_s", type=float, default=0.0,
                    help="warm for this many wall-clock seconds instead of a fixed "
                         "iteration count; equalizes thermal state across variants "
                         "that differ in runtime")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--check", action="store_true",
                    help="also run the correctness/error check in this process")
    ap.add_argument("--no-flush-l2", action="store_true")
    ap.add_argument("--time-only", action="store_true",
                    help="skip the error check entirely (fastest)")
    args = ap.parse_args()

    over = parse_set(args.setstr)
    cfg = common.make_config(args.dsl, args.variant, args.geom, **over)

    rec = {
        "cfg": cfg.to_dict(), "key": cfg.key(), "dsl": cfg.dsl,
        "variant": cfg.variant, "geom": args.geom, "dist": args.dist,
        "seed": args.seed, "rep": args.rep, "trials": args.trials,
        "warmup": args.warmup, "warmup_s": args.warmup_s,
        "flush_l2": not args.no_flush_l2,
        "pid": os.getpid(), "t_start": time.time(),
    }

    try:
        A32, B32 = common.load_inputs(args.dist, args.seed, device="cuda")
        torch.cuda.synchronize()

        built = variants.build(cfg)
        rec["compile_s"] = built.compile_s
        rec["notes"] = built.notes
        rec["artifacts"] = {k: v for k, v in built.artifacts.items()
                            if k in ("n_regs", "n_spills", "shared_bytes",
                                     "grid", "block", "backend_detail")}

        # Operand preparation lives OUTSIDE the timed region only when the
        # variant declares fp16 inputs (cast=precast). For cast=in_region and
        # cast=on_load the kernel receives the fp32 tensors and pays whatever
        # conversion it needs inside the timer. That is the casting control.
        if built.input_dtype == torch.float16:
            A = A32.half().contiguous()
            B = B32.half().contiguous()
        else:
            A, B = A32, B32
        torch.cuda.synchronize()

        if not args.time_only:
            with torch.no_grad():
                ref = common.reference_fp32(A32, B32)
                got = built.run(A, B)
                torch.cuda.synchronize()
            rec["out_dtype"] = str(got.dtype)
            rec["out_shape"] = list(got.shape)
            assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
            rec["error"] = common.gate_stats(ref, got.float())
            del ref, got
            torch.cuda.empty_cache()

        times = common.time_kernel(built.run, A, B,
                                   num_warmup=args.warmup,
                                   num_trials=args.trials,
                                   flush_l2=not args.no_flush_l2,
                                   warmup_s=args.warmup_s)
        rec["warmup_iters_actual"] = getattr(common.time_kernel, "last_warmup_iters", None)
        rec["timing"] = common.summarize(times)
        rec["times_ms"] = [round(t, 5) for t in times]
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["ok"] = False
        rec["error_msg"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()

    rec["env"] = common.env_fingerprint()
    rec["t_end"] = time.time()

    js = json.dumps(rec, default=str)
    if args.out:
        common.write_json(args.out, rec)
    print("###JSON###")
    print(js)
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
