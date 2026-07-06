#!/usr/bin/env python3
"""
ncu_driver — reconstruct ONE bench.py trial under Nsight Compute, for
DIRECTION-PICKING (not timing).

Why this exists
---------------
The 12-op optimization pass picked its per-iteration direction from *analytical*
rooflines, never live ncu. That is fine for verifying a "stop at floor" decision
(the endpoint was later ncu-checked in NCU_VALIDATION.md) but NOT for the
"continue — which lever next?" decision, which shapes the search path. This
driver makes that decision measured instead of reasoned.

Two subcommands
---------------
  run   : load ref + solution EXACTLY as bench.py does (imports bench.py's own
          loaders so it can't drift), warm up, clear L2 (256 MB, bench-faithful),
          then run ONE forward() bracketed by cudaProfilerStart/Stop. Meant to be
          launched UNDER ncu with `--profile-from-start off` so only the op's
          kernels are profiled. Writes a meta.json (tensor byte sizes) alongside.

  parse : read the CSV ncu emitted (`--csv --log-file <f>`) and print a bottleneck
          report built ONLY from replay-invariant, trustworthy counters:
          DRAM bytes -> passes (THE anchor), L2 hit rate, sectors/request
          (coalescing), achieved occupancy, launch count. Throughput %peak is
          shown ONLY under a "qualitative — saturated-or-not" banner and is NEVER
          the steering metric (ncu locks clocks + serializes replays, so %peak is
          depressed for many-launch/sub-us kernels — see NCU_VALIDATION.md).

Replay mode is the CALLER's job (ncu_profile.sh): app-replay + --cache-control
none for layer_norm / group_norm (their win is cross-launch L2 reuse that a
per-kernel flush would destroy); kernel-replay + --cache-control all otherwise.
"""

import argparse
import csv
import importlib.util
import json
import os
import sys

# --- import bench.py's loaders so we load solutions/inputs identically ---------
_BENCH = "/home/lxt230026/MultiKernelBench/AKO4ALL/bench/kernelbench/bench.py"


def _import_bench():
    spec = importlib.util.spec_from_file_location("ako_bench", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the trustworthy metric set (replay-invariant / structural) ----------------
# Order matters only for display grouping.
METRICS = [
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # qualitative-only (shown under a banner, never used for arithmetic):
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
]

QUALITATIVE = {
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
}


###############################################################################
# run — the profiling target
###############################################################################


def cmd_run(args):
    import torch

    bench = _import_bench()

    torch.cuda.set_device(0)
    device = torch.cuda.current_device()
    bench.set_seed(42)

    ref_src = bench.read_file(args.ref)
    sol_src = bench.read_file(args.solution)
    backend = args.backend or bench._auto_detect_backend(sol_src)
    uses_tempfile = backend.lower() in ("triton", "tilelang", "cute")

    # reference: we only need Model's init inputs + get_inputs (not to run it)
    ctx = {}
    Model, ref_gii, ref_gi = bench.load_original_model_and_inputs(
        ref_src, ctx, source_path=args.ref
    )
    if ref_gi is None:
        print("ERROR: reference defines no get_inputs()", file=sys.stderr)
        sys.exit(2)

    bench.set_seed(42)
    init_inputs = [] if ref_gii is None else ref_gii()
    init_inputs = [bench._process_input_tensor(x, device) for x in init_inputs]

    # solution
    modified = bench.prepare_solution_source(sol_src)
    temp_file = None
    if uses_tempfile:
        ModelNew, temp_file = bench.load_custom_model_with_tempfile(modified)
    else:
        ModelNew = bench.load_custom_model(
            modified, ctx, args.build_dir, source_path=args.solution
        )
    if ModelNew is None:
        print("ERROR: could not load ModelNew from solution", file=sys.stderr)
        sys.exit(2)

    with torch.no_grad():
        bench.set_seed(42)
        model = ModelNew(*init_inputs).to(device=device, dtype=torch.float32)

        bench.set_seed(42)
        inputs = [bench._process_input_tensor(x, device) for x in ref_gi()]

        # meta: byte sizes so parse can compute "passes" = DRAM bytes / tensor.
        # Primary tensor = the largest floating input (x for these ops).
        f_tensors = [
            t for t in inputs
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        primary = max((t.numel() * t.element_size() for t in f_tensors), default=0)
        total_in = sum(
            t.numel() * t.element_size()
            for t in inputs if isinstance(t, torch.Tensor)
        )

        for _ in range(args.warmup):
            out = model(*inputs)
            torch.cuda.synchronize(device)

        out_bytes = 0
        outs = out if isinstance(out, (tuple, list)) else (out,)
        for o in outs:
            if hasattr(o, "numel"):
                out_bytes += o.numel() * o.element_size()

        if args.meta_out:
            with open(args.meta_out, "w") as f:
                json.dump(
                    {
                        "op_backend": backend,
                        "primary_tensor_bytes": int(primary),
                        "total_input_bytes": int(total_in),
                        "output_bytes": int(out_bytes),
                    },
                    f,
                )

        # cold L2 -> profile exactly one forward()
        bench.clear_l2_cache(device=device)
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        _ = model(*inputs)
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()

    if temp_file:
        temp_file.close()
        try:
            os.remove(temp_file.name)
        except OSError:
            pass


###############################################################################
# parse — turn ncu CSV into a direction-picking report
###############################################################################


def _read_ncu_rows(path):
    """Tolerant reader for `ncu --csv` output. Returns (header, rows).

    Skips ncu banner / ==WARNING==/==PROF== lines; the header is the first CSV
    line that contains a "Kernel Name" column.
    """
    with open(path, newline="") as f:
        raw = f.readlines()
    start = None
    for i, ln in enumerate(raw):
        if '"Kernel Name"' in ln or ("Kernel Name" in ln and "," in ln):
            start = i
            break
    if start is None:
        raise SystemExit(f"parse: no CSV header with 'Kernel Name' found in {path}")
    reader = csv.reader(raw[start:])
    header = next(reader)
    ncols = len(header)
    rows = [r for r in reader if len(r) == ncols]
    return header, rows


def _to_float(s):
    if s is None:
        return None
    s = s.strip().replace(",", "")  # ncu may thousands-separate
    if s in ("", "N/A", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract(header, rows):
    """Return {kernel_name: {metric: [values...]}} handling long OR wide CSV."""
    h = {name: i for i, name in enumerate(header)}
    kcol = h.get("Kernel Name")
    per = {}  # name -> {metric: [values]}
    if "Metric Name" in h and "Metric Value" in h:  # long format
        mn, mv = h["Metric Name"], h["Metric Value"]
        for r in rows:
            name = r[kcol]
            per.setdefault(name, {}).setdefault(r[mn], []).append(_to_float(r[mv]))
    else:  # wide format: one column per metric (name may carry a unit suffix)
        metric_cols = {}
        for m in METRICS:
            for name, i in h.items():
                if name == m or name.startswith(m):
                    metric_cols[m] = i
                    break
        for r in rows:
            name = r[kcol]
            d = per.setdefault(name, {})
            for m, i in metric_cols.items():
                d.setdefault(m, []).append(_to_float(r[i]))
    return per


def _get(d, key):
    """First metric in d whose name equals-or-startswith key; list of floats."""
    for k, v in d.items():
        if k == key or k.startswith(key):
            return [x for x in v if x is not None]
    return []


def cmd_parse(args):
    header, rows = _read_ncu_rows(args.csv)
    per = _extract(header, rows)

    meta = {}
    if args.meta and os.path.exists(args.meta):
        with open(args.meta) as f:
            meta = json.load(f)
    tensor = meta.get("primary_tensor_bytes", 0)
    GiB = 1024 ** 3

    total_r = total_w = 0.0
    n_launches = 0
    lines = []
    for name, d in per.items():
        rd = sum(_get(d, "dram__bytes_read.sum"))
        wr = sum(_get(d, "dram__bytes_write.sum"))
        launches = max(
            len(_get(d, "dram__bytes_read.sum")),
            len(_get(d, "sm__warps_active.avg.pct_of_peak_sustained_active")),
            1,
        )
        total_r += rd
        total_w += wr
        n_launches += launches

        def avg(key):
            vals = _get(d, key)
            return sum(vals) / len(vals) if vals else None

        occ = avg("sm__warps_active.avg.pct_of_peak_sustained_active")
        l2 = avg("lts__t_sector_hit_rate.pct")
        secreq = avg("l1tex__average_t_sectors_per_request")
        drpk = avg("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        smpk = avg("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        short = (name[:38] + "…") if len(name) > 39 else name
        lines.append(
            (rd + wr, short, launches, rd / GiB, wr / GiB, occ, l2, secreq, drpk, smpk)
        )

    lines.sort(reverse=True)  # biggest-traffic kernel first (the bottleneck)
    total = total_r + total_w
    passes = (total / tensor) if tensor else float("nan")
    read_passes = (total_r / tensor) if tensor else float("nan")

    print("=" * 78)
    print("ncu BOTTLENECK REPORT — steer by these (bytes/passes/structural)")
    print("=" * 78)
    if tensor:
        print(f"primary tensor      : {tensor/GiB:.3f} GiB")
    print(f"DRAM read  total    : {total_r/GiB:.3f} GiB  ({read_passes:.2f}x tensor)")
    print(f"DRAM write total    : {total_w/GiB:.3f} GiB")
    print(f"DRAM total          : {total/GiB:.3f} GiB  = {passes:.2f} passes  "
          f"<-- THE ANCHOR (2=copy floor, 3=un-reused re-read)")
    print(f"kernel launches     : {n_launches}")
    print()
    print("per-kernel (sorted by traffic; occ/L2/sec-per-req are structural):")
    hdr = f"  {'kernel':40s} {'launch':>6s} {'rdGiB':>7s} {'wrGiB':>7s} {'occ%':>6s} {'L2hit%':>7s} {'sec/req':>7s}"
    print(hdr)
    for _, name, launches, rd, wr, occ, l2, secreq, drpk, smpk in lines:
        print(f"  {name:40s} {launches:6d} {rd:7.3f} {wr:7.3f} "
              f"{_fmt(occ):>6s} {_fmt(l2):>7s} {_fmt(secreq):>7s}")
    print()
    print("-" * 78)
    print("⚠ QUALITATIVE ONLY — ncu locks clocks & serializes replays, so these")
    print("  %peak numbers are DEPRESSED for many-launch/sub-µs kernels. Read as")
    print("  'saturated vs not', NEVER as a number to do arithmetic with, and NEVER")
    print("  as the steering metric. (NCU_VALIDATION.md)")
    for _, name, launches, rd, wr, occ, l2, secreq, drpk, smpk in lines:
        print(f"  {name:40s}  dram%={_fmt(drpk):>6s}  sm%={_fmt(smpk):>6s}")
    print("-" * 78)


def _fmt(x):
    return "—" if x is None else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="profiling target — launch under ncu")
    r.add_argument("--ref", required=True)
    r.add_argument("--solution", required=True)
    r.add_argument("--backend", default=None)
    r.add_argument("--build-dir", default=None)
    r.add_argument("--meta-out", default=None)
    r.add_argument("--warmup", type=int, default=5)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("parse", help="turn ncu CSV into a bottleneck report")
    p.add_argument("--csv", required=True)
    p.add_argument("--meta", default=None)
    p.set_defaults(func=cmd_parse)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
