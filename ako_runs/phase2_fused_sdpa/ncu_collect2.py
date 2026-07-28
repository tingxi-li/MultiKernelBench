#!/usr/bin/env python3
"""Nsight Compute collection for Phase 2, PER KERNEL.

The SDPA half of the study asks for score-tensor DRAM traffic, register
pressure, occupancy and runtime *for each kernel separately*, which is the only
way the two algorithms can be compared honestly: `K3` spends its time in three
kernels and `FLASH` in one, so a single fused number would hide exactly the
thing being measured. Every launch is profiled and kept, tagged with its kernel
name, rather than picking one by duration.

The score-tensor traffic falls straight out: in `K3` it is the DRAM bytes of the
softmax kernel (which reads S and writes P) plus the write side of the QK kernel;
in `FLASH` it is structurally zero, and the table showing that zero is the point.

ncu numbers are for COUNTERS ONLY. Phase 1 established that ncu's reported
durations sit at the cold-clock transient and disagree with the campaign by tens
of percent; runtimes in this study always come from `driver2.py`, never from here.

usage:
  python ncu_collect2.py --op sdpa --jobs jobs/sdpa_cross.json --gpu 0
  python ncu_collect2.py --op fused --dsl tilelang --variant GBGS
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common2  # noqa: E402

NCU = "/usr/local/cuda-13.1/bin/ncu"
if not os.path.exists(NCU):
    NCU = shutil.which("ncu") or NCU

METRICS = [
    "gpu__time_duration.sum",
    "sm__inst_executed_pipe_tensor_op_hmma_v2.sum",
    "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
]

SHORT = {
    "gpu__time_duration.sum": "dur_ns",
    "sm__inst_executed_pipe_tensor_op_hmma_v2.sum": "hmma_inst",
    "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active": "tensor_pct",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma_inst",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "dram__bytes_read.sum": "dram_read_B",
    "dram__bytes_write.sum": "dram_write_B",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_longsb",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio",
    "launch__registers_per_thread": "regs",
    "launch__shared_mem_per_block_static": "smem_static_B",
    "launch__shared_mem_per_block_dynamic": "smem_dyn_B",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
    "launch__occupancy_limit_registers": "occ_lim_regs",
    "launch__occupancy_limit_shared_mem": "occ_lim_smem",
}

# Kernels that are setup, not the algorithm: allocator zero-fills, dtype casts,
# torch's own elementwise helpers. Tagged rather than dropped, so a reader can
# see that they were seen and classified.
SETUP_HINTS = ("memset", "vectorized_elementwise", "fill", "CatArrayBatched",
               "unrolled_elementwise", "direct_copy", "transpose")


def parse_csv(text):
    lines = [l for l in text.splitlines() if l.strip()]
    start = next((i for i, l in enumerate(lines)
                  if l.startswith('"ID"') or l.startswith("ID,")), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def to_float(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not s or s in ("N/A", "n/a", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def kernel_id(name, m):
    """Identity for folding repeated launches.

    Folding on the name alone is wrong for TileLang, which names *every*
    generated kernel `main_kernel`: the GEMM and the softmax -- and all three of
    K3's SDPA kernels -- collide, and keeping the last launch silently discards
    the others. Since this study's whole point is to report each kernel
    separately, the launch geometry is folded in; two distinct kernels in one
    program do not share grid, block and register count.
    """
    return (name, int(m.get("grid") or 0), int(m.get("block") or 0),
            int(m.get("regs") or 0))


def profile(op, dsl, variant, setstr, gpu, iters=2):
    cmd = [NCU, "--csv", "--page", "raw", "--target-processes", "all",
           "--launch-count", "400", "--metrics", ",".join(METRICS),
           sys.executable, os.path.join(HERE, "profile_target2.py"),
           "--op", op, "--dsl", dsl, "--variant", variant, "--iters", str(iters)]
    if setstr:
        cmd += ["--set", setstr]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = HERE + ":" + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=HERE, timeout=3600)


def collect_one(op, job, gpu, iters=2):
    dsl, variant, setstr = job["dsl"], job["variant"], job.get("set", "")
    rec = {"op": op, "dsl": dsl, "variant": variant, "set": setstr}
    try:
        p = profile(op, dsl, variant, setstr, gpu, iters=iters)
    except subprocess.TimeoutExpired:
        rec["ok"] = False
        rec["error"] = "ncu timeout"
        return rec
    rows = parse_csv(p.stdout)
    if not rows:
        rec["ok"] = False
        rec["error"] = (p.stderr or p.stdout)[-800:]
        return rec

    # ncu --page raw gives one row per (kernel, launch). Fold repeated launches
    # of the same kernel by taking the LAST one -- the first launch of each is
    # the cold one and is not representative of the steady state the campaign
    # measures.
    by_kernel = {}
    for r in rows:
        name = r.get("Kernel Name") or r.get('"Kernel Name"') or "?"
        m = {}
        for k in METRICS:
            v = r.get(k)
            if v is None:
                for kk in r:
                    if kk.strip('"') == k:
                        v = r[kk]
                        break
            f = to_float(v)
            if f is not None:
                m[SHORT.get(k, k)] = f
        if not m:
            continue
        short = name.split("(")[0].strip()[:70]
        kid = kernel_id(short, m)
        if kid not in by_kernel:
            by_kernel[kid] = {"name": short, "order": len(by_kernel), "ms": []}
        by_kernel[kid]["ms"].append(m)

    # `profile_target2.py` makes one untimed warm-up call and then `iters` more,
    # so a kernel in the steady-state loop launches `iters + 1` times. A kernel
    # that launches fewer times than that is one-shot setup -- in the fused op
    # it is the host-side weight conversion, which the cached arm pays once and
    # which is not part of what the campaign times.
    n_calls = iters + 1
    kernels = []
    for kid, e in by_kernel.items():
        name, ms = e["name"], e["ms"]
        last = ms[-1]
        last["kernel"] = name
        last["n_launches"] = len(ms)
        last["launch_order"] = e["order"]
        last["grid_block_regs"] = list(kid[1:])
        last["is_setup"] = (any(h.lower() in name.lower() for h in SETUP_HINTS)
                            or len(ms) < n_calls)
        rd, wr = last.get("dram_read_B", 0.0), last.get("dram_write_B", 0.0)
        last["dram_total_GB"] = (rd + wr) / 1e9
        kernels.append(last)
    # Launch order, not duration: for K3 the reader needs QK -> softmax -> PV in
    # the order they run, which is the sequence the algorithm is named after.
    kernels.sort(key=lambda k: k["launch_order"])
    rec["kernels"] = kernels
    rec["n_algo_kernels"] = sum(1 for k in kernels if not k["is_setup"])
    rec["algo_dram_GB"] = sum(k["dram_total_GB"] for k in kernels
                              if not k["is_setup"])
    rec["setup_dram_GB"] = sum(k["dram_total_GB"] for k in kernels
                               if k["is_setup"])
    rec["ok"] = True
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=("fused", "sdpa"))
    ap.add_argument("--jobs", default="")
    ap.add_argument("--dsl", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.jobs:
        with open(a.jobs) as f:
            jobs = json.load(f)
    else:
        jobs = [{"dsl": a.dsl, "variant": a.variant, "set": a.setstr}]

    out = a.out or os.path.join(common2.RESULTS_DIR, f"ncu_{a.op}.json")
    recs = []
    for i, job in enumerate(jobs, 1):
        r = collect_one(a.op, job, a.gpu, iters=a.iters)
        recs.append(r)
        if r.get("ok"):
            algo = [k for k in r["kernels"] if not k["is_setup"]]
            print(f"[{i}/{len(jobs)}] {job['dsl']}/{job['variant']} "
                  f"{job.get('set','')}: {len(algo)} algo kernel(s), "
                  f"{r['algo_dram_GB']:.2f} GB DRAM "
                  f"(+{r['setup_dram_GB']:.2f} GB one-shot setup)"
                  + "".join(f"\n      {k['kernel'][:52]:<52s} "
                            f"{k.get('dur_ns',0)/1e6:7.3f} ms  "
                            f"regs={k.get('regs','?'):>4}  "
                            f"occ={k.get('occupancy_pct',0):5.1f}%  "
                            f"dram={k['dram_total_GB']:6.2f} GB"
                            for k in algo), flush=True)
        else:
            print(f"[{i}/{len(jobs)}] {job['dsl']}/{job['variant']}: FAILED "
                  f"{str(r.get('error'))[:100]}", flush=True)
        common2.write_json(out, {"metrics_requested": METRICS, "records": recs})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
