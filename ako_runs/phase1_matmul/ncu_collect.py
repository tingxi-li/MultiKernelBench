#!/usr/bin/env python3
"""Nsight Compute metric collection for the Phase-1 variants.

Answers the hardware half of the pipeline-control question:
  * achieved tensor-core FLOP/s and tensor-pipe utilization
  * occupancy
  * L1/shared vs global stall breakdown
  * DRAM traffic and L2 hit rate
  * registers, spills, shared memory actually allocated at launch

usage:
  python ncu_collect.py --dsl triton --variant D
  python ncu_collect.py --all --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

NCU = "/usr/local/cuda-13.1/bin/ncu"
if not os.path.exists(NCU):
    NCU = shutil.which("ncu") or NCU

METRICS = [
    "gpu__time_duration.sum",
    # tensor cores
    "sm__inst_executed_pipe_tensor_op_hmma_v2.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_active",
    # cuda-core fp32 (variant A should live here)
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    # occupancy + throughput
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # memory
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    # stalls
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
    # launch config
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
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum": "tensor_ops_f16f32",
    "sm__pipe_tensor_op_hmma_cycles_active_v2.avg.pct_of_peak_sustained_active": "tensor_pipe_pct",
    "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_active": "tensor_any_pct",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma_thread_inst",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "l1tex_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "dram__bytes_read.sum": "dram_rd_B",
    "dram__bytes_write.sum": "dram_wr_B",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "stall_short_sb",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_long_sb",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio": "stall_wait",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "stall_math",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio": "stall_noinst",
    "launch__registers_per_thread": "regs",
    "launch__shared_mem_per_block_static": "smem_static_B",
    "launch__shared_mem_per_block_dynamic": "smem_dyn_B",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
    "launch__occupancy_limit_registers": "occ_lim_regs",
    "launch__occupancy_limit_shared_mem": "occ_lim_smem",
}


def profile(dsl, variant, geom, setstr, gpu, outdir):
    here = os.path.dirname(os.path.abspath(__file__))
    # Profile EVERY launch and pick the GEMM afterwards by duration. A fixed
    # --launch-skip window is wrong here: each DSL emits a different number of
    # setup kernels (operand casts, zero-fills, allocator warmups) before the
    # GEMM, so a fixed skip silently profiles a 16-register memset for one lane
    # and the real kernel for another.
    cmd = [NCU, "--csv", "--page", "raw", "--target-processes", "all",
           "--launch-count", "200",
           "--metrics", ",".join(METRICS),
           sys.executable, os.path.join(here, "profile_target.py"),
           "--dsl", dsl, "--variant", variant, "--geom", geom, "--iters", "3"]
    if setstr:
        cmd += ["--set", setstr]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = here + ":" + env.get("PYTHONPATH", "")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=here, timeout=1800)
    return p


def parse_csv(text):
    """ncu --csv --page raw: one row per (kernel, launch); metrics are columns."""
    lines = [l for l in text.splitlines() if l.strip()]
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")), None)
    if start is None:
        return []
    rdr = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return list(rdr)


def to_float(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if s in ("", "n/a", "N/A", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variants", default="A,B,C,D")
    ap.add_argument("--dsls", default=",".join(common.DSLS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(common.RESULTS_DIR, "ncu.json"))
    args = ap.parse_args()

    outdir = os.path.join(common.ARTIFACTS_DIR, "ncu")
    os.makedirs(outdir, exist_ok=True)

    jobs = []
    if args.all:
        for d in [s for s in args.dsls.split(",") if s]:
            for v in [s for s in args.variants.split(",") if s]:
                jobs.append((d, v, args.geom, ""))
    else:
        jobs.append((args.dsl, args.variant, args.geom, args.setstr))

    recs = []
    for dsl, variant, geom, setstr in jobs:
        key = f"{dsl}.{variant}.{geom}" + (f".{setstr}" if setstr else "")
        try:
            p = profile(dsl, variant, geom, setstr, args.gpu, outdir)
        except subprocess.TimeoutExpired:
            recs.append({"key": key, "ok": False, "error": "ncu TIMEOUT"})
            print(f"{key:<48s} TIMEOUT")
            continue
        with open(os.path.join(outdir, key.replace("/", "_") + ".csv"), "w") as f:
            f.write(p.stdout)
        rows = parse_csv(p.stdout)
        if not rows:
            recs.append({"key": key, "ok": False, "error": "no ncu rows",
                         "stderr_tail": p.stderr[-2000:], "stdout_tail": p.stdout[-2000:]})
            print(f"{key:<48s} NO ROWS  {p.stderr.strip().splitlines()[-1][:110] if p.stderr.strip() else ''}")
            continue

        # The GEMM is the longest-running kernel that also actually does the
        # work: it must touch shared memory or issue tensor-core/FFMA math.
        # Duration alone is not enough -- the 256 MB L2-thrash fill_ kernel in
        # the harness is long and would otherwise win.
        def is_gemm(r):
            smem = (to_float(r.get("launch__shared_mem_per_block_static")) or 0) + \
                   (to_float(r.get("launch__shared_mem_per_block_dynamic")) or 0)
            math = (to_float(r.get("sm__inst_executed_pipe_tensor_op_hmma_v2.sum")) or 0) + \
                   (to_float(r.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum")) or 0)
            return smem > 0 and math > 0

        cands = [r for r in rows if is_gemm(r)] or rows
        best, best_dur = None, -1.0
        for r in cands:
            d = to_float(r.get("gpu__time_duration.sum"))
            if d is not None and d > best_dur:
                best, best_dur = r, d
        best = best or cands[0]
        n_gemm_like = len(cands)

        m = {}
        for k in METRICS:
            v = to_float(best.get(k))
            if v is not None:
                m[SHORT.get(k, k)] = v
        dur_s = (m.get("dur_ns", 0) or 0) * 1e-9
        if dur_s > 0:
            m["measured_ms"] = dur_s * 1e3
            m["effective_tflops"] = common.FLOPS / dur_s / 1e12
            if "tensor_ops_f16f32" in m:
                m["tensor_tflops_achieved"] = m["tensor_ops_f16f32"] / dur_s / 1e12
        m["kernel_name"] = best.get("Kernel Name", "")
        recs.append({"key": key, "dsl": dsl, "variant": variant, "geom": geom,
                     "set": setstr, "ok": True, "metrics": m,
                     "n_kernels_in_launch": len(rows),
                     "n_gemm_like": n_gemm_like})
        print(f"{key:<48s} {m.get('measured_ms', 0):7.3f} ms  "
              f"TC%={m.get('tensor_pipe_pct', 0):5.1f}  occ%={m.get('occupancy_pct', 0):5.1f}  "
              f"regs={m.get('regs', 0):.0f} smem={m.get('smem_static_B', 0) + m.get('smem_dyn_B', 0):.0f}B  "
              f"dram={(m.get('dram_rd_B', 0) + m.get('dram_wr_B', 0)) / 2**20:.0f}MiB  "
              f"TC_TF/s={m.get('tensor_tflops_achieved', 0):.1f}")

    common.write_json(args.out, {"metrics_requested": METRICS, "records": recs})
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
