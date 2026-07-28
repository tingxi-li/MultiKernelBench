#!/usr/bin/env python3
"""Figures for the Phase-1 report.

The KC figure is the one that carries an argument the tables state but do not
show: runtime is flat in KC while error is linear in it. Two quantities with
different units and opposite behaviour over the same x-axis is exactly the case
where a figure beats a table, so it gets a twin-axis plot with error on a log
scale and runtime on a linear one.

Everything is drawn from the same aggregate JSON the tables use, so a figure
cannot disagree with the table above it.

usage: python plots.py [--out artifacts/figs]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from analyze import aggregate, load  # noqa: E402

DSL_ORDER = ["tilelang", "triton", "cuda_noptx", "cuda_unlimited"]
COLORS = {"tilelang": "#1f77b4", "triton": "#d62728",
          "cuda_noptx": "#2ca02c", "cuda_unlimited": "#ff7f0e"}
MARKERS = {"tilelang": "o", "triton": "s", "cuda_noptx": "^", "cuda_unlimited": "D"}


def cfgparts(s):
    return dict(p.split("=", 1) for p in (s or "").split(";") if "=" in p)


def kc_runtime(tag="kc_sweep", variant="D"):
    """{dsl: {kc: ms}}"""
    out = {}
    for k, x in aggregate(load(tag)).items():
        if x.get("status") != "OK" or k[1] != variant:
            continue
        p = cfgparts(k[4])
        if not p.get("kc"):
            continue
        out.setdefault(k[0], {})[int(p["kc"])] = x["median_of_medians_ms"]
    return out


def kc_error():
    """{dist: {kc: (max_err, |bias|)}} from the precision campaign."""
    out = {}
    for path in glob.glob(os.path.join(common.RESULTS_DIR, "accuracy", "*.json")):
        try:
            d = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        cfg = d.get("cfg", {})
        if cfg.get("variant") not in ("B", "C"):
            continue
        # variant B is the no-flush arm, i.e. a single chunk spanning all of K
        kc = cfg.get("kc") or cfg.get("K", 8192)
        for dist, roll in (d.get("by_dist") or {}).items():
            mx = roll.get("max_abs_err_worst")
            bias = roll.get("signed_mean_err_mean")
            if mx is None:
                continue
            out.setdefault(dist, {})[int(kc)] = (mx, abs(bias) if bias is not None else None)
    return out


def fig_kc(outdir):
    rt = kc_runtime()
    err = kc_error()
    if not rt:
        print("  (no kc_sweep runtime data -- skipping fig_kc)")
        return None
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for d in DSL_ORDER:
        pts = sorted(rt.get(d, {}).items())
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[d],
                color=COLORS[d], label=f"{d} (runtime)", lw=1.8, ms=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192\n(= full K,\nvariant B)"],
                       fontsize=8.5)
    ax.set_xlabel("KC — accumulator chunk length (elements of K)")
    ax.set_ylabel("median runtime (ms)  — variant D")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, ls=":")

    ax2 = ax.twinx()
    rand = sorted((err.get("rand") or {}).items())
    if rand:
        ax2.plot([p[0] for p in rand], [p[1][0] for p in rand], color="k",
                 ls="--", marker="x", lw=2.0, label="max abs error (rand)")
        bias = [(k, v[1]) for k, v in rand if v[1] is not None]
        if bias:
            ax2.plot([b[0] for b in bias], [b[1] for b in bias], color="0.45",
                     ls=":", marker="+", lw=1.6, label="|signed bias| (rand)")
    ax2.axhline(0.205, color="crimson", lw=1.4, alpha=0.8)
    ax2.text(540, 0.22, "gate budget 0.205 — only KC=8192 crosses it",
             color="crimson", fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("error vs fp32 oracle (log scale)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right", ncol=2,
              framealpha=0.95)
    ax.set_title("Split-K is an accuracy lever, not a performance one:\n"
                 "error is linear in KC, runtime is flat", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "kc_sweep.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  -> {p}")
    return p


def fig_matched(outdir):
    """Two panels: the full range (which the A bar dominates) and a zoom on
    B/C/D. One panel cannot do both -- at a linear scale that shows A honestly,
    the 1.05-vs-1.38 ms spread at D is invisible, and a log scale that reveals it
    understates how large the fp16 step is."""
    agg = aggregate(load("matched"))
    from analyze import canonical
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    W = 0.2
    vals = {d: [] for d in DSL_ORDER}
    for d in DSL_ORDER:
        for v in "ABCD":
            r = canonical(agg, d, v, "primary", "rand")
            vals[d].append(r["median_of_medians_ms"] if r else float("nan"))

    for i, d in enumerate(DSL_ORDER):
        ax.bar([x + (i - 1.5) * W for x in range(4)], vals[d], W,
               label=d, color=COLORS[d])
        axz.bar([x + (i - 1.5) * W for x in range(3)], vals[d][1:], W,
                label=d, color=COLORS[d])

    tor = canonical(agg, "torch", "A", "primary", "rand")
    if tor:
        t = tor["median_of_medians_ms"]
        ax.axhline(t, color="k", ls="--", lw=1.4)
        ax.text(0.9, t * 1.03, f"torch.matmul fp32 = {t:.2f} ms", fontsize=8.5)
    torb = canonical(agg, "torch", "B", "primary", "rand")
    if torb:
        tb = torb["median_of_medians_ms"]
        axz.axhline(tb, color="k", ls="--", lw=1.4)
        axz.text(-0.46, 1.79,
                 f"dashed = torch.matmul fp16, {tb:.2f} ms — fails the gate "
                 f"(max err 1.82, 72.8% of elements)", fontsize=7.8)

    ax.set_xticks(list(range(4)))
    ax.set_xticklabels(["A\nfp32", "B\n+fp16 TC\n(FAILS gate)", "C\n+split-K",
                        "D\n+3-stage pipe"])
    ax.set_ylabel("median runtime (ms)")
    ax.set_title("Full range — the fp16 step dominates", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8.5)

    axz.set_xticks(list(range(3)))
    axz.set_xticklabels(["B\n+fp16 TC\n(FAILS gate)", "C\n+split-K",
                         "D\n+3-stage pipe"])
    axz.set_ylim(0, 1.85)
    axz.set_title("Zoom on B/C/D — where the DSLs actually differ", fontsize=10.5)
    axz.grid(axis="y", alpha=0.25, ls=":")
    for i, d in enumerate(DSL_ORDER):
        for j, v in enumerate(vals[d][1:]):
            if v == v:
                axz.text(j + (i - 1.5) * W, v + 0.025, f"{v:.2f}",
                         ha="center", fontsize=6.6)

    fig.suptitle("Matched configuration BM=128 BN=128 BK=32 — one factor per step",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = os.path.join(outdir, "matched.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  -> {p}")
    return p


def fig_pipeline(outdir):
    agg = aggregate(load("pipeline"))
    if not agg:
        print("  (no pipeline data -- skipping fig_pipeline)")
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for d in DSL_ORDER:
        pts = []
        for k, x in agg.items():
            if k[0] != d or x.get("status") != "OK":
                continue
            s = cfgparts(k[4]).get("stages")
            if s:
                pts.append((int(s), x["median_of_medians_ms"]))
        pts.sort()
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[d],
                    color=COLORS[d], label=d, lw=1.8, ms=6)
    ax.set_xlabel("pipeline stages")
    ax.set_ylabel("median runtime (ms)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    ax.set_title("Pipeline depth at the matched tile", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, "pipeline.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  -> {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(common.ARTIFACTS_DIR, "figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (fig_matched, fig_kc, fig_pipeline):
        try:
            fn(a.out)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
