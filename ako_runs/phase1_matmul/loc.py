#!/usr/bin/env python3
"""Device-code size per DSL and per abstraction arm.

This is the *only* quantitative handle this study has on "abstraction-enabled
exploration" -- the question of whether a higher-level interface makes it cheaper
to *find* a good kernel, as opposed to whether it runs fast once found. Runtime
tables cannot answer it; they measure the destination, not the cost of the search.

Line count is a crude proxy and is reported as one. What makes it usable here is
that the four modules produce **bit-identical output at every variant**, so this
is genuinely the same computation expressed four ways, not four different
kernels of differing ambition. The count therefore measures expression cost with
the artifact held exactly constant, which is unusual.

Counted: non-comment, non-blank lines inside the device-code body only --
`@triton.jit` / `T.prim_func` functions for the Python DSLs, and the text of the
`__global__` blocks for the CUDA lanes. Host glue, `build()`, artifact plumbing
and self-checks are excluded because they are study scaffolding, not kernel.

usage: python loc.py [--md]
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = os.path.join(HERE, "variants")

PY_DECOS = ("triton.jit", "prim_func", "tilelang.jit", "autotune", "heuristics")


def _n(lines, comment):
    return sum(1 for l in lines if l.strip() and not l.strip().startswith(comment))


def py_device_lines(path, only_names=None):
    """{func_name: lines} for decorated device functions and their inner defs."""
    src = open(path).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decos = [ast.unparse(d) for d in node.decorator_list]
        if not any(k in d for d in decos for k in PY_DECOS):
            continue
        if only_names and node.name not in only_names:
            continue
        out.setdefault(node.name, []).append(
            _n(lines[node.lineno - 1:node.end_lineno], "#"))
    return out


def cuda_device_lines(path):
    src = open(path).read()
    blocks = [b for b in re.findall(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', src, re.S)
              if "__global__" in b]
    txt = "\n".join(blocks)
    # count kernels, not source blocks -- a single triple-quoted string can hold
    # several __global__ functions, and it is the kernels that had to be written
    return _n(txt.splitlines(), "//"), txt.count("__global__")


def cross_dsl():
    rows = []
    for dsl, fname, kind, note in (
        ("triton", "triton_gemm.py", "py", "one `@triton.jit` kernel; variants via `tl.constexpr`"),
        ("tilelang", "tilelang_gemm.py", "py", "`T.gemm` + `T.Pipelined` + `T.copy`"),
        ("cuda_noptx", "cuda_noptx_gemm.py", "cuda", "WMMA C++ `<mma.h>` + `__pipeline_memcpy_async`"),
        ("cuda_unlimited", "cuda_unlimited_gemm.py", "cuda", "inline `mma.sync` / `ldmatrix` / `cp.async` PTX"),
    ):
        p = os.path.join(VARIANTS, fname)
        if not os.path.exists(p):
            continue
        if kind == "py":
            d = py_device_lines(p, only_names={"main", "_gemm_kernel"})
            n = sum(sum(v) for v in d.values())
            k = sum(len(v) for v in d.values())
        else:
            n, k = cuda_device_lines(p)
        rows.append((dsl, n, k, note))
    return rows


def abstraction_arms():
    p = os.path.join(VARIANTS, "tilelang_abstraction.py")
    if not os.path.exists(p):
        return []
    src = open(p).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    # each arm lives in its own _kernel_<X> builder; count the prim_func inside it
    out = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("_kernel_"):
            continue
        arm = node.name[len("_kernel_"):]
        inner = 0
        for sub in ast.walk(node):
            if isinstance(sub, ast.FunctionDef) and sub.name == "main":
                inner = _n(lines[sub.lineno - 1:sub.end_lineno], "#")
        out.append((arm, inner))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()

    x = cross_dsl()
    arms = abstraction_arms()
    base = min((n for _, n, _, _ in x), default=1) or 1

    L = ["**Expression cost for the identical kernel.** Non-comment device-code "
         "lines only (host glue and self-checks excluded). The four modules "
         "produce bit-identical output at every variant, so this compares the "
         "cost of *expressing* one fixed computation, not the cost of four "
         "different ones.\n",
         "| DSL | device-code lines | kernels | ÷ smallest | how the inner loop is written |",
         "|---|---|---|---|---|"]
    for dsl, n, k, note in x:
        L.append(f"| {dsl} | {n} | {k} | {n / base:.1f}× | {note} |")
    L.append("\nThe kernel count is part of the cost. Triton expresses variant A "
             "by changing one argument (`input_precision=\"ieee\"`) inside the same "
             "kernel; TileLang and both CUDA lanes need a **second, separately "
             "written kernel** for the fp32 arm, because at those levels there is "
             "no shared expression of \"this matmul, at that precision\".")

    if arms:
        order = {"H": 0, "M1": 1, "M2": 2, "S1": 3}
        arms.sort(key=lambda t: order.get(t[0], 9))
        L.append("\n**Within TileLang**, the same measurement across abstraction "
                 "levels (`prim_func` body lines):\n")
        L.append("| arm | body lines | note |")
        L.append("|---|---|---|")
        notes = {
            "H": "TL-H — H1 and H2 share this body and differ by **one integer** (`num_stages`)",
            "M1": "TL-M — explicit `T.copy` + barriers around `T.gemm`, no pipeline construct",
            "M2": "TL-M — hand-written double buffer: `T.async_copy` + `ptx_commit_group` + `ptx_wait_group`",
            "S1": "TL-SIMT — scalar FMA control; **a hardware control, not an abstraction level**",
        }
        for arm, n in arms:
            L.append(f"| {arm} | {n} | {notes.get(arm, '')} |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
