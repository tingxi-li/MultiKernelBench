#!/usr/bin/env python3
"""Generated-code inspection: PTX + SASS dump and instruction census.

"TileLang auto-pipelines better" is an inference until you look at what the
compiler emitted. This dumps the generated code for a variant and counts the
instructions that decide the question:

  SASS  HMMA/IMMA/BMMA  tensor-core MMA issued
        LDGSTS          cp.async global->shared (the async pipeline)
        LDSM            ldmatrix shared->fragment
        FFMA            fp32 CUDA-core FMA (should dominate variant A, be ~0 in B/C/D)
        BAR.SYNC        barriers
  PTX   mma.sync / wmma.mma / cp.async / ldmatrix

plus registers, spills (store+load), and static shared memory from ptxas -v.

usage:
  python inspect_code.py --dsl triton --variant D
  python inspect_code.py --all                     # every dsl x variant
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
common.setup_cuda_env()

import torch  # noqa: E402
import variants  # noqa: E402
from runner import parse_set  # noqa: E402

CUDA_BIN = "/usr/local/cuda-13.1/bin"
ARCH = "sm_89"

SASS_COUNT = ["HMMA", "IMMA", "BMMA", "LDGSTS", "LDSM", "FFMA", "FADD", "FMUL",
              "LDG", "STG", "LDS", "STS", "BAR.SYNC", "F2F", "I2F", "HADD2", "HFMA2"]
PTX_COUNT = ["mma.sync", "wmma.mma", "wmma.load", "wmma.store", "cp.async",
             "ldmatrix", "ld.global", "st.shared", "bar.sync", "fma.rn.f32",
             "cvt.rn.f16.f32"]


def _tool(name):
    p = os.path.join(CUDA_BIN, name)
    return p if os.path.exists(p) else (shutil.which(name) or p)


def count_tokens(text: str, tokens: list[str]) -> dict:
    out = {}
    for t in tokens:
        # SASS mnemonics appear as whole opcodes; PTX ones as instruction prefixes
        pat = re.compile(r"(?<![A-Za-z0-9_.])" + re.escape(t))
        out[t] = len(pat.findall(text))
    return out


def ptxas_stats(log: str) -> dict:
    """Parse `ptxas -v` / nvcc -Xptxas=-v output."""
    out = {}
    m = re.search(r"Used (\d+) registers", log)
    if m:
        out["n_regs"] = int(m.group(1))
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if m:
        out["stack_frame_bytes"] = int(m.group(1))
        out["spill_store_bytes"] = int(m.group(2))
        out["spill_load_bytes"] = int(m.group(3))
    m = re.search(r"(\d+) bytes smem", log)
    if m:
        out["static_smem_bytes"] = int(m.group(1))
    m = re.search(r"Compiling entry function '([^']+)'", log)
    if m:
        out["entry"] = m.group(1)
    return out


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=kw.pop("timeout", 600), **kw)


def sass_from_cubin(cubin_path: str) -> str:
    p = run([_tool("cuobjdump"), "-sass", cubin_path])
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout
    p = run([_tool("nvdisasm"), "-c", cubin_path])
    return p.stdout if p.returncode == 0 else f"<disasm failed: {p.stderr[:400]}>"


def _include_flags() -> list[str]:
    """Includes the generated sources need to compile standalone.

    tilelang emits `#include "tl_templates/cuda/gemm.h"` and the CUDA lanes emit
    `#include <torch/extension.h>`; neither is on the default search path, so a
    naive `nvcc -cubin` on the generated source fails and the whole SASS census
    silently comes back empty for those lanes.
    """
    flags = []
    try:
        import tilelang
        tlroot = os.path.dirname(tilelang.__file__)
        for sub in ("src",
                    # tl_templates pulls in cutlass headers transitively
                    os.path.join("3rdparty", "cutlass", "include"),
                    os.path.join("3rdparty", "cutlass", "tools", "util", "include")):
            d = os.path.join(tlroot, sub)
            if os.path.isdir(d):
                flags += ["-I", d]
    except Exception:
        pass
    try:
        from torch.utils import cpp_extension as _ce
        for p in _ce.include_paths(device_type="cuda"):
            flags += ["-I", p]
    except Exception:
        try:
            from torch.utils import cpp_extension as _ce
            for p in _ce.include_paths():
                flags += ["-I", p]
        except Exception:
            pass
    import sysconfig
    inc = sysconfig.get_paths().get("include")
    if inc:
        flags += ["-I", inc]
    return flags


def sass_from_shared_object(path: str) -> str:
    """SASS straight out of a built .so -- no recompilation, so what we count is
    exactly the code that ran."""
    p = run([_tool("cuobjdump"), "-sass", path])
    return p.stdout if p.returncode == 0 and p.stdout.strip() else ""


def from_cuda_source(src: str, outdir: str, tag: str) -> dict:
    """nvcc -cubin the generated CUDA, harvest ptxas -v stats and SASS."""
    cu = os.path.join(outdir, tag + ".cu")
    cubin = os.path.join(outdir, tag + ".cubin")
    ptx = os.path.join(outdir, tag + ".ptx")
    with open(cu, "w") as f:
        f.write(src)
    res = {"cu_path": cu}
    base = [_tool("nvcc"), f"-arch={ARCH}", "-std=c++17", "--expt-relaxed-constexpr",
            "-DTORCH_EXTENSION_NAME=x", "-D_GLIBCXX_USE_CXX11_ABI=1"] + _include_flags()
    p = run(base + ["-cubin", "-Xptxas=-v", "-o", cubin, cu])
    res["nvcc_rc"] = p.returncode
    res["ptxas_log"] = (p.stderr or "")[-4000:]
    if p.returncode == 0:
        res.update(ptxas_stats(p.stderr))
        res["cubin_path"] = cubin
        res["sass"] = sass_from_cubin(cubin)
    p2 = run(base + ["-ptx", "-o", ptx, cu])
    if p2.returncode == 0:
        res["ptx_path"] = ptx
        res["ptx"] = open(ptx).read()
    return res


def inspect(cfg, outdir) -> dict:
    built = variants.build(cfg)
    art = built.artifacts or {}
    tag = cfg.key().replace("/", "_")
    rec = {"key": cfg.key(), "dsl": cfg.dsl, "variant": cfg.variant,
           "cfg": cfg.to_dict(), "compile_s": built.compile_s,
           "artifact_keys": sorted(art.keys())}

    ptx_text = art.get("ptx", "")
    sass_text = ""

    # 1) a cubin handed to us directly (triton)
    cub = art.get("cubin_path")
    if not cub and art.get("cubin_bytes"):
        cub = os.path.join(outdir, tag + ".cubin")
        with open(cub, "wb") as f:
            f.write(art["cubin_bytes"])
    if cub and os.path.exists(cub):
        sass_text = sass_from_cubin(cub)
        rec["cubin_path"] = cub

    # 2) the actual built extension .so -- preferred for the CUDA lanes, because
    #    it is the exact binary that ran rather than a recompilation of it
    if not sass_text:
        ext = art.get("ext_name")
        so = None
        if ext:
            cand = os.path.join(common.HERE, ".torch_ext", ext, ext + ".so")
            so = cand if os.path.exists(cand) else None
        if so is None:
            import glob as _g
            hits = _g.glob(os.path.join(common.HERE, ".torch_ext", "*", "*.so"))
            # match on the config fingerprint the lanes put in their ext names
            want = f"{cfg.BM}_{cfg.BN}_{cfg.BK}"
            hits = [h for h in hits if want in os.path.basename(h)]
            so = hits[0] if len(hits) == 1 else None
        if so:
            s = sass_from_shared_object(so)
            if s:
                sass_text = s
                rec["sass_from_so"] = so

    # 3) generated CUDA we can compile ourselves (tilelang, cuda_*)
    if not sass_text and art.get("cuda_source"):
        r = from_cuda_source(art["cuda_source"], outdir, tag)
        rec["nvcc"] = {k: v for k, v in r.items() if k not in ("sass", "ptx")}
        sass_text = r.get("sass", "")
        ptx_text = ptx_text or r.get("ptx", "")
        for k in ("n_regs", "spill_store_bytes", "spill_load_bytes",
                  "static_smem_bytes", "stack_frame_bytes"):
            if k in r:
                rec[k] = r[k]

    # 3) a PTX string we can assemble
    if not sass_text and ptx_text:
        pp = os.path.join(outdir, tag + ".ptx")
        with open(pp, "w") as f:
            f.write(ptx_text)
        cb = os.path.join(outdir, tag + ".cubin")
        p = run([_tool("ptxas"), f"-arch={ARCH}", "-v", "-o", cb, pp])
        rec["ptxas_rc"] = p.returncode
        rec.update(ptxas_stats(p.stderr or ""))
        if p.returncode == 0:
            sass_text = sass_from_cubin(cb)
            rec["cubin_path"] = cb

    # what the DSL told us directly wins over what we re-derived
    for k in ("n_regs", "n_spills", "shared_bytes", "grid", "block", "backend_detail"):
        if k in art:
            rec["reported_" + k] = art[k]

    if sass_text:
        p = os.path.join(outdir, tag + ".sass")
        with open(p, "w") as f:
            f.write(sass_text)
        rec["sass_path"] = p
        rec["sass_counts"] = count_tokens(sass_text, SASS_COUNT)
        rec["sass_lines"] = sass_text.count("\n")
    if ptx_text:
        p = os.path.join(outdir, tag + ".ptx")
        if not os.path.exists(p):
            with open(p, "w") as f:
                f.write(ptx_text)
        rec["ptx_path"] = p
        rec["ptx_counts"] = count_tokens(ptx_text, PTX_COUNT)

    c = rec.get("sass_counts", {})
    rec["verdict"] = {
        "uses_tensor_cores": (c.get("HMMA", 0) + c.get("IMMA", 0) + c.get("BMMA", 0)) > 0,
        "uses_cp_async": c.get("LDGSTS", 0) > 0,
        "uses_ldmatrix": c.get("LDSM", 0) > 0,
        "fp32_fma_heavy": c.get("FFMA", 0) > 100,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsl", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--geom", default="primary")
    ap.add_argument("--set", dest="setstr", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(common.ARTIFACTS_DIR, "code"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    jobs = []
    if args.all:
        for d in common.DSLS:
            for v in ("A", "B", "C", "D"):
                jobs.append((d, v, {}))
    else:
        jobs.append((args.dsl, args.variant, parse_set(args.setstr)))

    recs = []
    for dsl, variant, over in jobs:
        cfg = common.make_config(dsl, variant, args.geom, **over)
        try:
            rec = inspect(cfg, args.outdir)
        except Exception as e:  # noqa: BLE001
            import traceback
            rec = {"key": cfg.key(), "dsl": dsl, "variant": variant,
                   "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()}
        recs.append(rec)
        c = rec.get("sass_counts", {})
        print(f"{rec['key']:<64s} HMMA={c.get('HMMA','?'):>6} LDGSTS={c.get('LDGSTS','?'):>5} "
              f"LDSM={c.get('LDSM','?'):>5} FFMA={c.get('FFMA','?'):>7} "
              f"regs={rec.get('n_regs','?')} spill={rec.get('spill_store_bytes','?')} "
              f"smem={rec.get('static_smem_bytes', rec.get('reported_shared_bytes','?'))}"
              + (f"  ERR {rec['error'][:80]}" if "error" in rec else ""))
        torch.cuda.empty_cache()

    out = os.path.join(common.RESULTS_DIR, "code_inspection.json")
    # MERGE by key rather than overwrite. Inspecting one target used to replace
    # the whole census: a later single-variant run silently reduced a 16-record
    # file to 1, and the report's SASS table rendered as an empty header. Losing
    # data by running a narrower version of the same command is not acceptable.
    keep = [{k: v for k, v in r.items() if k not in ("sass", "ptx")} for r in recs]
    merged = {r["key"]: r for r in keep}
    if os.path.exists(out):
        try:
            for r in json.load(open(out)).get("records", []):
                merged.setdefault(r.get("key"), r)
        except Exception:  # noqa: BLE001 -- a corrupt prior file must not block the new one
            pass
    common.write_json(out, {"records": [merged[k] for k in sorted(merged)]})
    print(f"-> {out}  ({len(merged)} records, {len(keep)} from this run)"
          f"\n-> dumps in {args.outdir}")


if __name__ == "__main__":
    main()
