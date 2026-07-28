"""TileLang lane of the Phase-1 matched-GEMM study.

Implements variants A/B/C/D of variants/SPEC.md in TileLang 0.1.11, parameterised
on cfg.BM/BN/BK/threads/kc/stages/arith/cast.  Nothing here is autotuned: every
schedule number comes from the Config the driver hands us.

Structure per variant (all at the same BM/BN/BK/threads):

  A  arith="fp32"  kc=0  stages=1
     Explicit scalar FMA loop.  `T.gemm` is NOT used, because on sm_89 it lowers
     to `mma.sync ... tf32` for fp32 operands and that would silently make the
     "no tensor core" floor a tensor-core kernel.  Instead the inner product is
     written as a `T.Parallel(BM, BN)` register tile with a `T.serial(BK)` FMA
     chain, which lowers to plain FFMA.  Verified from SASS: zero HMMA/IMMA.

  B  arith="fp16" kc=0 stages=1
     `T.gemm(As, Bs, Cacc)` with a single fp32 fragment accumulator carried
     across the whole K extent, one shared-memory buffer (`num_stages=1`).

  C  as B, but the tensor-core fragment `Cchunk` is cleared, accumulated over
     `cfg.kc` elements of K, and then flushed into a second fp32 fragment
     `Cacc`.  Pipelining still off.

  D  as C, plus `T.Pipelined(..., num_stages=cfg.stages)` on the global->shared
     loads (tilelang multi-buffers the smem tiles and software-pipelines the
     cp.async issue against the mma work).

cfg.cast (fp16 arms only):
  precast    kernel signature is fp16; the driver hands us fp16 operands.
  in_region  kernel signature is fp16; `run` calls `.half()` itself, inside the
             timed region.
  on_load    kernel signature is fp32, shared memory is fp16; the fp32->fp16
             conversion happens on the global->shared path (a second kernel
             body -- same schedule, different global dtype).

Grid is always the plain 2-D `(ceil(N/BN), ceil(M/BM))`; no swizzle, no
cross-block split-K, no atomics.
"""
#   NOTE: no `from __future__ import annotations` here on purpose -- TileLang's
#   eager builder resolves the `T.Tensor((M, K), dtype)` annotations with
#   typing.get_type_hints() against *module* globals, so stringised annotations
#   would fail to see the closure variables M/N/K/BM/BN.
import os
import re
import time

import torch

import tilelang
import tilelang.language as T

import common

# ---------------------------------------------------------------------------
# TileLang keeps a disk cache of compiled kernels.  With the cache on, compile_s
# would be "cold" for whichever process happens to run first and "warm" for the
# rest -- and the driver deliberately randomises process order, so the reported
# compile time would be a function of run order rather than of the DSL.  Disable
# it so compile_s is always a true source->launchable measurement.  Set
# PHASE1_TL_CACHE=1 to restore the default behaviour.
_CACHE_ENABLED = os.environ.get("PHASE1_TL_CACHE", "0") == "1"
if not _CACHE_ENABLED:
    tilelang.disable_cache()

# Dump PTX/SASS next to the study's artifacts (slow: invokes cuobjdump).  Off by
# default so the timing campaign is not paying for it on every process.
_DUMP_ASM = os.environ.get("PHASE1_TL_ASM", "0") == "1"


# ------------------------------------------------------------------ kernels ---
def _kernel_fp32(M, N, K, BM, BN, BK, threads, stages, kc):
    """Variant A: scalar fp32 FMA, no tensor cores anywhere."""
    use_chunk = kc > 0
    NC = K // kc if use_chunk else 1
    KI = (kc // BK) if use_chunk else (K // BK)

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), "float32"),
                 B: T.Tensor((K, N), "float32"),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float32")
                Bs = T.alloc_shared((BK, BN), "float32")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                if use_chunk:
                    Cchunk = T.alloc_fragment((BM, BN), "float32")
                    for c in T.serial(NC):
                        T.clear(Cchunk)
                        for ko in T.Pipelined(KI, num_stages=stages):
                            T.copy(A[by * BM, c * kc + ko * BK], As)
                            T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                            for i, j in T.Parallel(BM, BN):
                                for kk in T.serial(BK):
                                    Cchunk[i, j] += As[i, kk] * Bs[kk, j]
                        for i, j in T.Parallel(BM, BN):
                            Cacc[i, j] += Cchunk[i, j]
                else:
                    for ko in T.Pipelined(KI, num_stages=stages):
                        T.copy(A[by * BM, ko * BK], As)
                        T.copy(B[ko * BK, bx * BN], Bs)
                        for i, j in T.Parallel(BM, BN):
                            for kk in T.serial(BK):
                                Cacc[i, j] += As[i, kk] * Bs[kk, j]
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


def _kernel_fp16(M, N, K, BM, BN, BK, threads, stages, kc, gdtype):
    """Variants B/C/D: fp16 operands into T.gemm, fp32 fragment accumulator.

    `gdtype` is the dtype of the *global* operands: "float16" for
    cast=precast/in_region, "float32" for cast=on_load (shared memory stays
    fp16, so the conversion rides the global->shared copy).
    """
    use_chunk = kc > 0
    NC = K // kc if use_chunk else 1
    KI = (kc // BK) if use_chunk else (K // BK)

    @tilelang.jit(out_idx=[-1])
    def _k():
        @T.prim_func
        def main(A: T.Tensor((M, K), gdtype),
                 B: T.Tensor((K, N), gdtype),
                 C: T.Tensor((M, N), "float32")):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
                As = T.alloc_shared((BM, BK), "float16")
                Bs = T.alloc_shared((BK, BN), "float16")
                Cacc = T.alloc_fragment((BM, BN), "float32")
                T.clear(Cacc)
                if use_chunk:
                    Cchunk = T.alloc_fragment((BM, BN), "float32")
                    for c in T.serial(NC):
                        T.clear(Cchunk)
                        for ko in T.Pipelined(KI, num_stages=stages):
                            T.copy(A[by * BM, c * kc + ko * BK], As)
                            T.copy(B[c * kc + ko * BK, bx * BN], Bs)
                            T.gemm(As, Bs, Cchunk)
                        for i, j in T.Parallel(BM, BN):
                            Cacc[i, j] += Cchunk[i, j]
                else:
                    for ko in T.Pipelined(KI, num_stages=stages):
                        T.copy(A[by * BM, ko * BK], As)
                        T.copy(B[ko * BK, bx * BN], Bs)
                        T.gemm(As, Bs, Cacc)
                T.copy(Cacc, C[by * BM, bx * BN])
        return main

    return _k()


# ---------------------------------------------------------------- artifacts ---
_SHMEM_RE = re.compile(r"\[(\d+)\]\.v_int64\)\s*=\s*\(\(int64_t\)(\d+)\)")


def _shared_bytes_from_host(host_src: str):
    """The dynamic-shared-memory argument of the packed kernel launch.

    TileLang's host stub pushes (ptrs..., grid, block, dyn_smem) onto an FFI
    stack; the dynamic shared size is the last integer pushed.
    """
    best = None
    for m in _SHMEM_RE.finditer(host_src):
        idx, val = int(m.group(1)), int(m.group(2))
        if best is None or idx > best[0]:
            best = (idx, val)
    return best[1] if best else None


def _ptxas_resource_usage(ptx_path: str) -> dict:
    """Register / spill counts.  JITKernel.n_regs is None in tilelang 0.1.11, so
    re-assemble the exported PTX with `ptxas -v` (same arch, same tool that
    produced the shipped cubin) and parse its report."""
    import subprocess
    exe = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.1"), "bin", "ptxas")
    try:
        p = subprocess.run([exe, "-arch=sm_89", "-v", "-o", os.devnull, ptx_path],
                           capture_output=True, text=True, timeout=300)
    except Exception as e:  # noqa: BLE001
        return {"ptxas_error": repr(e)}
    txt = p.stderr + p.stdout
    out = {"ptxas_report": txt.strip()}
    m = re.search(r"Used (\d+) registers", txt)
    if m:
        out["n_regs"] = int(m.group(1))
    m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    if m:
        out["n_spills"] = int(m.group(1)) + int(m.group(2))
        out["spill_store_bytes"] = int(m.group(1))
        out["spill_load_bytes"] = int(m.group(2))
    m = re.search(r"(\d+) bytes stack frame", txt)
    if m:
        out["stack_frame_bytes"] = int(m.group(1))
    return out


def _sass_mma_counts(path: str) -> dict:
    try:
        with open(path, "r", errors="ignore") as f:
            s = f.read()
    except OSError:
        return {}
    return {
        "sass_hmma": len(re.findall(r"\bHMMA\b", s)),
        "sass_imma": len(re.findall(r"\bIMMA\b", s)),
        "sass_ffma": len(re.findall(r"\bFFMA\b", s)),
        "sass_lines": s.count("\n"),
    }


# -------------------------------------------------------------------- build ---
def build(cfg: common.Config) -> common.Built:
    M, N, K = cfg.M, cfg.N, cfg.K
    BM, BN, BK, threads = cfg.BM, cfg.BN, cfg.BK, cfg.threads
    kc, stages = cfg.kc, cfg.stages

    assert K % BK == 0, f"BK={BK} must divide K={K}"
    if kc:
        assert K % kc == 0, f"kc={kc} must divide K={K}"
        assert kc % BK == 0, f"kc={kc} must be a multiple of BK={BK}"

    t0 = time.perf_counter()

    if cfg.arith == "fp32":
        gdtype = "float32"
        kern = _kernel_fp32(M, N, K, BM, BN, BK, threads, stages, kc)
        cast_in_run = False
    else:
        if cfg.cast not in common.CAST_MODES:
            raise ValueError(f"unknown cast mode {cfg.cast!r}")
        gdtype = "float32" if cfg.cast == "on_load" else "float16"
        kern = _kernel_fp16(M, N, K, BM, BN, BK, threads, stages, kc, gdtype)
        cast_in_run = (cfg.cast == "in_region")

    if cast_in_run:
        def run(A, B):
            return kern(A.half(), B.half())
    else:
        def run(A, B):
            return kern(A, B)

    # Force the lazy CUDA module load / first-launch cost out of the timed
    # region.  The JIT is shape-specialised, so the warm launch has to be at the
    # real shape; run() is used so cast=in_region warms its .half() path too.
    warm_dtype = torch.float16 if (gdtype == "float16" and not cast_in_run) else torch.float32
    wa = torch.zeros((M, K), dtype=warm_dtype, device="cuda")
    wb = torch.zeros((K, N), dtype=warm_dtype, device="cuda")
    _ = run(wa, wb)
    torch.cuda.synchronize()
    del wa, wb, _
    torch.cuda.empty_cache()

    compile_s = time.perf_counter() - t0

    # ---- artifacts (measured after the compile clock stops) ----
    grid = (N // BN + (1 if N % BN else 0), M // BM + (1 if M % BM else 0), 1)
    block = (threads, 1, 1)
    if cfg.arith == "fp32":
        how = ("variant A: explicit T.Parallel(BM,BN) x T.serial(BK) scalar FMA "
               "register tile on fp32 shared tiles; T.gemm deliberately NOT used "
               "(it emits tf32 mma.sync on sm_89). Lowers to FFMA only.")
    else:
        how = (f"T.gemm(fp16 smem -> fp32 fragment) = ptx mma.sync m16n8k16 "
               f"f16.f16.f32; kc={kc} "
               + ("(single fragment across all K, no flush)" if kc == 0 else
                  f"(Cchunk cleared/flushed into fp32 Cacc every {kc} K)")
               + f"; T.Pipelined(num_stages={stages}) "
               + ("= single smem buffer, cp.async + wait<0> each iteration "
                  "(no multi-buffering)" if stages == 1
                  else f"= {stages}-deep multi-buffered smem pipeline")
               + f"; cast={cfg.cast} (global dtype {gdtype})")

    artifacts = {
        "grid": list(grid),
        "block": list(block),
        "backend_detail": how,
        "tilelang_version": tilelang.__version__,
        "tilelang_disk_cache": _CACHE_ENABLED,
        "global_dtype": gdtype,
        "smem_dtype": "float16" if cfg.arith != "fp32" else "float32",
    }
    try:
        artifacts["cuda_source"] = kern.get_kernel_source()
    except Exception as e:  # noqa: BLE001
        artifacts["cuda_source_error"] = repr(e)
    try:
        artifacts["shared_bytes"] = _shared_bytes_from_host(kern.get_host_source())
    except Exception as e:  # noqa: BLE001
        artifacts["shared_bytes_error"] = repr(e)
    for attr in ("n_regs", "n_spills"):
        v = getattr(kern, attr, None)
        if v is not None:
            artifacts[attr] = v

    if _DUMP_ASM:
        d = os.path.join(common.ARTIFACTS_DIR, "tilelang")
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, cfg.key().replace("/", "_"))
        try:
            kern.export_ptx(base + ".ptx")
            artifacts["ptx_path"] = base + ".ptx"
            with open(base + ".ptx", "r", errors="ignore") as f:
                artifacts["ptx"] = f.read()
            artifacts.update(_ptxas_resource_usage(base + ".ptx"))
        except Exception as e:  # noqa: BLE001
            artifacts["ptx_error"] = repr(e)
        try:
            kern.export_sass(base + ".sass")
            artifacts["sass_path"] = base + ".sass"
            artifacts.update(_sass_mma_counts(base + ".sass"))
        except Exception as e:  # noqa: BLE001
            artifacts["sass_error"] = repr(e)
        try:
            kern.export_library(base + ".so")
            artifacts["cubin_path"] = base + ".so"
        except Exception:  # noqa: BLE001
            pass
        with open(base + ".cu", "w") as f:
            f.write(artifacts.get("cuda_source", ""))

    notes = (f"tilelang {tilelang.__version__} variant {cfg.variant}: "
             f"{BM}x{BN}x{BK}/{threads}thr kc={kc} stages={stages} "
             f"arith={cfg.arith} cast={cfg.cast}; disk cache "
             f"{'ON' if _CACHE_ENABLED else 'OFF (compile_s is a cold compile)'}")

    return common.Built(run=run, compile_s=compile_s,
                        input_dtype=cfg.input_dtype,
                        artifacts=artifacts, notes=notes)
