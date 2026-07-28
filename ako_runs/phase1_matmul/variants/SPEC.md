# Variant implementation spec — Phase 1 matched GEMM

Every DSL module implements the **same four kernels** so that the only thing
differing between two measured numbers is the thing under test. Read this
whole file before writing code.

## Problem

`C = A @ B`, `A:(M,K)`, `B:(K,N)`, `M=2048, K=8192, N=4096`, row-major,
contiguous. Output `C` is **fp32** always, allocated inside `run()`.

## Entry point

```python
# variants/<dsl>_gemm.py
import common
def build(cfg: common.Config) -> common.Built: ...
```

`common.Built(run, compile_s, input_dtype, artifacts, notes)`:

- `run(A, B) -> C` — `A`,`B` arrive with dtype `cfg.input_dtype`; returns fp32
  `(M,N)`. **Allocate the output inside `run`** (a real KernelBench `forward()`
  does), so allocation cost is inside the timed region for every DSL equally.
- `compile_s` — wall-clock from "start building" to "kernel is launchable",
  measured by you. Force any lazy JIT by launching once on tiny tensors inside
  the compile window, so the first timed call is not a compile.
- `input_dtype` — `torch.float16` iff `cfg.arith=="fp16" and cfg.cast=="precast"`,
  else `torch.float32`. Just return `cfg.input_dtype`.
- `artifacts` — dict; populate whatever you can of `cuda_source`, `ptx`,
  `cubin_path`, `n_regs`, `n_spills`, `shared_bytes`, `grid`, `block`,
  `backend_detail`.

## The four variants

`cfg.variant`, and the fields that carry it:

| variant | `cfg.arith` | `cfg.kc` | `cfg.stages` | what it is |
|---|---|---|---|---|
| A | `"fp32"` | 0 | 1 | fp32 arithmetic, **no tensor cores**, one accumulator over all K, DSL-native pipelining |
| B | `"fp16"` | 0 | 1 | fp16 operands into tensor cores, fp32 accumulator, **one accumulator chain over all K**, pipelining **off** |
| C | `"fp16"` | 2048 | 1 | as B, but the tensor-core accumulator is **cleared and flushed into a second fp32 accumulator every `cfg.kc` elements of K**; pipelining still off |
| D | `"fp16"` | 2048 | 3 | as C, plus a `cfg.stages`-deep software pipeline on the global→shared loads |

`cfg.kc == 0` means "no chunk flush — a single accumulator across the whole K
extent". `cfg.kc > 0` must divide `K`; the driver only ever passes divisors.

### Variant A must not touch tensor cores

This is the no-tensor-core floor, so it must be genuine fp32 FMA on the CUDA
cores. Concretely:

- **triton** — `tl.dot(a, b, acc, input_precision="ieee")`. The default
  (`"tf32"`) silently uses tensor cores and would make A meaningless.
- **tilelang** — do **not** use `T.gemm` for A unless you have verified from the
  SASS that it emits no `HMMA`/`IMMA`. Prefer an explicit scalar/`T.Parallel`
  FMA loop at the same blocking.
- **cuda_noptx / cuda_unlimited** — plain `c += a*b` register-tiled SGEMM. No
  `wmma::`, no `mma.sync`.

Report in `artifacts["backend_detail"]` how you achieved it, and confirm from
SASS that A contains zero `HMMA` instructions.

### `cfg.stages == 1` must mean the pipeline is actually off

- **triton** — `num_stages=1` on the `@triton.jit` launch.
- **tilelang** — `T.Pipelined(..., num_stages=1)`, or a plain serial loop.
- **cuda_noptx** — synchronous single-buffered `__syncthreads()`-separated loads
  for stages==1; `__pipeline_memcpy_async` (`<cuda_pipeline.h>`) double/triple
  buffering for stages>1. `__pipeline_memcpy_async` is a CUDA intrinsic, **not**
  inline PTX, so it is legal in this lane.
- **cuda_unlimited** — synchronous for stages==1; inline `cp.async.cg.shared.global`
  + `cp.async.commit_group`/`wait_group` for stages>1.

## Matched geometry (do not deviate)

`cfg.BM`, `cfg.BN`, `cfg.BK`, `cfg.threads` are given and must be honoured
exactly. Your code must be parameterized on these, not hardcoded. Three points
are defined in `common.GEOMS`:

| name | BM | BN | BK | note |
|---|---|---|---|---|
| `primary` | 128 | 128 | 32 | the matched point; all tables default to it |
| `secondary` | 128 | 256 | 32 | wider N tile → ~1.6 GB through L2 per launch instead of ~2.15 GB |
| `incumbent` | 128 | 256 | 64 | the shape the published tilelang winner used |

**`incumbent` is not a matched point.** At `BK=64` a 3-stage pipeline needs
`(128·64 + 64·256)·2·3 = 147456 B` of shared memory and sm_89 allows 101376 B,
so variant D cannot run there at `stages=3`. It is carried only for the casting
comparison against the published solution, at `stages=2`. If you are asked for
`incumbent` at `stages=3`, fail loudly with the shared-memory arithmetic rather
than silently reducing the buffer count.

- **No block-index swizzle / group-M reordering.** Use the plain 2-D grid
  `(ceil(N/BN), ceil(M/BM))` in every DSL. A swizzle is a real L2 lever and
  would leak into the comparison unevenly.
- **No split-K across blocks / atomics.** "split-K" in this study means the
  *in-block accumulator chunk flush* described above, nothing else.
- **No autotuning.** The config is the config. `@triton.autotune` is forbidden;
  tilelang JIT must not be allowed to pick its own tile sizes.

## `cfg.cast` — the casting control

Only meaningful when `cfg.arith == "fp16"`. All three must work:

| `cfg.cast` | `input_dtype` | what `run` does |
|---|---|---|
| `precast` | fp16 | operands already fp16; kernel reads fp16 global. **This is the default for A–D.** |
| `in_region` | fp32 | `run` itself calls `.half()` on both operands, then launches the same fp16 kernel. The conversion is inside the timed region. |
| `on_load` | fp32 | kernel reads **fp32** global memory and converts to fp16 on the way into shared memory. No separate conversion pass, no fp16 copy of the operands anywhere. |

`on_load` needs a second kernel body (fp32 global pointers, fp16 smem). It is
the only one that changes the kernel; the other two differ only in the host glue.

## Correctness

The harness gate is `|ref - got| <= 1e-4 + 1e-4*|ref|` against `torch.matmul`
on fp32, elementwise, no exceptions. Under `torch.rand` inputs `|ref|≈2048`, so
the budget is ≈0.205. **Variant B is expected to fail this gate — that is a
result, not a bug.** Do not "fix" B by adding a chunk flush; B *is* the no-flush
arm. Do not silently change dtypes, tile sizes, or accumulation order to make a
variant pass.

## What you must NOT do

- No `torch.matmul`, `torch.mm`, `nn.Linear`, `F.linear`, or any cuBLAS call
  inside `run` — the kernel must do the whole GEMM.
- No caching of a converted operand across calls (that is a Phase-2 question and
  it would corrupt the timing here).
- No reading of the other DSLs' modules to copy a tile schedule. Implement your
  DSL's natural expression of the specified structure.

## Self-check before you report done

```bash
cd /home/lxt230026/MultiKernelBench/ako_runs/phase1_matmul
for v in A B C D; do
  CUDA_VISIBLE_DEVICES=$GPU python runner.py --dsl $DSL --variant $v \
    --warmup 50 --trials 20 | tail -1
done
```

Report for each variant: median ms, TFLOP/s, `max_abs_err`, `gate_pass`,
`compile_s`. Also run the secondary geometry (`--geom secondary`) and the two
non-default cast modes (`--set cast=in_region`, `--set cast=on_load`) at
variant D, and a `--set kc=512` and `--set kc=8192` at variant C to prove the
`kc` knob is live.
