# MultiKernelBench

A benchmark for evaluating LLMs' ability to generate kernels for various platform. Now supporting CUDA and triton kernels for GPUs, MUSA kernels for Moore Threads GPUs, Ascendc and TileLang kernels for NPUs, pallas kernels for TPUs and SYCL kernels for Intel GPUs.

---

## 📌 This branch — `cross-dsl-6op-ncu-redo`: AKO4ALL cross-DSL kernel optimization study

A research fork that ports and optimizes kernels across **four GPU DSLs** and asks *where each DSL's performance ceiling actually lies.* Each op is optimized in every DSL on one host (NVIDIA **RTX 6000 Ada**, nvcc 13.1, torch 2.10+cu128, TileLang 0.1.11) via the **AKO4ALL** profile→edit→bench→log loop, benched against the **same PyTorch golden**. All artifacts live under [`ako_runs/`](ako_runs/).

The four DSLs: **triton** · **cuda_noptx** (plain CUDA, no inline PTX) · **cuda_unlimited** (CUDA + inline PTX) · **tilelang**. Correctness is the harness fp32 1e-4 oracle; every solution's `forward()` is allocate/launch glue only and passes the anti-hack detector ([`utils/cheating_detection.py`](utils/cheating_detection.py)).

**Questions:** (1) does any DSL have a higher performance ceiling? (2) what is DSL-unique in the optimization trajectory? (3) do trajectories transfer between DSLs?

### Headline findings — two regimes

- **Memory-bound / index / elementwise / low-arithmetic-intensity ops** (reduction, depthwise conv, layer_norm, group_norm, gather, scatter, …): **no capability ceiling.** Every DSL reaches the roofline; the winning lever is algorithmic and transfers ~100% across DSLs. Inline PTX is a **red herring** (≤0.6% on every memory-bound op).
- **Tensor-core ops** (matmul, matmul+gelu+softmax, attention): a **real, wide capability ceiling — owned by the compiler DSL.**

**Verified compute-bound results** (speedup vs the PyTorch golden; all CORRECT @ fp32 1e-4, detector-clean, independently re-benched):

| op | triton | cuda_noptx | cuda_unlimited | tilelang |
|---|---|---|---|---|
| sum_reduction | 1.01 | 1.01 | 1.01 | 1.01 |
| conv_depthwise | 1.41 | 1.46 | 1.46 | 1.53 |
| **standard_matmul** | 0.79 | 0.59 | 1.11 | **4.13** |
| **matmul_gelu_softmax** | 2.36 | 1.05 | 1.24 | **5.04** |
| **scaled_dot_product_attention** | 1.27 | 1.73 | 1.71 | **3.44** |

The GEMM ceiling is set by two factors: **precision-managed tensor cores under the 1e-4 gate** (fp16/tf32 + split-K accuracy recovery) × **compiler auto-pipelining vs hand-built**. **PTX's role splits by op class:** a red herring on memory-bound ops, *decisive for `cuda_noptx`→parity* but *not sufficient for the frontier* on GEMM — `tilelang`'s compiler beat the hand-PTX lane ~3–4× with **zero PTX**.

**Convergence:** with every benched config logged through the wrapper, the compiler DSLs (triton/tilelang, JIT) converge to *higher* ceilings in **~5× less compute** than the nvcc lanes (matmul: tilelang 85 s → 4.13× vs cuda_unlimited 407 s → 1.11×).

### Reproducibility tooling ([`ako_runs/tools/`](ako_runs/tools/))

- `timed_bench.sh` — wraps a cell's bench, times compile+bench only, appends one row/variant to that cell's `convergence.csv` (`--gpu N` pins a card; `--serialize` GPU-lock keeps concurrent memory-bound benches from contaminating the ratio).
- `ncu_driver.py` / `ncu_profile.sh` — Nsight Compute in the loop; steer by DRAM bytes/passes, never `%peak`.
- `check_gate.py` + `committed_baseline.csv` — regression gate vs committed speedup floors.
- `convergence_log.py` — post-hoc `compute_s`-to-ceiling.

### Documents ([`ako_runs/`](ako_runs/))

- [`RESULTS.md`](ako_runs/RESULTS.md) — the 12 memory-bound / index / elementwise ops × 4 DSLs.
- [`CROSS_DSL_FINDINGS.md`](ako_runs/CROSS_DSL_FINDINGS.md) — Q1/Q2/Q3 answers + transferability rules.
- [`COMPUTE_FRONTIER_FINDINGS.md`](ako_runs/COMPUTE_FRONTIER_FINDINGS.md) — the 5 compute-bound ops, the tensor-core ceiling, and convergence.
- [`CONVERGENCE_PROTOCOL.md`](ako_runs/CONVERGENCE_PROTOCOL.md) — the frozen measurement protocol.
- [`GAP_ANALYSIS.md`](ako_runs/GAP_ANALYSIS.md) · [`P2_LEVER_TESTS.md`](ako_runs/P2_LEVER_TESTS.md) · [`NCU_VALIDATION.md`](ako_runs/NCU_VALIDATION.md) — supporting analyses and ncu ground truth.

*(The rest of this README describes the upstream MultiKernelBench harness.)*

---

## Directory Structure

```text
MultiKernelBench/
├── ascend_op_projects/     # Ascend operator projects and extensions
├── backends/               # Backend implementations (cuda, triton, ascendc, pallas, sycl, etc.)
├── prompt_generators/      # Prompt strategy implementations
├── prompts/                # Prompt templates and related resources
├── reference/              # PyTorch reference implementations used for correctness checks
│   ├── activation/
│   ├── attention/
│   ├── convolution/
│   ├── matmul/
│   └── ...
├── utils/                  # Utility modules
├── config.py               # Global runtime and model configuration
├── dataset.py              # Dataset loading and task organization
├── generate_and_write.py   # Generate kernels and write to output directory
├── generate_baseline_statistics.py  # Generate baseline statistics across tasks/categories
├── evaluation.py           # End-to-end evaluation entrypoint
└── eval_single_runner.py   # Single task/category evaluation runner
```

`reference/` is organized by `category`. The `--categories` argument should use these directory names (e.g., `activation`, `attention`, `convolution`, `matmul`, etc.).

## Latest News
- **18/06/2026** – Added **MUSA selected-shot prompt strategy** support for **Moore Threads GPUs**; use `--language musa --strategy selected_shot`.
- **11/06/2026** – Added initial **MUSA backend** support for **Moore Threads GPUs**; use `--language musa --strategy add_shot`.
- **28/05/2026** – Added **AscendC direct-launch backend** support for `<<<>>>` kernel launches; use `--language ascendc_direct_launch --strategy add_shot`.
- **28/05/2026** – Added **anti-hack detection** for generated code that replaces custom kernels with PyTorch/Python compute.
- **27/10/2025** – Introduced a new task category featuring 15 attention tasks, including MQA and GQA. 
- **08/10/2025** – Added **TileLang-Ascend backend** support for **Ascend NPUs**.  
- **12/08/2025** – Added **SYCL backend** support for **Intel GPUs** – thanks to **NinaWie** for the contribution!  
- **18/07/2025** – 🎉 Announced the open-source release of **MultiKernelBench**, a **multi-platform benchmark for kernel generation**, now publicly available!


## Quick start

### Set up
```bash
conda create --name multi-kernel-bench python=3.10
conda activate multi-kernel-bench
pip install -r requirements.txt

# For NPU users:
pip install torch-npu==2.7.1

# For Intel GPU (torch xpu) users:
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/xpu

# For Moore Threads GPU (MUSA) users:
# Install the torch_musa package matching your MUSA runtime.
```
You can rent GPU or NPU resources from online platforms such as [autodl](https://www.autodl.com/home). For TPU resources, you can use services like [Google Colab](https://colab.research.google.com/)

### Config
Set configurations in config.py, including temperature and top_p for LLM. For CUDA-based evaluation, set arch_list. For Ascendc evaluation, set ascendc_device = ai_core-<soc_version>.

### Set API keys for LLM
```
export DEEPSEEK_API_KEY=<your deepseek api key>
export DASHSCOPE_API_KEY=<your aliyun api key>
export OPEN_ROUNTER_KEY=<your openrouter api key>
```

### Generate kernels using LLM and write them to output
```bash
python generate_and_write.py --model deepseek-chat --language ascendc --strategy add_shot --categories activation
```
For MUSA generation, use `--language musa --strategy add_shot` or `--language musa --strategy selected_shot`.
Generated code is saved in ```output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}```.

#### Available Arguments

- `--runs`: Number of runs (default: `1`)
- `--model`: Model name (default: `deepseek-chat`)
- `--language`: Language used (default: `cuda`)
- `--strategy`: Prompt strategy type (default: `add_shot`)
- `--categories`: Space-separated list of categories (default: `activation`)  
  Use `all` to include all categories.

### Evalutation
```bash
python evaluation.py --model deepseek-chat --language ascendc --strategy add_shot --categories activation
```
Evaluation result is saved in ```output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}/result_{category}.json```.

### Evaluate a Single Response
Use `eval_single_runner.py` when you already have one generated response file and want to evaluate one task directly:

```bash
python eval_single_runner.py \
  --input prompts/ascendc_direct_launch_model_relu.json \
  --op relu \
  --language ascendc_direct_launch \
  --result relu_result.json
```

The runner writes one JSON object containing compile status, correctness, performance, hardware, and anti-hack detection fields.

## Adding a Prompting Strategy for a New or Existing Language

To add a custom prompting strategy, follow these steps:
1. **Create a Python file:**  
   Add a new file under `prompt_generators/` named as:  
   `prompt_generators/{language}_{strategy_name}.py`  

2. **Create a New Strategy Class**

   - Inherit from `BasePromptStrategy`.
   - Implement the `generate(self, op)` method.

2. **Register the Strategy**

   Use the `@register_prompt(language, strategy_name)` decorator with the desired language and strategy name.
## Adding a New Backend

To integrate a new backend, follow these steps:

1. **Create a New Python File**

   Add a new file under `backends/` named as:  
   `backends/{backend_name}.py`

2. **Create a Backend Class**

   - Inherit from `Backend`.
   - Implement all required methods:
     - `get_device()`
     - `get_hardware_name()`
     - `compile(generated_code, op)`
     - `correctness_execution(ref_src)`
     - `time_execution()`
     - `cleanup()` (optional)

3. **Register the Backend**

   Use the `@register_backend(name)` decorator with your backend's unique name.

## Credits

This project uses code from [KernelBench](https://github.com/ScalingIntelligence/KernelBench), licensed under the MIT License.
