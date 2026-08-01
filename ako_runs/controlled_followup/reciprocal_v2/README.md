# Reciprocal recipe transfer v2

This round-2 campaign turns recipe origin, destination, transfer mode, and
translator into crossed factors.  It adds the missing CUDA-native origin and
uses two source-isolated translators, so a destination effect can be separated
from donor anchoring and translator skill.

The two generated manifests each contain exactly 48 cells:

```text
3 origins x 4 destinations x 2 modes x 2 translators = 48
```

The origins are the frozen Phase-1 TileLang card, the grouped Triton-native
card, and `cuda_unlimited.D.128x128x32.kc2048.s3.fp16.precast`.  Retuned cells
attempt exactly 19 configurations; build failures consume the budget.  Every
candidate is checked against the byte-frozen matmul-v4 instrument.  The only
permitted correctness amendment is the common KC ladder
`8192,4096,2048,1024,512`; the largest value passing all destinations and both
translators is frozen before timing.  No threshold may change.

Audit precedes performance.  Primary timing requires a cell-specific audit
receipt and uses 15 randomized complete blocks on physical GPU 0.  The analyzer
reports cell intervals, translator bounds, paired destination contrasts, and
the origin-by-destination interaction.  All claims remain specific to the
finite recipe/search budget, workload, host, and Ada architecture.

Lifecycle and CPU-only commands:

```bash
python ako_runs/controlled_followup/reciprocal_v2/make_manifests.py --check
python ako_runs/controlled_followup/reciprocal_v2/validate.py
python ako_runs/controlled_followup/reciprocal_v2/freeze.py --freeze
python ako_runs/controlled_followup/reciprocal_v2/freeze.py --verify
python ako_runs/controlled_followup/reciprocal_v2/capture_evidence.py build --stage prereg --name prereg_v1
python ako_runs/controlled_followup/reciprocal_v2/launch.py --stage audit --list
python -m unittest ako_runs/controlled_followup/reciprocal_v2/test_reciprocal_v2.py
```

Execution has three fail-closed stages: `audit`, `screen`, then `primary`.
Source/config/work-mapping/dynamic-work/generated-code receipts for every
attempt must exist before screening. Screening records exactly two repetitions
for every audit-eligible attempt, freezes the minimum-median winner, then runs
the locked terminal v4 split. Only terminal-eligible selections may enter the
15-block primary confirmation. The raw primary stream must follow the frozen
complete-block permutations and use a fresh process instance for every cell.

The KC resolution lock is evidence-bearing rather than self-attested: it must
bind the implementation registry and v4 gate, contain the exact descending
ladder prefix through the selected value, and content-address a complete v4
summary for all 24 origin/destination/translator implementations at each tried
KC. A preceding all-pass value or a selected non-pass value is rejected.

Produced-artifact contracts are enforced by `validate.py`:

- `implementation_registry.json` has exactly 48 unique cell entries, each
  factor-bound and content-addressed beneath its assigned translator root.
- `translator_isolation.json` binds two distinct worktree identities and
  content-addressed isolation transcripts showing that each translator lacked
  access to the other's sources and to performance results. The implementation
  registry and pushed provenance must bind this lock.
- `recipe_resolution_lock.json` binds that registry and the gate lock. Its
  attempted KC values are the exact descending prefix through the selected KC;
  every attempt content-addresses 24 complete v4 summaries.
- Each audit receipt contains exactly one or 19 ordered attempts. Every built
  attempt binds five separate audit summaries (source, config, mapping, dynamic
  Tensor-Core work, and generated code) plus a tuning-split v4 summary. Failed
  builds are retained and consume their attempt.
- Each selection receipt has two screen records for every audit-eligible
  attempt, selects the deterministic minimum-median winner, and binds its
  complete terminal validation-split v4 summary.
- Each primary row binds all controlling locks, the selection receipt and
  selected source, its unique process instance, GPU UUID, block, and frozen
  block position. Exactly 720 serialized rows are accepted.

`validate.py --launch-ready` and `launch.py` are deliberately fail-closed.  They
will refuse GPU work until independent source registries, all retune plans, the
KC resolution, an externally timestamped pushed commit, and (for primary work)
all audit receipts are content-addressed.  A missing NVIDIA driver is an
additional hard blocker, never a reason to emit placeholder results.
