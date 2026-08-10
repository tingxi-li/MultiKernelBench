# TileLang abstraction v7 result memo (non-frozen)

This memo is reporting metadata, not part of the frozen campaign source or
evidence. The controlling local result is the untouched `ada_v7r1` result tree.

## Completion and provenance

- Source commit: `83f2badf5618b2201046321175e9f4e5b5e960b7`.
- Lock commit: `c28b19e2f5d727b6fdca3b164dc310a3c34bf677`.
- Admission checkpoint: `69c90cbb501314ea8327cf77a25b8d39a00fbcbd`.
- Campaign-lock SHA-256: `8cf8549b572c195a33c6b503804cf6a0a0fc9db8b63b4bb1716f560224877649`.
- Admission: 1,024/1,024 correctness rows passed, 512 per arm; both NCU
  profiles returned zero and matched their admitted executables. The two arm
  identities are distinct. Both artifact runtime-temp censuses are empty.
- Timing: 120/120 fresh-process records and 120/120 chained position receipts;
  every record has 100 trials and a successful exact-artifact load. The 60 sham
  records bind the one byte-identical F1 implementation.
- Timing-summary SHA-256: `c72b5ea3f75554dae57e896dfa9d4fde30f041540d7b83496b9cf1f80082bf6f`.
  An independent derivation in `/tmp` was byte-identical.
- Timing evidence-bundle SHA-256:
  `7b06f50eb13f04aa6c9c4eb849de4f82a992f0fc429adc22d0809b41ed3bc205`.

## Bounded result

The estimand is `T_low/T_high` for the fixed F4c/F1 fused-softmax pair on the
frozen Ada GPU. Trials 60--99 control.

| Distribution | Settled-tail interval | Median | Sham floor (log ratio) | Frozen classification |
|---|---:|---:|---:|---|
| positive | [0.988095, 0.991628] | 0.989211 | 0.001673 | unresolved; within equivalence bound |
| withheld-signed | [0.989015, 1.000000] | 0.993676 | 0.000000 | unresolved below sham resolution; within equivalence bound |

The positive-distribution interval is numerically below one and clears its sham
resolution floor, but the complete effect remains inside the preregistered
`0.05` log-ratio equivalence bound. The withheld-signed interval does not clear
its sham floor. The frozen analyzer therefore marks `reportable_direction=false`
for both distributions.

Accordingly, this experiment does **not** support a reportable claim that the
lower TileLang abstraction level improves efficiency. It supports the narrower
finding that, for this one matched pair, any observed advantage is small enough
to be practically equivalent under the frozen rule and is not robust across
the withheld distribution. No cross-kernel, cross-architecture, or general
TileLang abstraction-level ranking follows.
