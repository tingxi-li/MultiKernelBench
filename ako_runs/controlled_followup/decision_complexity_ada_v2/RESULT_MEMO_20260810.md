# Q1--Q4 GPU DSL study completion memo (non-frozen)

This memo is reporting metadata. It is not frozen campaign source or evidence,
and it does not alter any sealed result. All four question-focused experiments
are landed on `origin/cross-dsl-6op-ncu-redo`; a negative, unresolved, or
protocol-invalid outcome is considered landed only because its complete
evidence was retained, independently rederived, and pushed without relabeling.

## Bounded answers

| Question | Controlled evidence | Answer supported by this evidence |
|---|---|---|
| **Q1. Does any DSL have a higher performance ceiling?** | `finite_frontier_ada_v6`: 3/3 fresh artifact admissions and 120/120 terminal records; 15 paired blocks on each of two distributions; byte-identical sham labels; procedure-selected TileLang `g05` and Triton `g05`. | For this one frozen workload, shape, Ada GPU, finite candidate space, and selection procedure, Triton `g05` had lower latency. The TileLang/Triton ratio interval was `[1.033055, 1.048703]` on the positive distribution and `[1.015810, 1.045382]` on withheld-signed inputs, and both cleared the sham floor. This is **not** evidence of a general or theoretical DSL ceiling. |
| **Q2. Are optimization trajectories transferable?** | `native_trajectory_replication_ada_v3`: 12/12 admitted cells, 420/420 fresh-process timing records and 420/420 position receipts; 15 blocks, two distributions, fixed same-campaign recurrence rule, and 60 byte-identical sham records. | No destination met the preregistered gain-recurrence rule. Triton's second step cleared the floor in both distributions, but the TileLang reference step remained unresolved, so the required recurrence chain did not hold. No optimization-transfer claim is supported. |
| **Q3. Do TileLang's lower abstraction levels improve efficiency?** | `tilelang_abstraction_v7`: 1,024/1,024 correctness rows, two exact-executable NCU profiles, 120/120 fresh-process timing records and receipts, two distributions, and 60 sham records bound to one F1 implementation. | No reportable lower-level latency advantage was found for the fixed F4c/F1 pair. The positive interval cleared the sham floor but not the `0.03` log-ratio directional threshold, and it remained within the `0.05` equivalence bound; the withheld-signed interval did not strictly clear its sham floor. Both frozen classifications have `reportable_direction=false`. |
| **Q4. Do simpler kernels converge faster than complex kernels?** | `decision_complexity_ada_v2`: 24/24 serialized trajectories, 236/236 charged attempts and parent receipts, four replicates per arm, paired sham labels, and a target-first positive control. | **Unanswered.** The ledger is complete, but all 236 attempts failed the artifact audit before correctness or timing. The sham labels matched, while the target-first control failed in all four replicates. The frozen analyzer therefore sets `pilot_valid=false`, `interpretation_valid=false`, and `controlling=false`; no convergence or treatment interpretation is permitted. Even a valid pilot would compare only nested decision-set sizes for one TileLang family under one fixed uniform searcher, not semantic kernel simplicity in general. |

The useful cross-study conclusion is methodological: finite, matched,
sham-calibrated experiments can support local performance statements, but they
do not identify a language-wide ceiling or transfer law. Finite selection and
the withheld distribution bound Q1; sham and recurrence rules constrain Q2;
directional, equivalence, withheld, and sham rules constrain Q3; and Q4's
positive control invalidates that pilot.

## Q4 invalid-pilot incident

Q4 finished its frozen failure ledger rather than stopping or substituting
cells:

- 24 trajectory completions and 24 trajectory-parent receipts;
- 236 attempt records and 236 matching attempt-parent receipts;
- all attempts have `terminal_status=AUDIT_FAILED` and the exact error
  `ProtocolError: candidate implementation artifacts drift`;
- all six arms have zero events and are right-censored at their fixed caps;
- the label-sham control passes 4/4 exact comparisons;
- the target-hint control produces 0/4 required first-attempt events;
- no attempt reaches a correctness gate, latency trial, or ratio outcome.

Source inspection identifies a protocol comparator bug. The predecessor audit
compacted strings longer than 4,096 bytes before retaining build artifacts, and
Q4 hashed that compacted artifact dictionary into its material index. Q4 then
hashed the fresh builder's un-compacted artifact dictionary during execution.
The primary `cuda_source` field for each of the 19 cells is 8,569--15,415
bytes, so the two digest domains necessarily differ even for byte-identical
generated source. This is a measurement-protocol failure, not evidence of
implementation drift, TileLang nondeterminism, or optimization difficulty.

The apparent `+3` and `+13` attempt contrasts reproduce different frozen cap
lengths under universal audit failure. Their active-time counterparts measure
repeated build-plus-failed-audit overhead. Neither is convergence evidence. Any retry must
be a new campaign version that hashes the same canonical artifact
representation on both sides, records expected and observed component hashes,
and passes a target-first cold-cache preflight before launching a full ledger.
Even a repaired rerun would estimate finite decision-space burden for one
`register_fused.tilelang` family under one fixed uniform searcher, not semantic
simple-versus-complex kernel convergence in general.

## Evidence and provenance

| Experiment | Result artifact and SHA-256 | Pushed result commit |
|---|---|---|
| Q1 | `finite_frontier_ada_v6/results/final_summary.json` — `19e1a00c9f99a5820b5a72fff540075bc460a6e681f76079ed0ebc17ea389193` | `32ffc772b823a7eb5b9583b2831a88cf81be028d` |
| Q2 | `native_trajectory_replication_ada_v3/results/analysis.json` — `f2f6ff6bf1624551bab240d92132e7957a65c348d55181b5981abee2ad931ca9` | `c830c0a8d3cbb600d9bbaeacc5ab4a20edcb195c` |
| Q3 | `tilelang_abstraction_v7/results/ada_v7r1/timing/summary.json` — `c72b5ea3f75554dae57e896dfa9d4fde30f041540d7b83496b9cf1f80082bf6f` | `fb991965a756d3b7f72dc728ab4d118aef794ec1` |
| Q4 | `decision_complexity_ada_v2/results/analysis.json` — `0bf54a6ab0ec91913c72f828bcc2517ddb7b58e891694febd02b28c3076ec7ba` | `968926017d7c190ea6e443309609b783533a6f2d` |

Q4 additionally binds execution lock
`07839a323a4fe93ddcd4f57599b8f669b394a7f1b5e2212af6f0d7552a4e1e`,
source bundle
`e77929eac640c9dac9d0c0a6a8ae491096e53758dacb015dc3cdb4c17461b9f6`,
prelaunch provenance
`5330383ae1e1b6b7b04eb90735debcf18890584d8b1fc6eabb727d46fdaacd53`,
launch receipt
`13454e0179a9b0ee9ab0cd2c733efdac05c9a1bf508b5ddcc926ffd278e84c91`,
and run status
`aa3526728b5ca3c483f6b297c7ec21074f6a1cecac2ac07011eb24dd83643dfd`.
The official, independent, and pushed-tree Q4 analyses are byte-identical.

Q1 source/admission/result commits are `7287c13f1ae9441f6af6f5b5e259670c671202be`,
`408c5f607ba15b1f0157eca2b8d729390c8f2b9a`, and the result commit above.
Q2 prepare/freeze/admission/result commits are
`2d655593076cc1ff3b36205ab60dd7f34f26c9c8`,
`065322a0c729e578bd84daf74619f2aab0127dfc`,
`d16feedb8d08c3e3d82aa74ddf8ceecf8f5eff26`, and the result commit above.
Q3 source/lock/admission/result commits are
`83f2badf5618b2201046321175e9f4e5b5e960b7`,
`c28b19e2f5d727b6fdca3b164dc310a3c34bf677`,
`69c90cbb501314ea8327cf77a25b8d39a00fbcbd`, and the result commit above.
Q4 freeze and result commits are
`837dd764f8a7bc595ea9eb32a9827bbf7f5561bc` and the result commit above.

The legacy `reciprocal_v2`, `effort_frontier_v1`, and `convergence_v2`
programs remain non-results. These narrower successors do not retroactively
complete or revive them. Cross-architecture claims remain blocked until the
planned non-`sm_89` feasibility matrix exists.
