# GPU DSL experiment execution schedule — 2026-08-05

## Kickoff decision

Kickoff preflight is complete on physical GPUs 0–3. All four devices passed an
allocation, kernel, synchronization, and result check; no compute process was
active at capture. The machine-readable receipt is
[`EXPERIMENT_KICKOFF_20260805.json`](EXPERIMENT_KICKOFF_20260805.json), SHA-256
`6b3634b301bbcedb2ced9b39762e7ba3be092eeecd92b1dade6b526cd4a6a7e1`.

No experiment job was launched. Every detected device is an RTX 6000 Ada with
compute capability 8.9. The only experiment currently permitted by
`LATER_WORK_POLICY_20260731.md` is F0 on a specific non-sm_89 GPU. Using four
copies of the existing Ada architecture cannot identify a cross-architecture
effect and would violate the F0 treatment definition.

The user's instruction authorizes local use of GPUs 0–3, but it is not a remote
preregistration receipt and does not supply the missing frozen task, candidate,
gate, evaluator, or model/implementer materials.

## Dependency-gated schedule

| Order | Experiment | Trigger to enter | GPU schedule | Exit condition | Current state |
|---:|---|---|---|---|---|
| 0 | Hardware kickoff | Four visible GPUs and clean upstream-bound protocol commit | Run identity, occupancy, and smoke preflight once on GPUs 0–3 | Four successful device-bound checks | **Complete** |
| 1 | F0 second-architecture feasibility | A real non-sm_89 GPU; frozen identity, toolchain, source closure, support probes, gate, manifest, remote verification | Use only the bound non-sm_89 device; run 304 cells; no timing | Exactly 304 unique terminal receipts and independent archive verification | **Blocked: no non-sm_89 GPU** |
| 2 | Paper evidence checkpoint | Verified F0 matrix | Seal and report only; no GPU timing | Cross-architecture matrix committed and upstream | Blocked by F0 |
| 3 | A1 TileLang abstraction runtime | Post-paper successor; material receipt verifier; preregistered multi-family H/M pairs; power-derived repetitions | Randomize pair blocks over GPUs 0–3; keep both arms of a pair on the same GPU/block; balance H→M and M→H | Complete paired census, sham floor, receipts, independent verification | Blocked by materials and policy order |
| 4 | F1 finite frontier | Frozen public task corpus, common candidate census, current gates, A1 infrastructure, new successor lock | Stratify by task/lane and balance blocks over GPUs 0–3; separate feasibility from timing | Complete finite-candidate denominator and confirmed frontier | Blocked by F0, paper, and materials |
| 5 | T0/T1 trajectory transfer | Gate-legal donor prefixes from at least three origin DSLs; verified step artifacts; translation/execution ABI; F1 references | Translation remains isolated; target block execution rotates across GPUs 0–3; literal self-transfer controls run in every block | Complete prefix/mode/destination census including raw failures and order controls | Blocked by verified donors and ABI |
| 6 | A2 and C1/C2 search experiments | Frozen searcher/implementer, public tasks, separate tuning/terminal data, legal hidden references, power-derived replicates, remote preregistration | One isolated worker per GPU; frozen block randomization balances every arm across GPUs 0–3; no cross-worker result sharing | Exact manifest census, sham/positive controls, independent evaluator receipts and locked inference | Blocked by materials and predecessors |

This is an event-triggered schedule, not a calendar estimate. Dates would be
fiction until the hardware and material triggers exist.

## Four-GPU execution rules once a phase is eligible

1. Re-query index-to-UUID mapping immediately before launch; UUID, driver,
   clocks, temperature, toolchain and source hashes enter the launch receipt.
2. Freeze the complete manifest before assigning rows. Assignment comes only
   from the preregistered seed; it is never chosen from observed performance.
3. Use at most one experiment worker per physical GPU. Paired timing arms share
   the same GPU and block, and arm order is balanced.
4. Treat build, resource, launch and correctness failures as terminal outcomes;
   do not silently reschedule them onto a different GPU or candidate.
5. Stop the whole phase on missing/duplicate records, identity drift, source or
   lock mismatch, holdout leakage, failed positive controls, or a sham floor
   larger than the preregistered minimum effect.
6. Analyze only after the exact record census and all material receipts verify.

## Next admissible action

Attach or allocate a specific non-sm_89 GPU. The next action is then to create a
new hardware-bound F0 successor, remotely verify its frozen closure, and run the
304 feasibility cells without timing. Until that event, the four Ada GPUs remain
healthy and available but intentionally have no controlling experiment queued.
