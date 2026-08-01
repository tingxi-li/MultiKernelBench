# Effort-frontier non-controlling pilot

Status: **design-only and not launchable**. The preregistered 20-trajectory,
four-lane manifest remains historical and unchanged.

The pilot is four trajectories: cuBLASLt and Triton, two independent contexts per
lane, with checkpoints at 0.5 and 2 hours. It exists only to exercise the effort clock
and failure ordering. It cannot support lane-performance inference or control a later
full-campaign claim.

Each trajectory has a 100,000 provider-token request-admission cap. Usage is the sum
of provider-reported `total_tokens` on durable response events. A final response that
crosses the cap is retained and reported, and no later request is allowed. Across four
trajectories, the planned ceiling is eight active-effort hours and 400,000 admitted
provider tokens, excluding a retained final-response overshoot.

Model identity is recorded as the requested alias, the model string on every provider
response, and the UTC resolution timestamp. An immutable provider revision is not a
launch veto; any response-model drift is retained and named as a threat to validity.

The four historical-style GPU assignments are intentionally not rotated: cuBLASLt
uses slots 0/1 and Triton uses slots 1/2. This lane×GPU imbalance is a named pilot
limitation, not something four trajectories can repair.

No executor is added. Real, content-addressed cuBLASLt and Triton executors and a
pilot-specific launcher remain mandatory before a separately frozen pilot may run.
