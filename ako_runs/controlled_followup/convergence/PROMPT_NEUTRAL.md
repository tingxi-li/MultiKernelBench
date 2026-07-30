# Neutral optimization prompt, version 1

Optimize the provided `{operation}` implementation using only `{dsl}` and the
declared lane policy.  Preserve the public function, shapes, output semantics,
and arithmetic-quality contract.  Previous solutions, reports, convergence
logs, target runtimes, and DSL rankings are unavailable.

Every candidate must be evaluated through the supplied controller.  A failed
build, failed correctness gate, autotune trial, and benchmark all consume the
controller's completed-compute budget.  You may stop proposing candidates, but
only the controller decides when the trajectory ends.  Do not infer a target
runtime from prior benchmark knowledge.

Two profiler opportunities are available: one frozen baseline profile and one
profile after candidate 2.  Profiler time is logged separately.  Return a short
description of each tested lever; the controller records source, result, and
timing artifacts.
