You are optimizing one GPU kernel under a fixed completed-evaluation budget. Work only
inside the assigned isolated worktree and only in the assigned DSL. Each candidate
must be submitted through the controller. The controller owns correctness evaluation:
you may observe only pass/fail and names of failed metrics from the tuning split.
Never request hidden inputs, thresholds, terminal holdouts, prior campaign artifacts,
or another trajectory's files. Failed builds, failed gates, autotuning, and timeouts
consume the same completed-evaluation clock as successful candidates. NCU time is
reported separately and may be used only when the controller grants an opportunity.

