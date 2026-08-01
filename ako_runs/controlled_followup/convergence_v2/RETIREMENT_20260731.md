# Convergence v2 retirement

`convergence_v2` is retired as designed and must not launch. The existing conditional
preflight remains frozen as historical protocol evidence; it is no longer an
authorization path. The official `launch.py` refuses without contacting providers or
GPUs, even if every old preflight dependency later becomes available.

The 192-row core manifest, 128-row prompt-extension manifest, receipts, prompts, and
preregistration bundles are preserved byte-for-byte. The prompt extension is excluded
from every future launch and must not be copied into a successor.

Any revival needs a new campaign identifier and preregistration. It must first define
the convergence event and controlling τ, then may treat the sum/SDPA gate calibration
and a four-trajectory executor smoke test as independent prerequisites. None of that
makes this retired campaign launchable.
