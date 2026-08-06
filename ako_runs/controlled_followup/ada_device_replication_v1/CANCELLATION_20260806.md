# Ada device replication cancellation

Status: **cancelled, incomplete, and permanently non-controlling**.

The user called off `ada_device_replication_v1` on 2026-08-06 at approximately
01:24 UTC because it does not address the four GPU DSL research questions that
motivated the current work. The outer launcher received `SIGINT`, terminated
all four child process groups, and exited with status 130. A post-stop
`nvidia-smi` compute-process query was empty.

No wave completed. The retained partial census contains 178 of 1,216 requested
records:

- GPU 0: 43
- GPU 1: 44
- GPU 2: 48
- GPU 3: 43
- terminal outcomes: 128 `GATE_PASSED`, 31 `BUILD_FAILED`, 19 `UNSUPPORTED`

These records must not be completed, analyzed as a replication, used for
selection, or cited as evidence for performance ceiling, trajectory transfer,
TileLang abstraction efficiency, or convergence. They are retained only as an
audit trail of the cancelled launch.

Bound launch materials:

- launch commit: `259ce8aa5055211cca4d5de9175f02a212cf0dd8`
- execution lock SHA-256:
  `42e581c1d8d6c9feefeb51bd912741320dc20b5bcfb9ce29335f382fec75b3fd`
- launch receipt SHA-256:
  `f4c4f3457dde852fded266dee5f171e2ba835e0004b28c7657226110fd9e1ef0`
