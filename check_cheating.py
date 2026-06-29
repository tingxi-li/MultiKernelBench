#!/usr/bin/env python3
"""Run MultiKernelBench's own anti-hack detector over all 12 AKO solutions."""
import sys
sys.path.insert(0, "/home/lxt230026/MultiKernelBench")
from utils.cheating_detection import detect_python_kernel_cheating

ROOT = "/home/lxt230026/MultiKernelBench"
OPS = ["relu","sigmoid","hardsigmoid","swish","elu","gelu",
       "layer_norm","group_norm","gather","scatter","cumsum","lstm"]

for op in OPS:
    with open(f"{ROOT}/ako_runs/{op}/solution/{op}.py") as f:
        code = f.read()
    cheated, val = detect_python_kernel_cheating(code)
    if not cheated:
        print(f"{op:12s} OK")
    else:
        rt = val.get("regression_type")
        print(f"{op:12s} *** FLAGGED (regression_type={rt}) ***")
        for chk, d in val["checks"].items():
            for v in d.get("violations", []):
                print(f"             - {v['function']} L{v['line']}: {v['call']}  [{v['reason']}]")
            if d.get("error"):
                print(f"             ! {d['error']}")
