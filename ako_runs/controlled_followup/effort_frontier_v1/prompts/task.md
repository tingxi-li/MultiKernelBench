Optimize the assigned lane for the fixed M=1024, K=8192, N=8192 fused
matmul+bias+GELU+softmax operation. Inputs are pre-cast outside the timed region
and output is fp32. A candidate is eligible only after every frozen fused-v2
tuning case passes. Minimize confirmed kernel latency within the cumulative
active-effort checkpoints at 0.5, 2, and 8 hours. Treat evaluator errors as data;
do not repair the test harness or infer hidden holdout values.
