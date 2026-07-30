# Fused frontier closure v3

Append-only same-GPU closure of the fused reachability mechanism. The frozen
design is `PREREGISTRATION.md` plus `campaign.json`.

Expected sequence after committed receipts exist:

```bash
python -m unittest ako_runs.controlled_followup.fused_frontier_closure_v3.test_frontier
python ako_runs/controlled_followup/fused_frontier_closure_v3/provenance.py --check
python ako_runs/controlled_followup/fused_frontier_closure_v3/eligibility.py --check
python ako_runs/controlled_followup/fused_frontier_closure_v3/launch.py --gpu 3 --tag performance_v1 --dry-run
python ako_runs/controlled_followup/fused_frontier_closure_v3/launch.py --gpu 3 --tag performance_v1
python ako_runs/controlled_followup/fused_frontier_closure_v3/analyze.py --tag performance_v1
python ako_runs/controlled_followup/fused_frontier_closure_v3/capture_evidence.py build --name complete_v1 --tag performance_v1
python ako_runs/controlled_followup/fused_frontier_closure_v3/capture_evidence.py verify --index ako_runs/controlled_followup/fused_frontier_closure_v3/evidence/complete_v1.index.json
```
