# Lab 10: Retrieval systems and project synthesis

Unit: cs229s-U10. Date: 2026-10-06.
Run: `python3 verify_lab10.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-10.md`.

## Objective

Verify U10 mechanisms on CPU with numpy: index
amortization and break-even, dense vs PQ storage, the ANN
floor selection, RRF fusion scores, cascade costs, the
cluster bill with evidence grades, the claim-grading
function, proposal scoring, the profiling-hypothesis
survival rule, seed mean and spread, the quality gate
boundary, and the defense rubric total.

## Setup

```bash
python3 verify_lab10.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Amortized cost and break-even.
2. Storage bytes and ratio.
3. ANN floor selection.
4. RRF scores.
5. Cascade costs.
6. Cluster bill.
7. Claim grading.
8. Proposal scoring.
9. Hypothesis survival.
10. Seed statistics.
11. Quality gate.
12. Defense total.

## Replication proposal (PROPOSED, not executed)

Question: does the ANN recall/latency curve have a knee
in funnel width, and does the quality gate catch a known-
bad optimization? Hypothesis: the knee exists and the
gate catches the bad candidate. Method: build a toy ANN
on synthetic vectors, sweep widths, then run the gate.
Budget: CPU hours. Failure criterion: no knee or a gate
miss (then the toy is too easy, not the method wrong).
