# Lab 06: Scaling laws, adaptation, and data pipelines

Unit: cs229s-U06. Date: 2026-10-06.
Run: `python3 verify_lab06.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-06.md`.

## Objective

Verify U06 mechanisms on CPU with numpy: the two-point
scaling-law fit and its self-consistency, the k-shot
saturation deltas, the threshold-metric crossing, the masked
SFT loss, the Bradley-Terry probability and KL objective,
the constitutional score margins, the LoRA merge identity
and param ratio, the overlapped pipeline step time, the
preprocessing yield product, greedy packing efficiency, the
throughput and cost arithmetic, and the seed-sweep mean and
spread.

## Setup

```bash
python3 verify_lab06.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Fit the scaling law on two points, predict the third.
2. K-shot deltas shrink with k.
3. Threshold metric jumps while skill moves 0.15.
4. Masked SFT loss on the toy log-probs.
5. Bradley-Terry and KL objective values.
6. Constitutional margins under two weight settings.
7. LoRA merge identity and 128x param ratio.
8. Pipeline step time in three regimes.
9. Funnel yield product.
10. Greedy packing efficiency.
11. Throughput and cost per million tokens.
12. Seed sweep mean, std, and the tie call.

## Replication proposal (PROPOSED, not executed)

Question: does a threshold metric manufacture an apparent
emergence jump on a synthetic task where the true skill is a
smooth ramp? Hypothesis: the 0/1 metric jumps a full 1.0
while the continuous score moves 0.15, matching the C03
toy. Method: build the synthetic task, sweep scale over
four decades, record both metrics. Budget: CPU hours only.
Failure criterion: the metric and the score move together
(then the artifact story is wrong for this task).
