# Lab 07: Linear attention, SSMs, and FFT

Unit: cs229s-U07. Date: 2026-10-06.
Run: `python3 verify_lab07.py` in this directory. All checks
ran on this machine. Expected outputs are recorded in
`keys-lab-07.md`.

## Objective

Verify U07 mechanisms on CPU with numpy: recurrence vs
convolution on the toy, the quadratic FLOP and memory
counts with the 4x law, the linear-attention regrouping
identity and the 64x ratio, the SSM recurrence and pulse
decay, the convolution-view duality, the scan-tree total,
the FFT convolution theorem and the 33.0x ratio, the
stability numbers and fp16 death step, the causal leak
values, the copy-capacity shortfall, the traffic byte
counts, and the three-model workload table.

## Setup

```bash
python3 verify_lab07.py
```

No GPU needed. No network needed. Runtime is seconds.

## Tasks

1. Recurrence and convolution on u = [1, 0, 1, 0].
2. Quadratic counts and the doubling law.
3. Regrouping identity and 64x FLOP ratio.
4. SSM states and pulse decay.
5. Convolution-view duality.
6. Scan tree total.
7. FFT theorem and ratio.
8. Stability values and death step.
9. Causal vs centered leak.
10. Copy capacity shortfall.
11. Traffic bytes and ratio.
12. Workload table.

## Replication proposal (PROPOSED, not executed)

Question: does the recurrence/convolution/scan triple
identity hold on a random diagonal SSM, and where does
the FFT conv-view beat the sequential loop? Hypothesis:
all three agree to 1e-8, and conv-view wins past a
measured crossover T. Method: random diagonal A, compare
outputs, time both paths across T. Budget: CPU hours.
Failure criterion: any path disagrees (then the duality
is misimplemented, not wrong).
