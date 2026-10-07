# U16 interview bank , questions

Closed-book. Keys in `u16_key.md`. All numbers are synthetic toys
from `visuals/compute_u16.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Write the Bradley-Terry preference probability.
B2. What is a KL penalty for in RLHF?
B3. Write the PPO clipped objective.
B4. How does GRPO get a baseline without a value network?
B5. When do verifiable rewards beat learned rewards?
B6. What is reward hacking, and what reveals it?

## Deep ladders (2 x 5)

L1. PPO clipping.
- L1.1 Define the ratio.
- L1.2 Toy: ratios [0.7..1.6], clipped terms computed.
- L1.3 Derive min(rA, clip(r)A) and the pessimism.
- L1.4 Implement `ppo_obj`, state the r=1 check.
- L1.5 Compare with vanilla PG, debug noisy advantages,
  critique the bias, propose the eps sweep.

L2. DPO.
- L2.1 Define the reparameterization.
- L2.2 Toy: margin 0.139, loss 0.626.
- L2.3 Derive r = beta*log(pi/ref) from the KL-constrained
  optimum.
- L2.4 Implement `dpo_loss`, state the equal-policy check.
- L2.5 Compare with PPO, debug length bias, critique no-
  exploration, propose the bias-injection test.

## Analytical exercises (2)

E1. p=[0.5,0.3,0.2], q=[0.4,0.4,0.2], beta=0.1. (a) Compute
KL(p||q). (b) The penalty in the objective. (c) Double beta:
what changes in the optimal drift?
E2. Group rewards [1,0,1,0] vs [1,1,1,1]. (a) Compute GRPO
advantages for both. (b) Which group teaches, and why? (c) The
pipeline is full of all-correct groups. Fix it.

## Implementation/debug task (1)

D1. This advantage normalization divides by zero. Find the bug
and fix it.

```
adv = (r - r.mean()) / r.std()  # bug: std=0 on uniform groups
```

## Changed-constraint scenarios (2)

S1. The reward model and the policy co-adapt, the truth metric is
flat while the proxy climbs. List the actions in order.
S2. Rollouts arrive 5 versions stale in an async system. The KL
budget is 0.01. Decide: train, reweight, or discard?

## Research critique (1)

R1. An RLHF paper reports reward gains with no KL numbers, no
held-out eval, and no truth metric. List three gaps and the
measurement for each.
