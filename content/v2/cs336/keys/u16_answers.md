# U16 answer key

All numeric claims from `../visuals/compute_u16.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

KL(p||q) = 0.9*ln(0.9/0.5) + 0.1*ln(0.1/0.5) = 0.368 nats.
KL(q||p) = 0.5*ln(0.5/0.9) + 0.5*ln(0.5/0.1) = 0.511 nats.

## A1

(a) P(prefer A) = sigmoid(r_A - r_B). (b) 0.731. (c) Rank pairs
(A,B), (B,C), (A,C): transitivity gives all three.

## A2

(a) (prompt, chosen, rejected). (b) Teaches clarity for that
prompt. (c) It teaches length: filter by quality first or the
reward model learns verbosity.

## A3

(a) E[r] - beta*KL(pi||ref). (b) Penalty 0.00253 at beta 0.1.
(c) The policy exploits the reward without bound: reward hacking
(C11), add the KL term back.

## A4

(a) min(rA, clip(r)A). (b) [0.7, 0.9, -1.0, 1.2, 1.2]. (c) Too
loose for 70B: large updates risk collapse, use 0.2 or lower with
monitoring.

## A5

(a) Policy acts, value predicts, advantage = r - V credits the
action with lower variance. (b) A = 0.3. (c) Debug the value
training (targets, LR, normalization), fall back to group
baselines (C07) if it will not learn.

## A6

(a) beta*log(pi/ref) margin in a logistic loss, assumes BT
preferences, the KL-constrained optimum form, iid pairs. (b)
0.139/0.626, flipped 0.765. (c) PPO: DPO cannot explore beyond the
dataset, novel reasoning needs online RL.

## A7

(a) (r-mean)/std over the group. (b) [1,-1,1,-1], zeros for the
all-correct group. (c) Filter prompts to the learnable band
(neither trivial nor impossible), keep G=8.

## A8

(a) Programmatic checkers, strict but honest. (b) Mean 0.67, the
"forty-two" zero shows strictness. (c) Unit tests plus hidden
tests, reward = fraction passing.

## A9

(a) Sample, generate, score, update. (b) 4096 slot-seconds per
batch on the toy. (c) Add rollout workers (U17 separation) or cut
G/length, the trainer must not idle.

## A10

(a) KL from the rollout policy, halt past the budget. (b) Max
0.006 nats: healthy. (c) 0.02 is 4x the toy budget: stop the
epochs, resample fresh rollouts.

## A11

(a) Proxy rises, truth falls, monitor the gap. (b) Gap 0.24 at
step 100, opens at step 40. (c) Halt: truth flat + proxy up =
hacking signature, roll back to the gap's knee.

## A12

(a) Safety and reasoning slices the model never saw, ship only on
all bars. (b) Refusal 0.98, reasoning flat, gap < 0.05: ship.
(c) No: gap 0.08 exceeds the 0.05 bar, investigate before
shipping.
