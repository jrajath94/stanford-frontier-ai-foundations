# U16 interview key

## B1-B6

B1. sigmoid(r_w - r_l).
B2. Keeps the policy near the reference, stops reward hacking.
B3. min(r*A, clip(r, 1-eps, 1+eps)*A).
B4. Group mean/std as the baseline.
B5. Math, code, games: checkable answers.
B6. Proxy up, truth down, a held-out truth metric reveals it.

## L1

L1.1 r = pi_new/pi_old.
L1.2 Terms [0.7, 0.9, -1.0, 1.2, 1.2].
L1.3 The min is pessimistic: trims upside, keeps downside.
L1.4 `ppo_obj`, r=1 returns A.
L1.5 Vanilla PG unbiased but unstable, debug: noisy advantages.
critique: bias, experiment: eps sweep.

## L2

L2.1 r = beta*log(pi/pi_ref) inside the BT loss.
L2.2 Margin 0.139, loss 0.626.
L2.3 Invert the KL-constrained optimum.
L2.4 `dpo_loss`, equal policies give 0.693.
L2.5 PPO explores, debug: length bias, critique: no
exploration, experiment: bias injection.

## E1

(a) 0.0253 nats. (b) 0.00253. (c) The optimal drift shrinks.
more leash means less movement per unit reward. Rubric: (a) 1
pt, (b) 1 pt, (c) 2 pts. Red flag: treating KL as symmetric.

## E2

(a) [1,-1,1,-1] and [0,0,0,0]. (b) The first: nonzero
advantages. (c) Filter prompts to the learnable band, drop
trivial and impossible prompts. Rubric: (a) 2 pts, (b) 1 pt,
(c) 1 pt.

## D1

Bug: r.std() is 0 on uniform groups -> NaN advantages. Fix:
divide by (std + eps), or skip groups with std below eps.
Rubric: find 2 pts, fix 2 pts.

## S1

(1) Halt training. (2) Roll back to the gap's knee. (3) Audit the
proxy's blind spots. (4) Strengthen the truth metric, do not
resume on the proxy alone.

## S2

Discard or reweight heavily: 5 versions stale breaks the PPO
trust region at KL budget 0.01, training on it is not PPO
anymore.

## R1

Gaps: (1) no KL: drift unknown, report it. (2) no held-out eval:
the win may be memorized, gate on held-out. (3) no truth metric:
hacking invisible, add one.
