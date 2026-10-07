# U05 interview key

## B1-B6

B1. exp overflows on large logits, the loss is nan before log can
rescue it. The max-subtracted form is algebraically identical and
safe.
B2. inputs = ids[:, :-1], labels = ids[:, 1:], position t predicts
token t+1.
B3. m: exponential moving average of gradients (direction). v: moving
average of squared gradients (per-parameter scale).
B4. AdamW subtracts lam*w outside the adaptive scaling, Adam+L2 adds
lam*w to the gradient, so the scaling also scales the decay.
B5. It caps the update norm at lr*c and preserves direction. It does
not fix the cause of spikes, constant clipping hides instability.
B6. Weights, optimizer moments, step, scheduler state, RNG states,
data-loader position.

## L1

L1.1 m = b1 m + (1-b1) g, v = b2 v + (1-b2) g^2, divide by (1-b^t).
L1.2 w -> [0.99899, -0.498995].
L1.3 E[m_t] = (1-b^t)E[g] from the geometric series, division
unbiases. At t=1 the factors are 10.0 and 1000.0.
L1.4 The five-line step, check mhat == g and vhat == g^2 at t=1.
L1.5 Compare: matched-step toy, AdamW is less lr-sensitive.
Debug: Lion needs ~10-100x smaller lr, retune. Critique: gradients
are biased by dropout/data order in practice. Experiment: sweep lr
for both, plot steps to target loss.

## L2

L2.1 Warmup: 0 -> peak linear. Cosine: half-cosine to floor. WSD:
stable then fast decay.
L2.2 0, 1.5e-4, 3e-4, 1.65e-4, 3e-5.
L2.3 The cosine never reaches the floor, the final lr is too high
and the model under-converges.
L2.4 Both functions, checks lr(0)=0, lr(warm)=peak, lr(total)=
floor, WSD stable-phase check lr(500)=peak.
L2.5 WSD degrades less under early stop (decay is at the end).
Debug: recompute the schedule for the actual stop. Critique: total
is unknowable on preemptible clusters. Experiment: early-stop both
at half, compare final loss.

## E1

(a) 2M/8k = 250 steps. (b) Effective lr = 250x = 7.5e-2: immediate
divergence. (c) Checklist: loss inputs finite, data batch sane,
lr/schedule changed, recent code change, clip frequency, mixed-
precision overflow, bisect. Rubric: (a) 1 pt, (b) 1 pt, (c) 2 pts.

## E2

Loss jump: data order and dropout masks change at once, so the batch
statistics shift. Silent divergence: the trajectory differs from the
uninterrupted run and the comparison baseline is invalid. Fix: save
and restore the RNG states (all of them). Red flag: "just lower the
lr after resume".

## D1

Bug: `total` is the squared norm (sum of squares), not the norm, and
it is compared against c and divided without a square root. The
correct scale is c/sqrt(total) when sqrt(total) > c. As written, the
function clips far too aggressively (comparing a squared quantity to
c) and the scale is wrong dimensionally. Fix: n = sqrt(total), if
n > c: scale = c/n. Rubric: find 2 pts, fix 1 pt, explain the silent
dynamics change 1 pt.

## S1

Gradient accumulation with a=8. Guarantee: algebraic identity with
the big batch for mean losses. Breaks: (1) loss summed not averaged:
must divide grads by a, (2) per-microbatch dropout masks make the
equivalence in-expectation only. Red flag: "batchnorm statistics".

## S2

Pick WSD: the decay happens at the end over the last 10-20 percent,
so an extension just lengthens the stable phase and an early stop
only skips the decay. Cosine needs the total up front, a wrong total
wastes the run. Red flag: "constant lr, it does not matter".

## R1

Gaps: (1) single run: no variance estimate, need seeds. (2) no lr
sweep: the comparison may be a tuning artifact, need matched tuning
budgets. (3) wall-clock only: conflates step efficiency with
implementation speed, need steps-to-target and tokens-to-target.
