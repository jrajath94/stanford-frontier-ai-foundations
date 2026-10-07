# U05 answer key

All numeric claims come from `../visuals/compute_u05.py` or the lab run
(executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

logits [2,1,0], true class 0. m = 2. lse = 2 + log(1 + e^-1 + e^-2).
CE = lse - 2 = log(1 + 0.3679 + 0.1353) = log(1.5032) = 0.4076.

## A1

(a) CE = log(sum(exp(x))) - x_true, computed as m + log(sum(exp(x -
m))) - x_true with m = max(x). (b) Key beats: max subtraction keeps
exp args <= 0, hand demo matches the lab's 0.4076, naive overflows to
nan on [1000,1001,999], the form is algebraically exact. (c) With
V=1M the full logits cost O(B*T*V) memory: the lesson's sampled-
softmax branch or chunked loss, stability unchanged.

## A2

(a) inputs = ids[:, :-1], labels = ids[:, 1:]. (b) [5,6,7,8] gives
inputs [5,6,7], labels [6,7,8], reversed shift makes the loss collapse
to near zero while generation babbles. (c) Packed documents need a
loss mask at boundaries so positions do not predict across documents.

## A3

(a) m_t = 0.9 m_{t-1} + 0.1 g_t, v_t = 0.999 v_{t-1} + 0.001 g_t^2,
w -= lr (mhat/(sqrt(vhat)+eps) + lam w). (b) The hand step gives
[0.99899, -0.498995], t=1 gives mhat = g, vhat = g^2, eps guards the
zero-v division. (c) Sparse embedding grads: v for rare rows stays
near init, so bias correction and eps dominate, standard practice
uses the same Adam, sometimes with sparse variants.

## A4

(a) AdamW: w -= lr m(g)/sqrt(v) - lr lam w. Adam+L2: w -= lr
m(g + lam w)/sqrt(v). (b) The toy: decay term 0.0500 sits outside the
scaling in AdamW, L2 folds it into the scaled grad (0.1500 on the toy
numbers). (c) Decay matrices (weights), exclude norms, biases, and
the embedding.

## A5

(a) Divide m_t by (1 - b1^t), v_t by (1 - b2^t). (b) E[m_t] =
(1-b^t)E[g] by the geometric series, t=1 gives exactness, factors on
the toy: 10.0 and 1000.0. (c) At t=10000 the factors are ~1: no
effect, warmup covers part of the need but is not exact.

## A6

(a) g *= min(1, c/norm(g)), c often 1.0. (b) [3,4] norm 5 -> scaled
to norm 1.0000, cosine 1.000000 with the original, no-op below c.
(c) First investigate the data: a bad batch or a numerical event,
check the clip frequency log before touching the threshold.

## A7

(a) Warmup: 0 -> peak linear. Cosine: peak -> floor on a half cosine.
WSD: warmup, constant, fast decay. (b) Toy values: 0, 1.5e-4, 3e-4,
1.65e-4, 3e-5 at steps 0/50/100/550/1000. (c) Pick WSD: set the decay
over the last 10-20 percent of the expected (extended) budget.

## A8

(a) mean of microbatch grads = big-batch grad (algebraic identity for
mean losses). (b) Toy: ([0.2,0.4]+[0.6,-0.2])/2 = [0.4,0.1] exactly,
missing /a doubles the effective lr. (c) Microbatch=1 with dropout:
each microbatch draws a different mask, so the equivalence is only in
expectation, not exact.

## A9

(a) SGD+m: v = 0.9v + g, w -= lr v. Lion: c = 0.9m + 0.1g, w -= lr
sign(c), m = 0.99m + 0.01g. Adafactor: factored second moments.
(b) Toy w=[1.0], g=[0.5], lr=1e-3: SGD+m -> 0.999500, Lion ->
0.999000 (sign step). (c) 4 bytes/param: SGD+m or Lion, give up the
adaptive per-parameter scaling (and the robustness to lr choice).

## A10

(a) muP: hidden init std 1/sqrt(fan_in) with lr mult base_width/width,
output init std 1/width. (b) Table: widths 128/512/2048 give std
0.0884/0.0442/0.0221, hidden lr mult 1.0/0.25/0.0625, output init
0.00781/0.00195/0.00049. (c) Depth scaling needs its own rules (the
muP depth extension), the width rules alone are incomplete.

## A11

(a) Weights, optimizer moments, step, scheduler state, RNG states,
data-loader position. (b) save/restore reproduces the stream exactly
(lab T11). (c) Checkpoint every k steps where k balances write time
against lost work, keep at least two recent full states.

## A12

(a) Central differences (f(w+e)-f(w-e))/2e versus analytic, relative
error < 1e-5, invariants: finite loss, overfit-batch decrease, finite
nonzero grad norm. (b) Lab: 1.88e-10 on the (4,3) layer, fp16
rounding breaks the check. (c) Ordered checklist: loss inputs finite?
data batch sane? lr/schedule changed? recent code change? clip
frequency? mixed precision overflow? Then bisect.
