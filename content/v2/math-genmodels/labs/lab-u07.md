# Lab U07: autoregressive models

Unit: math-genmodels-U07. Date: 2026-10-06. Baseline: October 6, 2026.

Stack: Python 3, numpy 1.26.4, CPU only, float64. Seed every run and
print the seed. Answers in labs/keys/keys-u07.md. Do not open the keys
until the code runs.

## E1: chain-rule audit

Define the toy tables: p(a)=0.5, p(b|a)=0.4, p(c|a,b)=0.25, plus
full rows so every conditional sums to 1. Enumerate all 27
sequences, compute each joint by the chain rule, and confirm the
total is 1. Report p(abc), log p(abc), NLL, perplexity.

## E2: attention by hand

Q = K = I_2, V = [[2,3],[4,5]], causal mask. Compute scores,
masked softmax, and output with numpy. Confirm row 2 weights
(0.3302, 0.6698) and output (3.3395, 4.3395). Then remove the
1/sqrt(2) scale, multiply Q by 10, and report what happens to
the weights. Write the one-sentence lesson.

## E3: temperature sweep

Logits (1,2,3). Compute softmax at T in (0.1, 0.5, 1, 2, 10).
Report the five distributions. Confirm T=0.1 is near-argmax and
T=10 is near-uniform. Sample 2,000 tokens at T=2 (seed 12) and
compare empirical frequencies with the distribution.

## E4: cost model

For L in (128, 512, 2048, 8192), d = 64: compute attention
mults and KV-cache bytes (fp32) per layer. Confirm quadratic
scaling in L. State the L where the scores matrix alone exceeds
1 GB.

## E5: flow-matching loss

x_0 = 0, x_1 = 2, t = 0.25. Compute x_t, the target velocity,
and the loss for predictions v in (2.0, 1.8, 1.0). Confirm
0.0, 0.04, 1.0. Verify the endpoints: x_t at t=0 is x_0, at
t=1 is x_1.

## E6: masked versus autoregressive

Build the full toy joint from the tables in E1 (p3 rows: (a,b)
-> (0.5, 0.25, 0.25), all others -> (0.34, 0.33, 0.33)).
Compute the masked pseudo-likelihood PL(x) = p(x1|x2,x3) x
p(x2|x1,x3) x p(x3|x1,x2) for each of the 27 sequences. Sum it
over all sequences and report the total. State whether it
equals 1 and what that proves.
