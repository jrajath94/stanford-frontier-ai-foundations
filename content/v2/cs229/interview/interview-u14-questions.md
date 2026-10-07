# Interview bank, U14 LLM architecture and training

Date: 2026-10-06. Questions only. Keys in interview/keys-u14.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. What is a token, and what does byte-pair encoding do?
B2. Write the autoregressive factorization of p(x_1, ..., x_T).
B3. Write the next-token pretraining loss.
B4. Write single-head self-attention: the score formula and the
output formula.
B5. Why is the attention score divided by sqrt(d_h)?
B6. State the causal mask and the property it guarantees.

## Deep ladder D1, attention mechanics (5 follow-ups)

D1.1. Define single-head attention: the Q/K/V projections, the
scores, the output. Give all shapes.
D1.2. Toy: Q rows [1,0],[0,1],[1,1]. K rows [1,0],[0,1],[1,-1].
Compute the causal masked score matrix.
D1.3. Derive: from the per-position rule to the matrix form
H^out = softmax_row(Q K^T / c) V.
D1.4. Implement/debug: after training, perturbing x_5 changes
the logit u_3. Name the bug class and the two checks that catch
it.
D1.5. Changed constraint: you need bidirectional context for an
embedding model, not generation. What changes in the mask, and
which training loss from the lesson no longer applies?

## Deep ladder D2, costs and variants (5 follow-ups)

D2.1. State the per-head attention compute and KV-cache memory
for training/prefill and for decoding.
D2.2. Toy: d = 4096, n_h = 32, fp16, T = 8192. Report the
per-layer KV cache for MHA, GQA (n_g = 8), and MQA.
D2.3. Derive: why does the GQA cache scale with n_g and not n_h?
D2.4. Implement/debug: a serving system OOMs at 32k context. The
model is GQA with n_g = 8 but the capacity plan used MHA cache
numbers. Quantify the overprovisioning and name the fix.
D2.5. Research critique: "MQA strictly dominates MHA because the
cache is 32x smaller at equal quality." Attack the claim: name
the hidden assumption and the experiment that tests it.

## Analytical/quantitative (2)

Q1. Show that the t-th row of softmax_row(Q K^T / c + M) with
the causal mask equals the score vector of (17.23).
Q2. Logits [2.0, 1.0, 0.5, 0.1]. Compute the softmax at tau =
0.5, 1.0, 2.0. At tau = 1.0 give the top-p 0.9 kept set and its
renormalized distribution, and the top-k k = 2 renormalized
distribution.

## Implementation/debug (1)

T1. This code intends one decoding step with a KV cache:

```python
import numpy as np
# cache holds K_all (t, dh), V_all (t, dh) for one head
q = h @ Wq                      # (dh,)
scores = q @ K_all.T / np.sqrt(dh)
probs = np.exp(scores) / np.exp(scores).sum()
out = probs @ V_all
k_new = h @ Wk
v_new = h @ Wv
K_all = np.concatenate([K_all, k_new[None, :]], axis=0)
V_all = np.concatenate([V_all, v_new[None, :]], axis=0)
```

Two bugs: one makes every generated token ignore its own new
key/value in a way that compounds, and one is a shape bug that
appears only for GQA models. Identify both, fix them, and state
the check that proves the cached step matches uncached
attention.

## Changed-constraint scenarios (2)

S1. You must serve 128k context with a fixed 40 GB memory
budget per GPU. The model is dense MHA. Name the two cheapest
changes (one architectural, one systems) that cut the cache
term, and state what each one sacrifices.
S2. Your SFT data has very long prompts and short answers.
The loss mask is accidentally inverted (1 on prompt tokens, 0
on answer tokens). Describe the two observable symptoms during
training and the one-line fix.

## Research critique (1)

R1. "Our model beats the baseline on zero-shot benchmarks, so
it is the better pretrained model." Critique using SL-08: name
the confound (prompt wording), and design the evaluation
protocol that separates pretraining quality from prompt
engineering.
