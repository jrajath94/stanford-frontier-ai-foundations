# Interview bank, U13 foundation models and representations

Date: 2026-10-06. Questions only. Keys in interview/keys-u13.md.
Closed-book. Do not read the keys first.

## Breadth (6)

B1. Write the pretraining loss and name the two phases of the
foundation-model paradigm.
B2. State linear probing as an optimization problem and say what
is frozen.
B3. Write the LoRA update, its scaling, and its trainable
parameter count.
B4. In contrastive learning, what defines a positive pair and a
negative pair without labels?
B5. Write the SIMCLR loss and state what happens to it when a
positive inner product grows.
B6. Define Recall@k and NDCG@k, and state which one ignores
ranking.

## Deep ladder D1, adaptation (5 follow-ups)

D1.1. Write the probing objective (15.2) and the finetuning
objective (15.3). State the initialization of each.
D1.2. Toy: phi(x_i) = [i, 1] for i = 1..6, y = [1..6]. Solve the
probe in closed form and report the train MSE.
D1.3. Derive: why does the linear probe have a closed-form
solution while finetuning does not?
D1.4. Implement/debug: a teammate "probes" but the backbone
weights change every epoch and the result beats the frozen
baseline. Name the bug and explain why the comparison is void.
D1.5. Changed constraint: the downstream task has 40,000 labels
and is far from the pretraining distribution. Rank probe,
finetune, LP-FT, and LoRA (r = 8) for this regime and justify
the order.

## Deep ladder D2, contrastive and retrieval (5 follow-ups)

D2.1. Define the SIMCLR loss on a batch of size B and name the
role of each term.
D2.2. Toy: B = 4, d = 6, seed 9, embeddings built as in the
lesson. Report the loss and the effect of moving positives 10%
closer.
D2.3. Derive or justify: why does minimizing the loss pull
positives together and push random pairs apart? Use the scalar
monotonicity fact.
D2.4. Implement/debug: after 10 epochs every embedding in the
batch is the same vector and the loss is flat. Diagnose the
collapse and name two fixes.
D2.5. Research critique: "Our embedding model scores higher on
the benchmark suite, so it is a better model for our legal
document search." Attack the claim using SL-06: name the
limitation and design the one evaluation that decides the
question.

## Analytical/quantitative (2)

Q1. Prove that -log(p / (p + q)) decreases in p and increases in
q for p, q > 0.
Q2. A query has gold documents {d2, d4, d5} with grades 3, 2, 1.
The system returns [d1, d2, d3, d4, d5]. Compute Recall@3,
DCG@5, IDCG@5, and NDCG@5.

## Implementation/debug (1)

T1. This code intends the SIMCLR loss for a batch:

```python
import numpy as np
def simclr(Zhat, Ztilde):
    B = Zhat.shape[0]
    S = Zhat @ Ztilde.T
    total = 0.0
    for i in range(B):
        pos = np.exp(S[i, i])
        denom = np.exp(S[i, :]).sum()
        total += -np.log(pos / denom)
    return total
```

Two bugs relative to the notes' variant: one in the denominator
(which pairs it sums over) and one missing preprocessing step on
the embeddings. Identify both, fix them, and state the numeric
check that proves the fixed loss has the right monotonicity.

## Changed-constraint scenarios (2)

S1. Your corpus has 2 billion documents and queries must return
in under 50 ms. Brute-force O(N m) search is impossible. Name
the index family you would use, the approximation it makes, and
the failure mode you must monitor.
S2. Your pretraining data are paired image-text examples, not
single images. Positives are no longer "two augmentations of one
image". Redefine the positive and negative pairs for this data
and state which part of the SIMCLR argument survives unchanged.

## Research critique (1)

R1. "We replaced full finetuning with LoRA (r = 8) on all 40 of
our tasks and saw no accuracy drop, so LoRA is strictly better
than finetuning." Critique: name the hidden condition that makes
this true, describe the regime where it fails, and design the
experiment that finds the boundary.
