# U04 interview bank , questions

Closed-book. Keys in `u04_key.md`. Quotas: 6 breadth, 2 deep ladders of
5, 2 analytical, 1 implementation/debug, 2 changed-constraint, 1
research-critique.

## Breadth (6)

B1. How does the KV cache scale with the number of KV heads, and what
are the MHA, GQA, and MQA choices?
B2. What is the FLOP cost of windowed attention versus full attention?
B3. How does linear attention turn O(T^2) into O(T)?
B4. In MoE, what is the difference between total and active parameters?
B5. What is a dropped token, and what controls the drop rate?
B6. What does the load-balancing aux loss penalize?

## Deep ladders (2 x 5)

L1. KV cache compression.
- L1.1 Define the per-token per-layer cache bytes.
- L1.2 Toy: compute the MHA and MQA cache for B=2, T=1024, d=512,
  L=12, bf16.
- L1.3 Derive why GQA barely changes FLOPs but divides cache bytes.
- L1.4 Implement GQA attention via head repeat, state the g=h check.
- L1.5 Compare MQA, GQA, MLA at equal cache bytes, debug the
  averaged-heads conversion, critique the fixed-dh assumption, propose
  the fixed-FLOP quality comparison.

L2. MoE routing and balance.
- L2.1 Define router scores, top-k, capacity, dropped tokens.
- L2.2 Toy: 16 tokens, E=8, k=2, factor 1.25. Compute capacity and
  explain the drops.
- L2.3 Derive the aux loss and its uniform value.
- L2.4 Implement dispatch with capacity, state the conservation
  invariant.
- L2.5 Compare token-choice with expert-choice, debug router collapse,
  critique the drop-is-fine assumption, propose the alpha sweep.

## Analytical exercises (2)

E1. A model has d=2048, 32 query heads, GQA with 8 KV heads, L=32,
bf16, serving T=8192, B=16. Compute the KV cache in GB. Then compute
it for MQA and state the savings.
E2. An MoE has E=64 experts, k=2, d=4096, dff=14336 (SwiGLU). Compute
total FFN params, active FFN params per token, and the ratio. If the
router collapsed to 4 experts, what is the effective active count?

## Implementation/debug task (1)

D1. This GQA implementation gives wrong outputs when g < h but works
when g = h. Find the bug and fix it.

```
def gqa_buggy(Q, K, V, g):
    # Q: (B,h,T,dh), K,V: (B,g,T,dh)
    K = np.tile(K, (1, h // g, 1, 1))   # <-- suspect
    V = np.tile(V, (1, h // g, 1, 1))
    ...standard attention...
```

## Changed-constraint scenarios (2)

S1. Serving requires 128k context on fixed HBM. Rank these by cache
bytes: MHA, GQA-8, MQA, MLA-128. Then name the quality risk of the
smallest.
S2. Training shows 40 percent of tokens dropped at factor 1.0. Name two
fixes, the mechanism of each, and which you try first.

## Research critique (1)

R1. A paper claims MoE "beats dense at equal parameters" but compares
a 64-expert MoE's total params against a dense model's params with no
active-FLOP numbers and no load statistics. List three gaps and the
measurement for each.
