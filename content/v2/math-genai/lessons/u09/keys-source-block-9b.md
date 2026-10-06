# Answer keys, source block 09b (W10 titles, title-level)

Date: 2026-10-06. Ground truth: lesson-09 and compute_run5b.py.
No transcript inspected, all answers stay at the title boundary.

## Q1

W10L40 -> C05, C06. W10L41 -> C07, C08. W10L42 ->
C08. W10L43 -> C08. W10L44 -> C08 (bridges
P11/R21). W10L45 -> C08. W10L46 -> C06, C09,
C10, C11. Ambiguous: W10L43 (how much of the
architecture?), W10L46 (does it cover the KV
cache?).

## Q2

Skip connections: carry the input past a block
so gradients have a direct path (job: trainable
depth). Normalization: stabilize activations
across the batch/positions (job: stable
optimization). Prepared by: C08 states them,
R20-R22 (P11 bridge) carry the gradient story.

## Q3

(1) Parallelism: training does all positions at
once via the mask, inference generates
sequentially. (2) The mask's role: it makes
training honest, at inference each step only
has the past anyway. (3) Cost: training pays
O(n^2) once per batch, inference pays per
token with the KV cache (U10-C02).

## Q4

Shared idea: an integer (timestep t, position i)
becomes a vector the net can add to its
activations. Difference: the timestep embedding
marks noise level (the task changes with t),
the position embedding marks order (content is
identical, position differs).

## Q5

(1) Scores QK^T/sqrt(d): (n, n). (2) Weights
after masked softmax: (n, n), rows sum to 1,
upper triangle exactly 0.

## Q6

Chain rule: p(x_1..x_n) = prod p(x_i |
x_<i). Toy NLL 1.73697 bits. Watching would
add: the lecturer's motivation, the bigram-to-
neural bridge, worked examples, and whether
teacher forcing is named. None of that follows
from the title.

## R1

Seven titles prove seven scheduled topics in
Week 10. They prove nothing about definitions,
derivations, code, or depth.

## R2

Prediction: embedding -> masked attention ->
residual + norm -> feedforward -> residual +
norm, repeated L times, then linear to vocab +
softmax.

## R3

The lesson computes: embeddings, QKV with
identity, scores, masked softmax, output
rows. It only states: residual, norm,
feedforward, stacking, the final linear. The
stated parts are flagged as such in C08.

## R4

W10L42 promises the transformer used for AR
(the application). W10L43 promises the
architecture itself (the blocks). The second
should add residuals, norms, feedforward, and
depth, the first should add the mask-to-chain-
rule connection.

## R5

Attack: G2 (no transcript inspected) means
"implements" is unmeasured, G6 (repo contents
uninspected) means the notebook
IITM_DGM_MHA.ipynb is a name, not evidence.
Titles schedule topics, they do not prove
implementations were shown or run.
