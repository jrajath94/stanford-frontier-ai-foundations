# Lesson 09b, Eighth source block: auto-regressive models and inference

Unit: math-genai-U09. Date: 2026-10-06. Baseline: October 6, 2026.

## Source mapping

Title-level treatment of seven playlist entries (SRC-04):
W10L40 Auto-Regressive Models, W10L41 Attention
Mechanism, W10L42 Transformers for Auto-Regressive
Models, W10L43 Transformers architecture, W10L44
Transformers: Skip Connections and Normalization,
W10L45 Transformers: Position Embeddings, W10L46
Transformers: Training and Inference. No transcript
inspected (source_gaps.md G2). Each section states
what the title promises, the question it answers,
where it lands in the lesson, and what to verify
when watching. Nothing below claims lecture content.

## Scope and objectives

Scope: the seven titles above, mapped to U09 leaf
concepts C05-C12. Objectives: after this block the
learner can say what each title should teach, point
to the lesson section that prepares them for it,
and list the verification questions to ask while
watching.

## SB01, W10L40 Auto-Regressive Models

The title promises: the AR factorization as a
generative model class. The question: how does
"predict next" become a sampler? Lands in: C05
(chain rule) and C06 (teacher forcing). Preparation:
the bigram NLL 1.73697 bits, perplexity 1.82574,
and the exposure gap 0.7653 bits/token. Verify when
watching: whether the lecture states the chain
rule, names teacher forcing, and mentions the
train/test mismatch.

## SB02, W10L41 Attention Mechanism

The title promises: attention as a mechanism. The
question: how does a position read its past? Lands
in: C07 (causal mask) and C08 (transformer).
Preparation: the 3x3 masked weights ([1,0,0],
[0.3302,0.6698,0], [0.2483,0.2483,0.5035]).
Verify when watching: whether queries, keys, and
values are defined, whether the 1/sqrt(d) scale
is justified, and whether the mask is shown.

## SB03, W10L42 Transformers for Auto-Regressive Models

The title promises: the transformer applied to AR
modelling. The question: what makes the
transformer the AR workhorse? Lands in: C08.
Preparation: the full toy forward pass (O rows
computed, O[0] = X[0] by the mask). Verify when
watching: whether the lecture connects the mask
to the chain rule (each row is one conditional),
and whether multi-head attention is introduced.

## SB04, W10L43 Transformers architecture

The title promises: the architecture around
attention. The question: what else is in the
block? Lands in: C08 (stated, not computed:
residual, norm, feedforward, stacked layers).
Preparation: the attention core from SB02/SB03.
Verify when watching: the block diagram (residual
paths, norm placement, feedforward), the depth
(number of layers), and the final linear +
softmax to vocab.

## SB05, W10L44 Transformers: Skip Connections and Normalization

The title promises: the two architectural details
that make depth trainable. The question: why do
residuals and norms matter? Lands in: C08
(bridges to P11/R20-R22: gradients through depth).
Preparation: R21 (a neuron, backprop) from the
prerequisites. Verify when watching: whether the
lecture shows the gradient path through a
residual, and where the norm sits (pre vs post).

## SB06, W10L45 Transformers: Position Embeddings

The title promises: how order enters the model.
The question: what breaks without positions? Lands
in: C08 (the P matrix, the 0.0983 difference).
Preparation: the with-pos vs no-pos comparison
and the U08-C05 timestep embedding (the same
idea: an integer becomes a vector). Verify when
watching: sinusoidal vs learned, and whether the
lecture demonstrates order-sensitivity.

## SB07, W10L46 Transformers: Training and Inference

The title promises: how transformers train and how
they generate. The question: what changes between
the two phases? Lands in: C06 (teacher forcing),
C09 (likelihood), C10 (sampling), C11 (budgets).
Preparation: parallel training via the mask vs
sequential generation, the perplexity number,
the sampling distributions, the 0.60 GB at n =
1024. Verify when watching: whether the lecture
names the KV cache (U10-C02), states the O(n^2)
cost, and distinguishes training loss from
generation quality.

## Source-block exercises (questions. Answers in keys-source-block-9b.md)

Q1. Map each of the seven titles to its U09 leaf
concept(s). Flag any title whose mapping is
ambiguous from the title alone.
Q2. W10L44 names two mechanisms. State the
one-line job of each and the lesson section that
prepares the learner.
Q3. W10L46 spans training and inference. List
three things that differ between the phases on
the toy (parallelism, the mask's role, cost).
Q4. SB06 and U08-C05 both embed an integer.
State the shared idea and one difference
(timestep vs position).
Q5. For SB02, write the two shape asserts you
would want the lecture to show (scores, weights).
Q6. For SB01, state the chain rule and the toy
NLL. What would watching add beyond the title?

## Deep oral ladder (questions. Answers in keys-source-block-9b.md)

R1. Define the title boundary again for this
block: what do seven titles prove?
R2. Take W10L43: predict the block diagram from
the title alone.
R3. Check the prediction against C08: which
parts does the lesson compute, and which does
it only state?
R4. Compare W10L42 and W10L43: what does the
second add that the first does not promise?
R5. Critique: "the transformer titles prove
the course implements transformers." Attack it
with G2 and G6.
