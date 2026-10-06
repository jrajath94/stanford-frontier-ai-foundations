---
page_id: cs229-l14
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 14
nav: "L14 · Transformers"
title: "Lecture 14: Transformers and Autoregressive Language Models"
summary: "Tokenization, the autoregressive paradigm, attention from Q/K/V, causal masking, and the quadratic bottleneck."
date: "2026-05-20"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "1:17:22"
video_id: pwQ0l4hFCVI
video_title: "Lecture 14: Transformers"
video_caption: "Original lecture. Tengyu Ma builds the transformer: tokenization, autoregressive modeling, attention, and the quadratic bottleneck."
concepts: [transformer, autoregressive, tokenization, attention, query-key-value, causal-mask, RMSNorm, residual, quadratic-complexity]
sources:
  - tag: video
    label: "Lecture 14 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=pwQ0l4hFCVI
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: finish the sentence

"The cat sat on the". A human finishes it: "mat". The job: build a
machine that, given any text prefix, predicts the next piece of
text. Do this well enough, at scale, and you get the foundation
models of lecture 12. This chapter builds the machine that does it:
the transformer.

## First attempt: predict whole sentences

The naive idea: model entire sentences at once. Learn p("the cat
sat on the mat") as one number. Three facts kill it. Sentences have
different lengths: a model for 6-word sentences cannot score 7-word
ones. The space explodes: with 50,000 tokens, there are 50,000^10
possible 10-token sentences. And whole-sentence scores cannot
generate: you need the next word, not a verdict on a finished
sentence.

## The autoregressive fix

The **chain rule** of probability factors the joint into a product
of next-step predictions:

```ascii
p("the cat sat") = p("the") * p("cat"|"the") * p("sat"|"the cat")
```

Each factor is one small question: given the words so far, what
comes next? Train a model to answer that question on billions of
words, and the product scores whole sequences. This is the
**autoregressive** paradigm: predict one piece at a time, feed each
prediction back as context for the next. Generation is just repeated
prediction.

## Tokenization: what is a piece?

The model eats numbers, not text. **Tokenization** cuts text into
pieces (**tokens**) and maps each to an integer. What should a piece
be? Characters are too many steps per sentence. Words fail on rare
words: "internationalization" as one token never shares anything
with "internationalized" or "nation", though they share meaning.

The answer is **subword** tokenization: split rare words into
frequent pieces. "Internationalization" becomes "international" +
"ization". The model learns "international" once and reuses it.
The lecture's live example: "LLMefication" splits into pieces the
model already knows. Vocabulary around 50,000 pieces covers a
language. Every token becomes a vector (**embedding**). The model
never sees raw text, only sequences of vectors.

## First attempt at context: fixed windows

Each token's vector should carry its context: "frame" means one
thing in "frame the situation" and another in "photo frame". The
naive idea: mix each token with its, say, 5 neighbors using fixed
weights. It fails where it matters: in "The counselor helped frame
the situation", the word that disambiguates "frame" ("counselor")
sits 3 back, but in another sentence the key word sits 20 back. Fixed
windows cannot reach it. Fixed weights cannot know which neighbor
matters. The mixing must be data-dependent: each token decides, per
sentence, which other tokens to listen to.

## The key question

What if every token could look directly at every other token and
decide how much each one matters, with the decision computed from
the tokens themselves?

## Attention, by hand

**Attention** is a lookup. Each token produces three vectors: a
**query** (what I am looking for), a **key** (what I contain), and a
**value** (what I offer). To build the new representation of token
i: compare its query against every token's key (scores), turn scores
into weights summing to 1 (**softmax**), mix the values by the
weights.

Work it on a toy. Three tokens, 2-D vectors, identity projections
(the real model learns them. The mechanism is identical).

```ascii
tokens:  counselor = [1, 0]   helped = [0, 1]   frame = [1, 1]
query for "frame": q = [1, 1]

scores (dot products of q with each key):
  q . counselor = 1,   q . helped = 1,   q . frame = 2

softmax: e^1 = 2.72, e^1 = 2.72, e^2 = 7.39; total = 12.83
weights = [0.21, 0.21, 0.58]

new "frame" = 0.21*[1,0] + 0.21*[0,1] + 0.58*[1,1] = [0.79, 0.79]
```

"Frame" pulled 21 percent from "counselor", 21 percent from
"helped", 58 percent from itself. The mix came from the data: the
query-key matches decided it. Every token does this against every
other token, simultaneously. That is attention, and the result is a
**contextual representation**: each token's vector now carries its
sentence.

The full operation is four matrix steps. Stack N tokens into N x d
matrix X. Project to Q, K, V (three learned matrices). Scores =
QK^T (N x N). Softmax each row (scaled by 1/sqrt(d)). Output =
weights * V.

![Attention](assets/svg/l14-attn.svg "Attention. Queries meet keys, scores become weights, values mix. The toy: frame pulls 21/21/58 percent. Source: original plate for Stanford Frontier AI.")

## Causal masking: no peeking

For next-word prediction, token 5 may only use tokens 1-4. If
"frame" could see "situation" (token 6) during training, the model
would cheat: it predicts the future from the future. The **causal
mask** forbids it: before the softmax, set scores for future
positions to negative infinity. Softmax turns them to exactly 0.
The weight matrix becomes lower-triangular: each row only mixes the
past. At generation time this matches reality: the future does not
exist yet.

![Causal mask](assets/svg/l14-mask.svg "The causal mask. Future scores become zero weight. Each token sees only its past. Source: original plate for Stanford Frontier AI.")

## The block: attention plus MLP, with residuals

One transformer block: attention (mix across positions), then an
MLP (bend within each position, lecture 7), each wrapped in a
**residual connection** (lecture 7: out = x + F(x)) and
**RMSNorm** (rescale each vector to unit root-mean-square:
stabilizes the signal through dozens of blocks). Stack 32-96
blocks. The final block's last-position vector feeds a softmax over
the 50,000-token vocabulary: the next-token distribution. Train
with cross-entropy (lecture 4) against the true next token.

![Transformer block](assets/svg/l14-t2.svg "The transformer block. Attention mixes across positions, MLP bends within positions, residuals and RMSNorm stabilize the stack. Source: original plate for Stanford Frontier AI.")

## The honest price: quadratic

The score matrix is N x N: every token pairs with every token.
Double the length, quadruple the work. At N = 4,096: 16.7M scores
per layer per head, most of which must sit in memory. This single
fact, attention is quadratic in sequence length, is the bottleneck
the next lecture attacks. The transformer won training (fully
parallel, unlike the RNN's serial chain). The field now pays for
inference.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| Autoregressive factorization | Whole sentences: variable length, 50,000^10 space, cannot generate | Chain rule: one next-token question at a time; generation is repeated prediction |
| Subword tokenization | Characters too many steps; words cannot share across rare forms | "Internationalization" -> "international"+"ization"; ~50K pieces cover a language |
| Attention | Fixed windows cannot reach far context with data-dependent weights | Q/K/V lookup; toy: frame = 0.21 counselor + 0.21 helped + 0.58 self |
| Causal mask | Training would peek at the future | Future scores -> zero weight; lower-triangular; matches generation reality |
| Block design | Deep stacks destabilize | Attention + MLP, residuals (lecture 7), RMSNorm; 32-96 blocks |
| Quadratic price | Nothing is free | N x N scores: 16.7M at N=4,096; the next lecture's target |

> [!QA]
> Q: Why predict one token at a time instead of whole sentences?
> A: Three reasons. Sentences vary in length: a fixed-size model cannot score all lengths. The space explodes: 50,000^10 possible 10-token sentences cannot be tabulated. Generation needs the next word, not a whole-sentence verdict. The chain rule turns the joint distribution into a product of next-token conditionals, each a small learnable question. Train on billions of those questions and the product scores anything.
> Follow-up: What is teacher forcing?
> A: During training, feed the true prefix at every position instead of the model's own predictions. All positions train in parallel against known targets. The mismatch: at generation time the model feeds itself its own (possibly wrong) predictions, so errors compound. That train/generate gap is called exposure bias.

> [!QA]
> Q: Work the attention toy: why does "frame" get weights [0.21, 0.21, 0.58]?
> A: Query [1,1] dot each key: counselor [1,0] gives 1, helped [0,1] gives 1, frame [1,1] gives 2. Softmax: e^1=2.72 twice, e^2=7.39, total 12.83. Weights: 2.72/12.83=0.21, 0.21, 7.39/12.83=0.58. New frame = 0.21*[1,0]+0.21*[0,1]+0.58*[1,1] = [0.79,0.79]. The query-key matches set the mix: "frame" matches itself best, the other two equally.
> Follow-up: Why divide by sqrt(d) before the softmax?
> A: Dot products grow with dimension: variance d for random vectors. Unscaled, the softmax saturates in large models: one weight nears 1, the rest near 0, gradients through the tiny weights die. Dividing by sqrt(d) restores unit variance and keeps the softmax responsive. Forget it and large-model attention collapses onto one token per query early in training.

> [!QA]
> Q: Why is attention O(N^2), and what does the causal mask change about the cost?
> A: The score matrix QK^T has one entry per token pair: N^2 entries to compute and (naively) store. At N=4,096 that is 16.7M per layer per head. The causal mask zeroes the upper triangle but the standard implementation still computes all N^2 scores. It saves nothing computationally, only enforces the no-peeking rule. True savings need different algorithms (the next lecture) or different architectures.
> Follow-up: Why did transformers beat RNNs if attention is quadratic?
> A: Training economics. RNNs are serial: step t waits for t-1, leaving thousands of GPU cores idle. Attention computes all pairs in parallel: every core works. At scale, the architecture that trains fast on huge data wins even if each step costs more. Inference is where the quadratic bill comes due.

## Recap: the whole lesson on one screen

1. **The job.** Finish "The cat sat on the". Predict the next
   piece, repeatedly.
2. **Whole sentences fail.** Lengths vary, 50,000^10 space, cannot
   generate.
3. **Autoregressive.** Chain rule: p = product of next-token
   conditionals. Generation is repeated prediction.
4. **Tokenization.** Subword pieces: "internationalization" shares
   "international". ~50K vocabulary.
5. **Fixed windows fail.** Key context sits at varying distances.
   Weights must be data-dependent.
6. **Attention.** Q/K/V lookup. Toy: frame = [0.79, 0.79] from
   weights [0.21, 0.21, 0.58]. Contextual representations.
7. **Causal mask.** Future scores to zero. No peeking. Matches
   generation.
8. **The block.** Attention + MLP, residuals, RMSNorm, 32-96 deep.
   Softmax over vocabulary, cross-entropy loss.
9. **The honest price.** N x N scores: 16.7M at N = 4,096. Won
   training. Inference pays.

## Official sources and further reading

**Official:**
- Lecture 14 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=pwQ0l4hFCVI — Tengyu Ma builds
  tokenization (subword, the "internationalization" and
  "LLMefication" examples), the autoregressive paradigm, attention
  from Q/K/V, causal masking, and the quadratic bottleneck.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the formal
  attention equations.

**Caveats from these sources.** The attention toy in this lesson is
an original miniature following the lecture's Q/K/V framing with
identity projections for visible arithmetic. The 16.7M figure is
N^2 at N = 4,096 per layer per head, the lecture's canonical
bottleneck illustration. RMSNorm is presented as the lecture
presents it: per-vector rescaling for stack stability.

## Connections to the other courses

- **CS229 L04:** softmax and cross-entropy: the output layer and
  the training loss.
- **CS229 L07:** residuals and the MLP inside each block. The RNN
  contrast from the sequence lesson.
- **CS229 L12:** this machine, pre-trained: the foundation model.
- **CS229 L15:** attacking the quadratic price: KV cache, GQA,
  MoE. Then ICL and SFT.
- **CS224N:** attention from the NLP side: the same mechanism,
  translation as the driving example.
- **CS336:** the transformer block built in full: the systems
  view of this lesson.
