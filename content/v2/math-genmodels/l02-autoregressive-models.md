---
page_id: math-genmodels-l02
course_slug: math-genmodels
course_name: "Mathematical Foundations of Generative Models"
course_order: 12
order: 2
nav: "L02 · Autoregressive Models"
title: "Lecture 2: Autoregressive Models — The Storyteller"
summary: "The storyteller answers the one question step by step: factor the distribution with the chain rule, learn each conditional, sample one token at a time. Teacher forcing worked by hand, exposure bias demonstrated with numbers, and the serial price."
date: "2026-10-05"
instructor: "Prof. Prathosh A P"
offering: "2025"
video_id: PtDFqdTbQUY
concepts: [autoregressive-model, chain-rule, conditional-probability, teacher-forcing, exposure-bias, perplexity, causal-masking]
sources:
  - tag: video
    label: "W10L40: Auto-regressive models (video PtDFqdTbQUY)"
    url: https://www.youtube.com/watch?v=PtDFqdTbQUY
---

## The question, for sequences

Restate the one question for sequences: you have samples of
sequences (sentences, audio, pixel rows). Learn the rule behind
them. Draw fresh sequences from it.

The storyteller's answer: do not learn the whole sequence at once.
Tell it one word at a time. To write a sentence, predict the first
word, then the second word given the first, then the third given
the first two. Learning the rule means learning each of these
small prediction steps. Sampling means running the steps and
writing down what comes out.

## First attempt: count the conditionals

A **conditional probability** is a probability with a given past.
P(cat | the) reads "the probability of cat given that the came
before." Concretely, it is a fraction: among all the times "the"
appeared, how often did "cat" follow?

Watch it on a toy corpus of 10 sentences:

```ascii
"the cat sat"   x6
"the cat ran"   x2
"a  dog sat"    x2
```

The **chain rule** breaks a sequence probability into a product of
conditionals:

```ascii
P("the cat sat") = P(the) x P(cat | the) x P(sat | the cat)
```

Count each factor from the corpus:

```ascii
P(the)         = 8/10 = 0.80   ("the" starts 8 of 10 sentences)
P(cat | the)   = 8/8  = 1.00   (after "the", always "cat")
P(sat | the cat) = 6/8 = 0.75  (after "the cat", "sat" 6 of 8 times)

product: 0.80 x 1.00 x 0.75 = 0.60
```

Check against the corpus: 6 of 10 sentences are "the cat sat",
so the true fraction is 0.60. The chain rule reproduces it
exactly. The factorization is not an approximation. It is an
identity.

To sample, run it forward. Draw word 1 from P(word). Suppose "the".
Draw word 2 from P(word | the). Suppose "cat". Draw word 3 from
P(word | the cat). The machine wrote "the cat sat" one draw at a
time. That is an **autoregressive** model: each step regresses on
(depends on) the earlier steps.

## Where counting conditionals breaks

The counting machine needs one counter per history. To predict the
11th word from the previous 10, with a 50,000-word vocabulary,
there are 50,000^10 possible histories. That number has 47 digits.
Every history needs its own counter for the next word: another
factor of 50,000. The conditional table is the same exploding
table from L01, now with histories as rows.

```ascii
history length   rows in the table
1                50,000
5                50,000^5  (24 digits)
10               50,000^10 (47 digits)
```

Worse, most histories never appear. "the cat sat on the" might
appear never, yet the machine must still predict what follows.
The counter for an unseen history is empty. Counting cannot
generalize across histories.

## The key question

What if one smooth predictor learned all the conditionals at
once, trained on true histories, sharing strength between similar
pasts?

## The new idea: one predictor, forced teachers

Replace the table with a neural network. The network reads a
history and outputs a probability for every vocabulary word. One
set of weights serves all histories. Similar histories flow
through similar paths, so "the cat sat on the" borrows from "a dog
sat on the". The table is gone. The sharing from L01 does the
work.

Training uses **teacher forcing**. The name is literal: the teacher
(the true data) forces the correct history at every step. To train
on "the cat sat", the model never feeds its own guesses back in.
Each step gets the true past:

```ascii
step 1: input [START]            target: the
step 2: input [START, the]       target: cat
step 3: input [START, the, cat]  target: sat
```

The loss is the negative log probability of each true target,
summed. Suppose the model outputs:

```ascii
P(the | START)       = 0.70
P(cat | the)         = 0.90
P(sat | the cat)     = 0.50

loss = -(log 0.70 + log 0.90 + log 0.50)
     = -(-0.357 - 0.105 - 0.693)
     = 1.155 nats
```

**Perplexity** turns this into plain language: exp(average loss
per token) = exp(1.155/3) = exp(0.385) = 1.47. The model is as
confused as if choosing among 1.47 equally likely words. Lower is
better. Perplexity 1.47 on a 3-word toy. Real models report
perplexity on held-out text, and a few points of difference decide
leaderboards.

Teacher forcing has a second gift: parallelism. Every step's true
history is known upfront, so all steps train at once. The model
predicts all positions in one forward pass. This is why the
transformer trains fast: the chain rule's serial story becomes a
parallel computation, with a **causal mask** hiding the future
from each position.

## Tying to CS229S L02: the mask, worked

The gold standard lesson built attention by hand: for "frame" with
query [1,1], scores [1,1,2] became weights [0.21, 0.21, 0.58] and
output [0.79, 0.79]. That was full attention: every token sees
every token. The autoregressive transformer adds one rule: each
token sees only the past.

Run the same toy with a causal mask. Tokens: counselor = [1,0],
helped = [0,1], frame = [1,1]. Scores are dot products, as before.

```ascii
full score matrix:        masked (keep left + self):
counselor: [1, ., .]      counselor: [1]
helped:    [1, 1, .]      helped:    [1, 1]
frame:     [1, 1, 2]      frame:     [1, 1, 2]
```

Row by row, softmax over the kept scores:

```ascii
counselor: softmax([1])    = [1.00]            -> 1.00 x [1,0] = [1.00, 0]
helped:    softmax([1,1])  = [0.50, 0.50]      -> [0.50, 0.50]
frame:     softmax([1,1,2]) = [0.21, 0.21, 0.58] -> [0.79, 0.79]
```

"frame" is unchanged: its past is the whole toy. But "helped"
changed: with full attention it would mix all three tokens. Masked,
it mixes only counselor and itself into [0.50, 0.50]. The mask is
the chain rule wearing attention's clothes: position t predicts
from positions 1..t only. For more on the attention mechanism
itself, CS229S L02 is the full treatment. Here it serves the
storyteller.

## Mapping back: what each property fixes

| Counting failure | Storyteller answer | How |
|---|---|---|
| Conditional table explodes: 50,000^10 histories | One smooth predictor for all histories | Similar pasts share weights; no per-history counter |
| Unseen histories have empty counters | Neural generalization | "the cat sat on the" borrows from seen neighbors |
| Training would be serial | Teacher forcing | True histories known upfront; all positions train in one pass |

## The honest price: two bills

**Bill 1: exposure bias, demonstrated.** Training always shows
true histories. Sampling shows the model's own guesses. The first
mistake poisons every step after it. Work it out: suppose each
step has a 10% chance of picking a wrong token. The chance that a
20-token sample stays fully on track is 0.9^20 = 0.12. Only 12 in
100 samples avoid any mistake.

Watch one mistake compound on the toy. The model samples "dog"
after "the" (the 10% event). Now step 3 conditions on "the dog",
a history absent from training. The model's guess P(sat | the
dog) might be 0.10, against the true-history value P(sat | the
cat) = 0.75. One wrong step cut the next step's confidence more
than sevenfold. The model was never trained on its own mistakes,
so it handles them badly.

**Bill 2: sampling is serial, demonstrated.** Step t needs the
output of step t-1. A 1,000-token sample needs 1,000 sequential
forward passes. At 20 ms per pass, one sample takes 20 seconds.
Training parallelizes. Sampling does not. This is the mirror of
the RNN story in CS229S L02: there the serial chain hurt training,
here teacher forcing fixed training and the serial chain survives
in sampling.

> [!QA]
> Q: What is teacher forcing, exactly?
> A: During training, feed the model the true previous tokens at every step, never its own predictions. For "the cat sat", step 2 sees [START, the] even if the model would have guessed "a" at step 1. The loss sums negative log probabilities of the true targets: 1.155 nats on the toy, perplexity 1.47. It makes training parallel across positions and stable, because the model always conditions on clean histories.
> Follow-up: If teacher forcing is so good, why does sampling still fail?
> A: Because sampling removes the teacher. The model then conditions on its own guesses, which it never saw during training. With a 10% per-step error rate, only 0.9^20 = 12% of 20-token samples stay fully on track, and one wrong token ("the dog") can cut the next step's confidence from 0.75 to 0.10. This train-test mismatch is exposure bias.

> [!QA]
> Q: Why is the chain rule not an approximation?
> A: It is an identity of probability: P(A,B,C) = P(A) x P(B|A) x P(C|A,B) holds for any distribution, always. The toy verifies it numerically: 0.80 x 1.00 x 0.75 = 0.60 matches the corpus fraction 6/10 exactly. Approximation enters only when a model estimates the conditionals, never in the factorization itself.
> Follow-up: Then why do autoregressive models make mistakes the counting version would not?
> A: The counting version memorizes exact conditional fractions but needs 50,000^10 table rows for length-10 histories and fails on unseen histories. The neural version approximates every conditional with shared weights, so it generalizes to unseen histories but can misestimate seen ones. Exact factorization, approximate factors.

> [!QA]
> Q: What does the causal mask do to the CS229S attention toy?
> A: It restricts each token to its past. In the worked toy, "frame" keeps its full-context output [0.79, 0.79] because its past is the whole sequence, but "helped" drops from a three-way mix to [0.50, 0.50] over counselor and itself. The mask turns attention into the chain rule: position t may only use positions 1 through t.
> Follow-up: Why not just train left-to-right serially like an RNN?
> A: Because teacher forcing makes all true histories available upfront, so one masked forward pass computes every position's prediction simultaneously. The RNN's serial bottleneck was in training. The masked transformer trains in parallel and pays the serial price only at sampling time.

![Chapter plate: exact factorization, the serial bill moves to sampling](assets/plate-l02.png "Chapter plate. Exact factorization; the serial bill moves from training to sampling. Source: original plate for Stanford Frontier AI.")

## Recap: the whole lesson on one screen

1. **The question, for sequences.** Learn the rule behind sequences. Draw fresh ones.
2. **First attempt: count conditionals.** P("the cat sat") = 0.80 x 1.00 x 0.75 = 0.60, matching the corpus exactly. The chain rule is an identity.
3. **The conditional table explodes.** 50,000^10 histories for length 10. Unseen histories have empty counters.
4. **The key question.** What if one smooth predictor learned all conditionals at once, trained on true histories?
5. **Teacher forcing.** Feed true pasts at every step. Toy loss 1.155 nats, perplexity 1.47. All positions train in parallel.
6. **The mask, worked.** Causal masking on the CS229S toy: "helped" becomes [0.50, 0.50], "frame" stays [0.79, 0.79]. The chain rule in attention's clothes.
7. **Bill 1: exposure bias.** 0.9^20 = 0.12 of 20-token samples stay clean. One wrong token cuts the next confidence 0.75 to 0.10.
8. **Bill 2: serial sampling.** 1,000 tokens x 20 ms = 20 seconds per sample. Training parallelizes. Sampling does not.

## Official sources and further reading

**Official:**
- W10L40: Auto-regressive models (video PtDFqdTbQUY): the lecture this chapter follows. [uncertain]: the lecture's exact toy examples are not verified. The corpus counts and the worked loss here are original toys built to the lecture's topic.

**Further reading:**
- CS229S L02 (this system): the attention mechanism and the causal mask's home. The worked [0.79, 0.79] toy is reused from there.
- Bengio et al., "Neural Probabilistic Language Models" (2003): the original neural autoregressive model.

**Caveats from these sources.** Perplexity 1.47 is a toy value on three tokens. Real perplexities are measured on held-out corpora with vocabularies in the tens of thousands. The 20 ms per step is illustrative. Real per-step latency depends on model size and hardware. Exposure bias is real but its practical severity is debated. Scheduled sampling and other fixes exist.

## Connections to the other courses

- **CS229S L02:** the gold standard: attention built by hand, the [0.21, 0.21, 0.58] weights, and why the transformer trains in parallel. This lesson's mask section is a worked extension of it.
- **CS336:** the transformer block in full: how the masked attention stack becomes a language model, and training at scale.
- **CS224N:** the same autoregressive story from the NLP side, with machine translation as the driving example.
- **math-genai (sibling):** MLE and KL: teacher forcing minimizes KL between the data conditionals and the model conditionals, one position at a time.
