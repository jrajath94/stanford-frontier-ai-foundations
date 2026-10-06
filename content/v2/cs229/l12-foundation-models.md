---
page_id: cs229-l12
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 12
nav: "L12 · Foundation Models"
title: "Lecture 12: Foundation Models and Representation Learning"
summary: "The paradigm shift after GPT-3: pre-train on massive unlabeled data, then adapt. Representations, linear probing, LoRA, parallelism."
date: "2026-05-13"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "1:19:44"
video_id: _kREM2UAiJ8
video_title: "Lecture 12: Foundation Models"
video_caption: "Original lecture. Tengyu Ma presents the pre-train/adapt paradigm, representation learning, linear probing, and LoRA."
concepts: [foundation-model, pre-training, adaptation, representation-learning, embedding, linear-probing, LoRA, multitask, parallelism]
sources:
  - tag: video
    label: "Lecture 12 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=_kREM2UAiJ8
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: a sentiment classifier with 500 labels

A startup has 500 labeled movie reviews (positive/negative) and
wants a classifier. Lectures 2 through 8 say: fit a model on the
500. A linear model on 500 examples with 10,000 word features will
memorize noise (lecture 6). A neural network will do worse. The
classical toolkit starves on 500 labels.

But the startup also has the entire internet: billions of unlabeled
sentences. The key question: can the unlabeled billions teach the
model about language first, so the 500 labels only need to teach the
sentiment part?

## First attempt: train on the 500 alone

Watch it fail with numbers. Vocabulary 10,000 words, logistic
regression: 10,001 knobs, 500 examples. The model fits training
perfectly (loss near 0) and scores 58 percent on fresh reviews:
barely above coin flip. The word "masterpiece" appears twice, both
positive. The model assigns it a huge weight. In fresh reviews
"masterpiece" appears in sarcastic negatives and the classifier
burns. This is lecture 6's variance with the numbers filled in:
n << d, memorization, no generalization.

## The paradigm: pre-train, then adapt

A **foundation model** is a large model **pre-trained** on massive
broad data, then **adapted** to specific tasks. Pre-training uses a
self-supervised objective that needs no labels: predict the next
word (lecture 14), denoise an image (lecture 11), fill masked words.
The model learns **representations**: internal vectors that capture
meaning. "Masterpiece" and "triumph" end up near each other in
representation space because they appear in similar contexts across
billions of sentences.

Then adaptation spends the 500 labels wisely. The representations
already know language. The labels only teach the sentiment
direction. The lecture calls this the emergent paradigm after
GPT-3: one pre-trained model serves thousands of downstream tasks,
and the per-task data need drops by orders of magnitude.

Why it works, mechanistically: pre-training moves the model to a
region of parameter space where good solutions for many tasks are
nearby. Fine-tuning on 500 labels is a short walk from an excellent
starting point, not a random search. The 500 labels steer. The
billions built the map.

![Foundation model paradigm](assets/svg/l12-fm.svg "The foundation model paradigm. Pre-train once on massive unlabeled data, adapt cheaply per task. Representations transfer. Labels steer. Source: original plate for Stanford Frontier AI.")

## Linear probing: how good are the representations?

**Linear probing** is the honesty test. Freeze the pre-trained
model. Take its representations of your 500 reviews. Train only a
linear classifier (lecture 2's machinery) on top. If the
representations are good, a linear probe scores well: the sentiment
direction is already linearly separable in representation space.

Numbers from the lecture's framing: the linear probe on frozen
representations hits ~88 percent on fresh reviews versus 58 percent
for the from-scratch model. The 30-point gap is the value of
pre-training, isolated: same 500 labels, same linear head, the only
difference is the representations underneath. When a probe fails,
the representations lack the task's information and you need deeper
adaptation.

## LoRA: adapt without moving billions

Full fine-tuning updates every parameter: for a 7B model, 7 billion
knobs move on 500 labels. Overkill and overfitting-prone, and each
task needs its own 7B copy. **LoRA** (low-rank adaptation) restricts
the update's degrees of freedom. For each weight matrix W, write the
adapted matrix as:

```ascii
W_adapted = W + A * B
```

W stays frozen. A and B are thin: if W is d by d, A is d by r and B
is r by d, with **rank** r small. Only A and B train.

Count the savings with the lecture's numbers: d = 1,000, r = 10.
Full update: d^2 = 1,000,000 degrees of freedom. LoRA: 2*d*r =
20,000. Fifty times fewer. For real models (d = 4,096, r = 8):
16.8M full vs 65,536 LoRA: a 256x reduction. The adapter (A, B)
is megabytes. The base model stays shared across tasks. Ship one
7B model plus a 10MB adapter per task instead of a 7B copy per
task.

Why low rank works: adaptation is a small change (the lecture's
point: "adapting means you don't have to change much"), and small
changes to big matrices live in low-dimensional subspaces. The
rank r is the dial: r = 8 to 64 in practice, tuned on dev.

## Multitask and parallelism

Two more pieces from the lecture. **Multitask** pre-training and
adaptation: train on many tasks at once so representations serve
all of them. The shared structure (grammar, facts, reasoning
patterns) transfers. And **parallelism**: foundation models only
exist because training parallelizes across thousands of GPUs
(data, tensor, and pipeline parallelism split the work). The
paradigm is a systems achievement as much as an algorithmic one:
the pre-training run that makes everything else cheap is itself
enormously expensive, amortized over all downstream uses.

## The honest price

The paradigm buys label efficiency and pays four prices. First,
pre-training cost: one run costs millions of dollars and months.
Only a few organizations can pay it. Second, opacity squared: not
only are the representations unreadable, their failures are
unpredictable: the model inherits every bias and poison in the
unlabeled billions. Third, adaptation limits: linear probing fails
when the task needs information the pre-training never captured;
LoRA's low rank cannot express large behavioral changes. Fourth,
evaluation contamination: the "unlabeled" internet contains the
test sets, so reported numbers may measure memorization, not
generalization. The lecture is clear-eyed: the paradigm is
emergent, powerful, and not yet fully understood.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| Pre-train then adapt | 500 labels, 10,000 features: 58 percent, memorization | Billions of unlabeled sentences build representations; 500 labels steer |
| Representations | Words are atomic symbols | "Masterpiece" near "triumph" in vector space from shared contexts |
| Linear probing | Are the representations any good? | Freeze, train linear head: 88 vs 58 percent isolates pre-training's value |
| LoRA | Full fine-tuning moves 7B knobs per task | W + AB, rank r: d=1000, r=10 gives 20,000 vs 1,000,000 dof; MB adapters |
| Multitask + parallelism | One model per task; training too big for one GPU | Shared structure transfers; 3D parallelism makes the pre-train run possible |

> [!QA]
> Q: What is a foundation model, and what changed with GPT-3?
> A: A large model pre-trained on massive broad data and adapted per task, instead of trained per task from scratch. Before, each task needed its own labeled dataset and model: 500 labels meant a 500-label model. After GPT-3, one pre-trained model serves thousands of tasks: the 500 labels adapt a model that already knows language. The lecture calls it an emergent paradigm: scale unlocked behavior (few-shot learning, broad transfer) that smaller models did not show.
> Follow-up: Why does pre-training help with only 500 labels?
> A: It moves the starting point. Random initialization plus 500 labels is a blind search in a huge space: memorization. Pre-trained weights start near good solutions for many tasks, so 500 labels do a short, guided walk instead. Mechanistically, the representations already separate the concepts. The labels only identify which direction is sentiment.

> [!QA]
> Q: How does LoRA work, and how much does it save?
> A: Freeze the pre-trained weight W (d by d). Learn the update as a product of thin matrices: W + A*B with A d-by-r, B r-by-d, rank r small. Only A and B train. At d = 1,000, r = 10: 20,000 trainable values versus 1,000,000 for a full update, a 50x cut. At d = 4,096, r = 8: 65,536 versus 16.8M, a 256x cut. Each task ships a megabyte adapter. The base model is shared.
> Follow-up: When does LoRA fail and full fine-tuning win?
> A: When adaptation needs a large behavioral change that does not fit in a rank-r subspace: e.g., teaching the model a wholly new domain or unlearning deeply ingrained behavior. Rank r is the capacity dial. Raise it and LoRA approaches full fine-tuning's expressiveness while losing its savings. Linear probing failing first is the signal that shallow adaptation is insufficient.

> [!QA]
> Q: What does linear probing tell you that fine-tuning accuracy does not?
> A: Whether the representations already contain the task's information. Freeze the model, train only a linear head: high probe accuracy means the concept is linearly readable in representation space, and pre-training did the real work. Fine-tuning accuracy mixes representation quality with the adapter's power to reshape them. Probe first: it is cheap, and it tells you whether to trust the representations or distrust the pre-training.
> Follow-up: The probe scores 88 percent but full fine-tuning scores 89. What do you conclude?
> A: The representations already solve the task. Fine-tuning adds almost nothing. Ship the probe: it is simpler, faster, and less prone to overfitting the 500 labels. Save deep adaptation for tasks where the probe lags.

## Recap: the whole lesson on one screen

1. **The job.** Sentiment from 500 labels. From scratch: 58
   percent, memorization.
2. **The key question.** Can unlabeled billions teach language
   first, so 500 labels teach only sentiment?
3. **The paradigm.** Pre-train on broad data (no labels needed),
   adapt per task. One model, thousands of tasks.
4. **Representations.** Vectors capturing meaning from context.
   The map the labels steer on.
5. **Linear probing.** Freeze, linear head: 88 vs 58 percent. The
   honesty test for representations.
6. **LoRA.** W + AB, rank r. d=1000, r=10: 20K vs 1M dof. MB
   adapters, shared base.
7. **Multitask, parallelism.** Shared structure transfers; 3D
   parallelism makes pre-training possible.
8. **The honest price.** Millions per pre-train, inherited biases,
   adaptation limits, test contamination.

## Official sources and further reading

**Official:**
- Lecture 12 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=_kREM2UAiJ8 — Tengyu Ma presents
  the pre-train/adapt paradigm, representation learning, linear
  probing, LoRA with the rank arithmetic, multitask, and
  parallelism.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the formal
  treatment.

**Caveats from these sources.** The LoRA arithmetic (d = 1,000,
r = 10, 2dr vs d^2) is the lecture's own numerical illustration.
The 500-label sentiment toy and its 58/88 percent numbers are an
original miniature demonstrating the lecture's paradigm claims.
The lecture presents the paradigm as emergent and not fully
understood. The mechanisms given here are the lecture's.

## Connections to the other courses

- **CS229 L06:** the bias-variance roots of the 500-label failure.
- **CS229 L13:** contrastive learning: another way to build
  representations without labels.
- **CS229 L14-L15:** the transformer: the architecture that made
  foundation models scale; SFT as adaptation.
- **CS229 L17:** RL as the adaptation step for reasoning.
- **CS336:** pre-training at scale: the systems behind the
  paradigm.
- **CS224N:** pre-training for language: from word vectors to
  foundation models.
