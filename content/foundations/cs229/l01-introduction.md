---
title: "L01: Introduction"
course: cs229
type: lesson
video: DATnpGoGhM8
duration: "36:59"
instructor: Tengyu Ma
term: Spring 2026
prev: index.html
next: l02-supervised-setup.html
---

A map of the course and of machine learning itself. Tengyu Ma argues that the training methodology of ML has barely changed since the 1950s, but how we use the models has changed completely. This course teaches the training side: the mathematical foundations that still drive everything from linear regression to LLM post-training.

## The definition that still holds

Arthur Samuel, 1959: machine learning is the field of study that gives computers the ability to learn without being explicitly programmed. [05:18](ts:318)

Tom Mitchell, 1998, made it precise: a program learns from experience E with respect to tasks T and performance measure P if its performance on T, measured by P, improves with E. [06:07](ts:367)

Three components matter:

- **Experience E** is data, broadly. Web text, human labels, synthetic data, even a model's own thinking tokens. [06:34](ts:394)
- **Tasks T** used to be narrow (one classifier, one price predictor). Now one model serves millions of tasks. [07:08](ts:428)
- **Performance P** is the goal that drives learning. If more data does not improve performance, the system is not learning. [07:16](ts:436)

## The three paradigms

Supervised, unsupervised, and reinforcement learning. The boundaries blur more every year, and Ma treats them as tools rather than end goals. Nobody's end goal is "do supervised learning." The end goal is a general-purpose learning agent, and supervised learning is one tool used to build it. Supervised fine-tuning (SFT) is supervised learning wearing a modern name. [08:43](ts:523)

## Supervised learning: the running example

House price prediction. Each property is a point: square feet on the x-axis, price on the y-axis. Fit a line, read off the prediction at a new x. Fit a quadratic if the data curves. [09:48](ts:588)

Notation used through the whole course: **x** is the input, **y** is the output. Inputs are also called features. Outputs are also called labels. Ma prefers "input" and "output" to avoid confusion. [10:21](ts:621)

**Regression** means the output is continuous (price). **Classification** means the output is discrete (house vs townhouse, flamingo vs cat). The classifier draws a boundary that splits the input space. [12:43](ts:763)

Why classification matters in 2026: every large language model is fundamentally a classifier. Next-token prediction chooses one word out of 50,000 to 250,000 possible tokens. That is classification with a very large label set. [13:57](ts:837)

## From ImageNet to general models

Fei-Fei Li's ImageNet gave the field 1 million labeled images, 20x bigger than anything before. AlexNet trained on it with neural networks and started the deep learning era. [15:26](ts:926)

The progression since: vision, then language, then large language models. Each wave reused the same training methodology with bigger data and bigger models. That is why a course on the foundations still applies. [16:09](ts:969)

## What this course covers

The lecture walks the full syllabus at high level:

1. **Supervised learning** (weeks 1-3): regression, classification, the probabilistic view. The tools behind SFT.
2. **Deep learning** (weeks 4-5): neural network architecture and backpropagation. The engine of modern models.
3. **Generalization** (week 6): bias-variance, double descent, why bigger models can generalize.
4. **Unsupervised learning** (weeks 7-8): clustering, EM, PCA. Learning structure without labels.
5. **Generative and foundation models** (weeks 9-10): diffusion, representation learning, LLMs, in-context learning.
6. **Reinforcement learning** (weeks 11-12): MDPs, policy gradients, PPO. The machinery behind RLHF and RLVR.

## Prerequisites and posture

Probability and linear algebra (CS109/STAT116 level). The course is mathematically intense: derivations and modeling formulations, not programming tutorials. Friday TA lectures cover background gaps. [02:14](ts:134)

AI tools are allowed as collaborators, not as answer generators. The point of the course is training yourself, not training your agent. [02:50](ts:170)

> **Interview line:** When asked "what is machine learning," give Mitchell's definition (E, T, P), then add the 2026 update: the training methodology is classical, but the models are general-purpose and the experience now includes synthetic data. Then name the three paradigms and where each appears in an LLM pipeline: pretraining is unsupervised, SFT is supervised, RLHF is reinforcement learning.

## Sources

- Video: [Lecture 1: Introduction](https://www.youtube.com/watch?v=DATnpGoGhM8) (36:59)
- Notes: CS229 Spring 2026 lecture notes, Chapters 1-2 (linear regression, classification)
