---
page_id: cs229-l01
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 1
nav: "L01 · Introduction"
title: "Lecture 1: What Machine Learning Is"
summary: "Two classic definitions of machine learning, the three learning paradigms, and the map of the whole course."
date: "2026-04-06"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "36:47"
video_id: DATnpGoGhM8
video_title: "Lecture 1: Introduction"
video_caption: "Original lecture. Tengyu Ma defines machine learning, surveys the three paradigms, and maps the course."
concepts: [machine-learning, supervised-learning, unsupervised-learning, reinforcement-learning, Samuel, Mitchell]
sources:
  - tag: video
    label: "Lecture 1 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=DATnpGoGhM8
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## How to read this lesson

This lesson has two levels. **Level 1 (Core)** contains what you need to
understand everything that follows in CS229 and the courses that build on
it. **Level 2 (Deep)** contains what you need for correct, interview-grade
understanding. Read Level 1 straight through. Return to Level 2 when you
want depth.

No prerequisites are assumed. Every term is defined at first use.

## Level 1: Two definitions of machine learning

Machine learning is usually defined in one of two ways. The older one is
from Arthur Samuel in 1959. He called it the field of study that gives
computers the ability to learn without being explicitly programmed
[05:57](ts:05:57).

The sharper one is from Tom Mitchell in 1998. A computer program learns
from experience E with respect to tasks T and performance measure P, if
its performance on T, measured by P, improves with experience E.

Read that slowly. Three slots. The task T: what the program must do. The
experience E: the data it learns from. The measure P: the score that says
whether it got better. Every lecture in this course fills those three
slots for one more algorithm.

> [!QA]
> Q: What is machine learning, in one sentence?
> A: A program that gets better at a task as it sees more data, judged by a fixed score. Mitchell's version names the three slots: task T, experience E, performance P. Samuel's version is shorter: learning without being explicitly programmed. Interviewers accept either. Mitchell's is more precise.
> Follow-up: Why does the definition matter in practice?
> A: It forces you to name the score before you train. Teams that skip P argue about results forever. The definition also separates ML from ordinary programming: no one writes the rules. The data writes them.

## Level 1: The three paradigms

Every algorithm in this course belongs to one of three families. The
difference is what the data contains.

![Three learning paradigms](assets/svg/l01-paradigms.svg "Supervised: labeled pairs, predict y from x. Unsupervised: x only, find structure. Reinforcement: act, get reward, sequential decisions. Original plate.")

**Supervised learning.** The data has answers. Each training example is a
pair (x, y): an input x and its correct label y. The program learns to
predict y from x. Regression predicts a number, like a house price.
Classification picks a category, like cat versus dog. What makes it
supervised is the training set. No labels, no supervision.

**Unsupervised learning.** The data has no answers. Each example is just
x. The program must find structure on its own: groups of similar points
(clustering), or the directions where the data varies most. This family
had a renaissance in the last ten years.

**Reinforcement learning.** The program acts in a world and receives
rewards. Decisions come in sequences, and each decision changes what
happens next. Robots, game players, and the reasoning step of modern
language models all live here.

> [!QA]
> Q: How do you tell which paradigm a problem belongs to?
> A: Look at the data and the feedback. Labeled pairs with right answers: supervised. Raw data with no labels: unsupervised. An agent that acts and gets scored over time: reinforcement. A common interview trap is calling clustering supervised because the groups look obvious to a human. If no labels were given to the program, it is unsupervised.

## Level 1: The map of this course

The course runs in three blocks, in this order.

![Course arc](assets/svg/l01-roadmap.svg "Block 1, lectures 2 to 8: supervised learning. Block 2, lectures 9 to 10: unsupervised learning. Block 3, lectures 11 to 17: diffusion, LLMs, RL. Original plate.")

Block one, lectures 2 to 8, is supervised learning. Regression, then
classification, then neural networks. Block two, lectures 9 and 10, is
unsupervised learning: clustering and dimensionality reduction. Block
three, lectures 11 to 17, is the modern half: diffusion models, large
language models, and reinforcement learning.

Each block builds on the previous one. Loss functions defined in lecture
2 reappear in lecture 14. Probability from lecture 3 powers lecture 11.
Nothing is taught twice. When an idea returns, the lesson links back to
its first definition.

## Level 2: Why foundations when models are huge

A fair question. Modern models have billions of parameters and train on
the internet. Why study the mathematics of small models from decades ago?

Tengyu Ma's answer has two parts. The methodology of training is still
similar to the old days: define a loss, compute gradients, update
parameters. What changed is the usage. You used to train a model per
task. Now you prompt one model for many tasks. The foundation did not
move. The building on top got taller.

Chris Ré puts it more bluntly in lecture 2: the field made the models a
lot bigger, made the GPUs go fast, and got AI. Scale changed the results.
It did not change the underlying trade. Machine learning is still a
tradeoff between computation and prediction. The rest of the course makes
that tradeoff exact.

> [!QA]
> Q: Do I need the math, or can I just use the libraries?
> A: You can use the libraries without the math until something breaks. Then the math is the only debugger you have. A model that does not converge, a loss that explodes, a cluster count that looks wrong: each of these is a mathematical symptom. This course teaches you to read the symptoms. That is the whole point of a foundations course.

## Level 2: Course mechanics worth knowing

The course is mathematically intense. Derivations matter more than
programming. The stated prerequisites are probability and linear algebra,
at the level of CS109 or STAT116. Every proof in the course assumes you
can follow a probability argument and a matrix argument.

AI tools are allowed as collaborators. They cannot be copied into
submissions. The policy treats the tool like a study partner: useful for
understanding, not a substitute for your own derivations.

## Recap: the whole lesson on one screen

Eight ideas carry this lecture. Read each card. Say the core sentence out
loud. If you can, you own the lesson.

<div class="recap-grid">
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Three learning paradigms">
<div class="rc-body">
<strong>1. Samuel 1959: learn without being programmed</strong>
<p>The oldest definition. Computers learn from data instead of following
hand-written rules. Short and famous. Less precise than Mitchell.</p>
<p class="rc-num">Key: learning without explicit programming</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Three learning paradigms">
<div class="rc-body">
<strong>2. Mitchell 1998: T, E, P</strong>
<p>A program learns from experience E on tasks T, measured by P, if P
improves with E. Three slots. Every algorithm in this course fills them.</p>
<p class="rc-num">Key: task, experience, performance measure</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Supervised learning">
<div class="rc-body">
<strong>3. Supervised: the data has answers</strong>
<p>Training pairs (x, y). Predict y from x. Regression predicts numbers.
Classification picks categories. The labels are what make it
supervised.</p>
<p class="rc-num">Key: (x, y) pairs; predict y</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Unsupervised learning">
<div class="rc-body">
<strong>4. Unsupervised: no answers, find structure</strong>
<p>Data is x only. Find groups or directions of variation. Clustering and
PCA live here. Renaissance in the last ten years.</p>
<p class="rc-num">Key: x only; structure, not labels</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Reinforcement learning">
<div class="rc-body">
<strong>5. Reinforcement: act, then get scored</strong>
<p>An agent takes actions in sequence and receives rewards. Decisions
change the future. Robots, games, LLM reasoning.</p>
<p class="rc-num">Key: sequential decisions plus reward</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-roadmap.svg" alt="Course arc">
<div class="rc-body">
<strong>6. The course has three blocks</strong>
<p>Lectures 2 to 8: supervised. Lectures 9 to 10: unsupervised. Lectures
11 to 17: diffusion, LLMs, RL. Each block builds on the last.</p>
<p class="rc-num">Key: 2-8 supervised, 9-10 unsupervised, 11-17 modern</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-roadmap.svg" alt="Course arc">
<div class="rc-body">
<strong>7. Methodology is old, usage is new</strong>
<p>Training still means: define a loss, compute gradients, update. What
changed is scale and prompting. Foundations did not move.</p>
<p class="rc-num">Key: loss plus gradients plus scale</p>
</div>
</div>
<div class="recap-card">
<img src="assets/svg/l01-paradigms.svg" alt="Prerequisites">
<div class="rc-body">
<strong>8. Prerequisites: probability and linear algebra</strong>
<p>CS109 or STAT116 level. Derivations over programming. AI tools allowed
as collaborators, never as submissions.</p>
<p class="rc-num">Key: probability plus linear algebra</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 1 video: definitions at [05:57](ts:05:57), course agenda in the opening minutes.
- CS229 Spring 2026 official course notes: the probability and linear algebra review sections.

**Further reading:**
- Mitchell, Machine Learning (1997), Chapter 1: the full T/E/P treatment.
- Samuel (1959), "Some Studies in Machine Learning Using the Game of Checkers": the original definition.

**Caveats from these sources.** The definitions are philosophical, not
mathematical: no theorem follows from them. Mitchell's T/E/P framing is
the one interviewers expect. The "renaissance" of unsupervised learning
refers to the 2015-2025 period and is a qualitative claim, not a measured
one.

## Connections to the other courses

- **CS336:** the LLM block of CS229 (lectures 12 to 17) is the conceptual prerequisite for CS336's full language-modeling pipeline.
- **CS224N:** word-level models from 2024 meet the same three paradigms; tokenization changes the input, not the paradigm.
- **CS329H:** reinforcement learning from human feedback extends lecture 16 and 17's reward formalism.

> [!CHEAT]
> **Introduction cheatsheet.** Samuel 1959: learn without explicit programming. Mitchell 1998: improve P on T with E. Three paradigms: supervised ((x,y), predict y), unsupervised (x only, find structure), reinforcement (act, get reward, sequential). Course: L02-08 supervised, L09-10 unsupervised, L11-17 diffusion plus LLMs plus RL. Prerequisites: probability, linear algebra. AI tools: collaborators, not submissions.

> [!MEMORY]
> **T, E, P.** Every algorithm answers three questions. What is the task. What is the experience. What is the score. If you cannot name all three, you do not understand the algorithm yet.
