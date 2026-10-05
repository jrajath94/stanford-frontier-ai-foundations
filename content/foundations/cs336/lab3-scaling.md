---
page_id: cs336-lab3
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 93
nav: "Lab 3 · Scaling Laws"
title: "Lab 3: Scaling Laws"
summary: "Fit a scaling law from training curves and predict the compute-optimal model. 45 minutes."
sources:
  - tag: assignment
    label: "CS336 Assignment 3: Scaling"
  - tag: video
    label: "Lecture 9 video"
    url: https://www.youtube.com/watch?v=Q15rhEWZPQ4
---

Background: [Lecture 9](l09-scaling-laws-1.html), [Lecture 11](l11-scaling-laws-2.html).

## Exercise 1: Fit a data scaling law (20 min)

You trained 5 models on 1B, 2B, 4B, 8B, and 16B tokens. Final losses: 3.20, 3.02, 2.88, 2.77, 2.68.

1. Fit L(D) = a + b·D^(-α) on paper or in code. Report a, b, α.
2. Predict the loss at 100B tokens.
3. Your colleague says the fit is meaningless because the models are undertrained. Explain when this criticism is valid and when it is not.

> [!INTERVIEW] Know the difference between fitting a law and trusting it. Interviewers probe: what breaks the extrapolation?

## Exercise 2: Compute-optimal allocation (15 min)

You have 1e23 FLOPs. Assume L(N, D) = E + A·N^(-α) + B·D^(-β) with α = 0.34, β = 0.28, and C = 6ND.

1. Derive the optimal N and D. Show the Lagrangian.
2. Compute the Chinchilla ratio D/N from your exponents. Compare with 20 tokens per parameter.
3. You must serve the model to 100M users. How does your answer change, and why?

> [!INTERVIEW] The third question separates researchers from engineers. Training-optimal is not serving-optimal. Say why, with numbers.

## Exercise 3: Hyperparameter transfer (10 min)

You tuned the learning rate on a 100M model. Your 7B run diverges.

1. Name two reasons the optimal learning rate shifts with scale.
2. Explain the μP prescription in one paragraph: what stays constant, what scales, and with what rule.
3. When does μP break? Name one concrete case from the lecture.
