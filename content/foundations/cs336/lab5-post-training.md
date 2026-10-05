---
page_id: cs336-lab5
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 95
nav: "Lab 5 · Post-Training"
title: "Lab 5: Post-Training, SFT, DPO, RLVR"
summary: "Implement the DPO loss and derive the policy gradient. 45 minutes."
sources:
  - tag: assignment
    label: "CS336 Assignment 5: Alignment"
  - tag: video
    label: "Lecture 15 video"
    url: https://www.youtube.com/watch?v=2oH6PWPrYFo
---

Background: [Lecture 15](l15-post-training-sft-rlhf.html), [Lecture 16](l16-post-training-rlvr.html).

## Exercise 1: The DPO loss (20 min)

Implement the DPO loss from scratch.

```python
def dpo_loss(pi_logps_chosen, pi_logps_rejected, ref_logps_chosen, ref_logps_rejected, beta):
    """Return the DPO loss. All inputs are log-probabilities of full responses."""
    # Your code here.
```

1. Derive it: start from the Bradley-Terry preference model and the KL-constrained reward maximization. Show where the reward model disappears.
2. What does beta control? Describe the failure mode when beta is too large and when it is too small.
3. Your DPO run decreases the log-probability of BOTH chosen and rejected responses. Explain why this can still be correct.

> [!INTERVIEW] This is the most asked-about alignment derivation in 2025-2026 interviews. Know it cold: the exponential tilting step is where candidates get stuck.

## Exercise 2: Policy gradient basics (15 min)

1. Derive the REINFORCE gradient estimator from the expected return. State the one assumption you need.
2. Add a baseline. Prove it does not bias the gradient.
3. GRPO uses group-relative advantages instead of a learned value function. Explain the tradeoff in two sentences: what you gain, what you lose.

## Exercise 3: SFT vs RL (10 min)

1. Your SFT model follows instructions but hallucinates on tail knowledge. Explain the mechanism: why does SFT cause this specific failure?
2. When is SFT sufficient and RL unnecessary? Give a concrete task and justify.
3. Name one thing RLHF optimizes that SFT cannot, even with infinite SFT data.
