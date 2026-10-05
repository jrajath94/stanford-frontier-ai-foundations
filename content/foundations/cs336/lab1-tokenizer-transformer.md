---
page_id: cs336-lab1
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 91
nav: "Lab 1 · Tokenizer, Transformer"
title: "Lab 1: Tokenizer and Transformer from Scratch"
summary: "Implement BPE training and encoding, then a full Transformer forward pass. 45 minutes."
sources:
  - tag: assignment
    label: "CS336 Assignment 1: Basics"
  - tag: video
    label: "Lecture 1 video"
    url: https://www.youtube.com/watch?v=JuoVZkPBiKk
---

Background: [Lecture 1](l01-overview-tokenization.html), [Lecture 3](l03-architectures-hyperparameters.html).

## Exercise 1: BPE training (15 min)

Implement BPE merge learning on a small corpus. Input: a string and a target vocabulary size. Output: the ordered merge list.

```python
def learn_bpe(corpus: str, vocab_size: int) -> list[tuple[bytes, bytes]]:
    """Return merges in the order learned. Start from 256 byte tokens."""
    # Your code here.
```

Requirements:
- Pre-tokenize the corpus into words first. Explain why this matters for both speed and quality.
- Count adjacent pairs efficiently. State the complexity of your counting step.
- Handle the merge application without quadratic blowup on long inputs.

> [!INTERVIEW] Interviewers ask: why do merges apply in learned order during encoding, not by frequency at encode time? What breaks if you apply them out of order?

## Exercise 2: Encode and decode (10 min)

Given the merge list from Exercise 1, implement `encode` and `decode`. Prove the roundtrip property on three tricky inputs: an emoji, a Chinese character, and a string with leading spaces.

> [!INTERVIEW] Common follow-up: your encoder is slow on a 1MB document. Name two concrete optimizations and quantify the speedup each gives.

## Exercise 3: Attention from scratch (20 min)

Implement single-head scaled dot-product attention with a causal mask, using only basic tensor ops. Then answer:

1. Write the FLOP count as a function of sequence length T and head dimension d.
2. At what T does the attention matrix stop fitting in SRAM for d = 128? Show the arithmetic.
3. Explain why the causal mask is applied as an additive `-inf` before softmax rather than zeroing after.

```python
def attention(q, k, v):
    """q, k, v: (T, d). Return (T, d). Causal. No library attention."""
    # Your code here.
```

> [!INTERVIEW] The classic trap: candidates forget the scaling factor. Be ready to derive why dividing by sqrt(d) matters from the variance of the dot product.
