---
page_id: cs336-lab4
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 94
nav: "Lab 4 · Data"
title: "Lab 4: Data Pipelines"
summary: "MinHash deduplication and filtering decisions. 45 minutes."
sources:
  - tag: assignment
    label: "CS336 Assignment 4: Data"
  - tag: video
    label: "Lecture 14 video"
    url: https://www.youtube.com/watch?v=5sxHosTLPF8
---

Background: [Lecture 13](l13-data-sources.html), [Lecture 14](l14-data-processing.html).

## Exercise 1: MinHash from scratch (20 min)

Implement MinHash deduplication for a list of documents.

```python
def minhash_signature(shingles: set, num_perm: int, prime: int) -> list[int]:
    """Return the MinHash signature. shingles: set of hashed shingle ids."""
    # Your code here.

def jaccard_estimate(sig1: list[int], sig2: list[int]) -> float:
    # Your code here.
```

1. Prove that P(minhash match) = Jaccard similarity. Two sentences.
2. With 128 permutations, two documents have estimated similarity 0.85. Give a 95% confidence interval for the true Jaccard.
3. Explain LSH banding: with b bands and r rows, what similarity threshold does the scheme target? Derive (1/b)^(1/r).

> [!INTERVIEW] Deduplication questions test whether you know the probabilistic machinery, not just the library call.

## Exercise 2: Filtering decisions (15 min)

You have 10T tokens of web crawl. Your compute budget allows 2T training tokens.

1. Name three filtering stages in order, and what each removes.
2. A classifier flags 40% of your data as low quality. Your model trained on the filtered 60% gets worse perplexity than one trained on a random 60%. Diagnose: what are the two most likely causes?
3. When would you deliberately train on unfiltered data? Give a concrete scenario.

## Exercise 3: Data mixing (10 min)

You have code, math, and web text. Your target task is code generation.

1. Write the mixing proportions you would start with and justify each number.
2. Your code benchmark improves but your general benchmark collapses. What happened, and what is the fix?
3. Explain why epoching the small code corpus 50 times is worse than it looks. Use the epochs formula.
