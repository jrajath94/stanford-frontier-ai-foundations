---
page_id: math-ml-l01
course_slug: math-ml
course_name: "Mathematical Foundations of Machine Learning"
course_order: 10
order: 1
nav: "L01 · Vectors as Data"
title: "Lecture 1: Vectors as Data"
summary: "Why machine learning speaks in vectors: the dot product as a similarity meter, the norm as a size meter, and both computed by hand on real numbers."
date: "2026-10-05"
instructor: "Prof. Sanjeev Kumar and Prof. S. K. Gupta"
offering: "NPTEL (IIT Roorkee)"
concepts: [vector, dot-product, norm, cosine-similarity, linear-combination, linear-independence]
sources:
  - tag: video
    label: "Essential Mathematics for Machine Learning — Lectures 01 (Vectors), 02 (Matrix Algebra), 07 (Norms and Spaces)"
    url: https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
  - tag: supplement
    label: "Deisenroth, Faisal, Ong, Mathematics for Machine Learning, ch. 2"
    url: https://mml-book.github.io
  - tag: supplement
    label: "Strang, Introduction to Linear Algebra, ch. 1"
---

## The task: turn the world into numbers a machine can compare

A machine learning model cannot read a review, look at a photo, or hear
a word. It can only do arithmetic. So every modern ML system starts
with the same move: turn each object into a list of numbers. That list
is a **vector**.

```ascii
movie review -->  [ 0.9, -0.4,  0.2 ]   "glowing review"
movie review -->  [-0.8,  0.7, -0.3 ]   "angry review"
photo pixel  -->  [ 214,  96,  31 ]    orange-red pixel
```

A vector is an ordered list of numbers. Order matters: [1, 2] and
[2, 1] are different vectors. Each number is a **component**. A vector
with n components lives in a space called R^n, read "R n": R^2 is the
plane, R^3 is ordinary 3-D space, and language models live in R^12,288.
The vectors that turn words into lists of numbers are called
**embeddings**, and attention scores in the transformer are dot
products of embeddings (CS229S L02).

This lesson builds the three tools every ML system uses on vectors:
multiply two vectors together to measure agreement, measure a vector's
size, and compare directions while ignoring size.

## First attempt: count shared words

The simplest way to compare two documents is to count words. Make a
vector of word counts and compare the counts.

```ascii
review A: "great film, great acting"   --> [ great:2, film:1, boring:0 ]
review B: "great film, boring plot"    --> [ great:1, film:1, boring:1 ]
```

How similar are A and B? The shared words are "great" and "film".
Count them with weights: multiply the counts component by component
and add.

```ascii
agreement = 2*1 + 1*1 + 0*1 = 2 + 1 + 0 = 3
```

This operation has a name: the **dot product**. For vectors
x = [x1, x2, ..., xn] and y = [y1, y2, ..., yn],

```ascii
x . y = x1*y1 + x2*y2 + ... + xn*yn
```

It is the single most used operation in machine learning. Attention
compares queries to keys with it. Neurons sum weighted inputs with
it. Classifiers score examples with it.

But raw counts hide a trap. Watch what happens when one review is
just longer.

## Where counts break: size confuses the comparison

Take review A and a copy of it pasted three times, call it A3.

```ascii
A:  [2, 1, 0]
A3: [6, 3, 0]    (same opinions, three times the words)
B:  [1, 1, 1]
```

Score each against B with the dot product:

```ascii
A  . B = 2*1 + 1*1 + 0*1 = 3
A3 . B = 6*1 + 3*1 + 0*1 = 9
```

A3 scores three times higher, but it says nothing new. The dot
product mixed two things together: how much the directions agree,
and how long the vectors are. A long vector always wins, even when
it carries no extra meaning.

So we need a separate measure of size: the **norm**. The norm of a
vector is the square root of the sum of squared components:

```ascii
||x|| = sqrt(x1^2 + x2^2 + ... + xn^2)

||A||  = sqrt(4 + 1 + 0) = sqrt(5)  = 2.24
||A3|| = sqrt(36 + 9 + 0) = sqrt(45) = 6.71
```

(||x|| is the notation for "the norm of x".) The norm is the length
of the arrow if you draw the vector. A3 is three times longer than
A, which is exactly the trap: the dot product rewarded length.

## The key question

How do you compare two vectors by direction alone, so that a long
vector and a short vector pointing the same way count as identical?

## The new idea: divide out the size

The fix is one division. Divide the dot product by both norms.
The result is the **cosine similarity**, a number between -1 and 1
that measures direction only:

```ascii
cosine(x, y) = (x . y) / (||x|| * ||y||)

cosine(A, B)  = 3 / (2.24 * 1.73) = 3 / 3.88 = 0.77
cosine(A3, B) = 9 / (6.71 * 1.73) = 9 / 11.6 = 0.77
```

Identical. The length canceled out, as it should: A and A3 point
the same way. Cosine similarity is why a short tweet and a long
article about the same topic can be recognized as the same topic.
Search engines and recommendation systems live on this ratio.

The name comes from geometry: the dot product also equals
||x|| * ||y|| * cos(theta), where theta is the angle between the
two vectors. Parallel vectors give 1, perpendicular vectors give
0, opposite vectors give -1. The lecture works the same formula
from this geometric side [uncertain: exact worked example unknown].

## Building blocks: linear combination and independence

Two more ideas from the lecture complete the vector toolkit.

A **linear combination** mixes vectors by scaling and adding. For
scalars c1, c2 and vectors v1, v2, the vector c1*v1 + c2*v2 is a
linear combination. The lecture's own example: in R^2, every vector
[c1, c2] is c1*[1,0] + c2*[0,1]. The vectors [1,0] and [0,1] are the
**standard vectors**; they are the pure axes.

**Linear independence** asks whether one vector in a set is secretly
a copy of the others. Vectors v1, ..., vk are **dependent** if some
mix c1*v1 + ... + ck*vk equals the zero vector with at least one
nonzero coefficient. The lecture's example: [1,1] and [2,2] are
dependent, because 2*[1,1] - 1*[2,2] = [0,0]. One is just twice the
other, so it adds no new direction.

Why does this matter for ML? A dataset where one feature is twice
another (say, price in dollars and price in cents) carries no extra
information. The second feature is a dependent vector. It wastes a
dimension and, as L09 shows, it can make regression break.

| Tool | Formula | What it measures |
|---|---|---|
| Dot product | x . y = sum of xi*yi | Agreement, but mixed with size |
| Norm | ||x|| = sqrt(sum of xi^2) | Size only |
| Cosine similarity | (x . y) / (||x|| ||y||) | Direction only, between -1 and 1 |
| Linear combination | c1*v1 + ... + ck*vk | Mixing vectors by scaling and adding |
| Independence | no nonzero mix gives zero | Whether a vector adds a new direction |

> [!QA]
> Q: What is a vector, and why does ML use them for everything?
> A: A vector is an ordered list of numbers, like [0.9, -0.4, 0.2]. ML uses vectors because arithmetic is all a computer can do: a review, a pixel, or a word must become numbers before a model can touch it. Embeddings turn words into vectors in R^12,288, and attention scores are dot products between them.
> Follow-up: Why ordered? Why is [1, 2] different from [2, 1]?
> A: Because each position has a fixed meaning. In a word-count vector, position 1 might mean "great" and position 2 "boring". Swapping them swaps the meaning. In an embedding, position 47 tracks some learned property; scrambling positions destroys the code.

> [!QA]
> Q: When do you use the dot product and when do you use cosine similarity?
> A: Use the dot product inside the model: attention scores, neuron sums, classifier margins. Use cosine similarity when comparing finished vectors, like ranking search results or measuring how close two embeddings are. In the worked toy, A and its tripled copy A3 scored 3 and 9 by dot product but both scored 0.77 by cosine. Cosine ignores the length; the dot product does not.
> Follow-up: Can the dot product be negative?
> A: Yes. A negative dot product means the vectors point in opposing directions: mostly disagreement. Cosine similarity maps this to -1. In attention this matters: a strongly negative score becomes a near-zero weight after softmax, so the model ignores that token.

> [!QA]
> Q: What does linear independence mean for a dataset?
> A: It means no feature is a copy of the others. [1,1] and [2,2] are dependent because 2*[1,1] - [2,2] = [0,0]. In a dataset, price-in-dollars and price-in-cents are dependent features: the second adds no direction. Dependent features waste dimensions and can make regression singular, which L09 demonstrates.
> Follow-up: How do you test it on real vectors?
> A: Stack them as rows of a matrix and row-reduce. If any row becomes all zeros, the set is dependent. The lecture works this exact procedure on [1,1,8,1], [1,0,3,0], [3,1,14,1] and finds a zero row: dependent.

## Recap: the whole lesson on one screen

1. **The task.** Machines only do arithmetic, so objects become vectors: ordered lists of numbers.
2. **First attempt.** Compare documents by multiplying matching word counts and adding: the dot product, x . y = 3 in the toy.
3. **Where it breaks.** Length confuses the score: a tripled review scores 3x higher (9 vs 3) with no new content.
4. **Measure size separately.** The norm ||x|| = sqrt(sum of squares): ||A|| = 2.24, ||A3|| = 6.71.
5. **The key question.** How to compare direction alone?
6. **The fix.** Cosine similarity divides out both sizes: 0.77 for A and A3 alike. Geometry agrees: it is the cosine of the angle.
7. **The building blocks.** Linear combinations mix vectors; independence means no vector is a copy of the rest.
8. **The price and the bridge.** Cosine throws away size, which sometimes matters (a long detailed review may deserve more weight). Keep both tools. L02 turns vectors into matrices: machines that transform whole datasets at once.

## Official sources and further reading

**Official:**
- "Essential Mathematics for Machine Learning" playlist, Lectures 01, 02, 07:
  https://www.youtube.com/playlist?list=PLLy_2iUCG87D1CXFxE-SxCFZUiJzQ3IvE
- NPTEL course page (111107137): https://archive.nptel.ac.in/courses/111/107/111107137/

**Further reading:**
- Deisenroth, Faisal, Ong, "Mathematics for Machine Learning", ch. 2 (free):
  https://mml-book.github.io — vectors, norms, dot products, independence.
- Strang, "Introduction to Linear Algebra", ch. 1 — the geometric view.

**Caveats.** The lecture-1 transcript was recovered in full; its vector definitions, dot product, linear combination, and independence examples are quoted above as the lecture presents them. The cosine-similarity framing is standard material consistent with Lectures 01/07; the exact toy numbers are the lesson's own. [uncertain]

## Connections to the other courses

- **CS229S L02:** attention scores are dot products of query and key embeddings; this lesson is the arithmetic underneath.
- **CS336:** token embeddings are vectors in R^d; cosine similarity ranks nearest neighbors in embedding space.
- **CS229 L10:** PCA starts by centering data vectors; norms measure reconstruction error.
