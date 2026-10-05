---
title: "Lab 2: Bias-Variance in Practice"
course: cs229
type: lab
---

Diagnose and fix. 45 minutes. You are given learning curves, not code.

## Setup

You trained a model on 10,000 examples. You plot training error and validation error against training set size.

**Case A:** Training error is 2%. Validation error is 18%. The gap stays wide as you add data.

**Case B:** Training error is 15%. Validation error is 16%. Both stay flat as you add data.

**Case C:** Training error is 1%. Validation error is 12%. As you add data from 1k to 10k examples, the validation error keeps falling and the gap keeps narrowing.

## Problem 1: Diagnose (15 min)

For each case, state: high bias or high variance (or both)? Write one sentence of evidence for each diagnosis.

**Check:**
- A: high variance. Low training error plus a large persistent gap means the model fits the training data but does not generalize.
- B: high bias. Both errors are high and close together. The model cannot even fit the training data.
- C: high variance, but improving. The narrowing gap says more data is helping. Do not change the model yet.

## Problem 2: Prescribe (15 min)

For each case, list exactly two actions you would take next, in order. Be specific (name the technique, not "fix the model").

**Check:**
- A (high variance): (1) add regularization. ridge penalty, dropout, or early stopping; (2) get more training data or augment. Do not add features or make the model bigger.
- B (high bias): (1) make the model more expressive. add features, polynomial terms, or a deeper network; (2) reduce regularization if any is applied. More data will not help much.
- C: (1) collect or generate more training data; (2) hold off on regularization changes until the curve flattens.

**Interview trap:** the interviewer will push back on one of your prescriptions ("why not just add data for case B?"). The answer: with high bias, the model underfits even the training set, so more data of the same kind cannot fix a model that is too simple. Point at the flat training curve as evidence.

## Problem 3: The double-descent twist (15 min)

You increase model width past the interpolation threshold. Test error first rises, then falls again, ending below its previous best.

1. Sketch the curve and label the three regimes.
2. In one paragraph, explain why the classical bias-variance story predicted this would not happen, and what actually drives the second descent (implicit regularization of the optimizer, benign overfitting).
3. Name one practical consequence for LLM training.

**Check:** regimes are under-parameterized (classical U), interpolation threshold (peak), over-parameterized (second descent). Practical consequence: training very large models to near-zero training loss is not just memorization; the optimizer's implicit bias selects simple interpolating solutions, which is part of why scale works.

## Scoring

- Correct diagnoses plus defensible prescriptions: you can debug models in interviews.
- If you prescribed "more data" for high bias, reread L06 before Lab 3.
