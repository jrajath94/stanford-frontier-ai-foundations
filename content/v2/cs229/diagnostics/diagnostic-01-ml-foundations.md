# Diagnostic 01: ML foundations

Date: 2026-10-06. Attempt closed-book. 20 minutes. Answers in
keys-01.md. This is a placement test, not a grade.

## D01

A hospital table has one row per patient. Columns: age, blood
pressure, cholesterol, and a final column "heart attack in 10
years: yes/no". Name the features and the target.

## D02

A model reports 98 percent accuracy on its training data and
72 percent on new data. In one sentence, name the failure.

## D03

You fit a straight line to data that curves. The training error
stays high. Name the failure and state one fix.

## D04

What does it mean for a model to generalize?

## D05

Two students train on the same data. Student A tunes 40 settings
on the test set until the score is high. Student B tunes on a
validation set and tests once. Whose test score can you trust,
and why?

## D06

A classifier labels every patient "healthy". The dataset has 5
percent sick patients. What is its accuracy? Is accuracy a good
metric here?

## D07

Write the shape of a vector `x` with 4 features. Write the shape
of a design matrix for 200 examples with 4 features.

## D08

A loss is 0 when the prediction is right and 1 when it is wrong.
The model outputs probabilities. Why is this loss a poor training
signal for gradient methods?
