# U12 lab , measurement validity on synthetic toys

Instructions: implement each task as a small numpy function, then run
`python3 u12_lab_run.py`. Your outputs must match `u12_lab_key.md`
exactly. All corpora and scores are synthetic.

## Task 1 , perplexity (C01)

1a. Convert losses 2.3 and 1.7: perplexities 9.97 and 5.47.

## Task 2 , contamination (C02)

2a. Report overlap 100.0% and canary recall 1.00 on the toy.

## Task 3 , tokenizer comparability (C03)

3a. Bits/byte A=0.687, B=0.605: B wins despite higher per-token
loss.

## Task 4 , internal vs external (C04)

4a. 20 decisions at 5% FP: P(>=1 bogus)=0.64.

## Task 5 , task suites (C05)

5a. Report math 0.80+-0.124, code 0.65+-0.121 with CIs.

## Task 6 , formats (C06)

6a. Format accuracies 0.717/0.723/0.657, spread 0.066.

## Task 7 , generation settings (C07)

7a. Write the eval config record: model, item hash, T=0, seed 0.

## Task 8 , judge reliability (C08)

8a. On the 4-item toy: agreement 0.75, kappa 0.50.

## Task 9 , seed variation (C09)

9a. 5-seed range on the toy: 0.034.

## Task 10 , slices (C10)

10a. Gap 0.15 with gap SE 0.088: not significant.

## Task 11 , ecological validity (C11)

11a. Name the three gaps: input length, format, grading.

## Task 12 , cost per correct (C12)

12a. Model A $0.0022, model B $0.0056 per correct task.
