# Prerequisites: math-genmodels

Date: 2026-10-06. Baseline: October 6, 2026.

Shared modules live in
~/workspace/stanford-frontier-ai/v2-pack/shared/prerequisites/.
Each entry below links a shared module and names the unit-local
remediation that teaches the same idea inside the lesson. A link is
navigation. The local remediation is the teaching.

## Bridge map

| Code | Module | File | Used by | Local remediation |
| --- | --- | --- | --- | --- |
| P01 | Numeracy, algebra, notation | p01_numeracy.md | U01 | U01 toy arithmetic before symbols |
| P04 | Spectral and numerical linear algebra | p04_spectral.md | U03, U04 | U03-C04, U04-C02 local derivations |
| P05 | Scalar and multivariable calculus | p05_calculus.md | U02, U04, U06 | U02-C04, U04-C03, U06-C01 local derivations |
| P06 | Probability, events to distributions | p06_probability.md | U01, U06 | U01 is the local rebuild |
| P07 | Statistical estimation and uncertainty | p07_estimation.md | U01, U08 | U01-C09, U08-C01 local derivations |
| P08 | Information theory and density objectives | p08_information.md | U01-U06 | U01-C10/C11 local rebuild |
| P09 | Optimization and constrained problems | p09_optimization.md | U05 | U05-C07/C08 local derivations |
| P10 | ML foundations and evaluation | p10_ml_foundations.md | U08 | U08-C01 local protocol |
| P11 | Neural networks and autodiff | p11_neural_nets.md | U03, U05 | U03-C03, U05-C04 local mechanics |
| P13 | Language and sequence modelling | p13_language.md | U07 | U07-C03/C04 local rebuild |
| P14 | Transformer mechanics | p14_transformer.md | U07 | U07-C06 local mechanics |
| P15 | Hardware and computer architecture | p15_hardware.md | U07 | U07-C09 local cost model |
| P18 | Bayesian inference, latent variables, sampling | p18_bayesian.md | U02, U03, U06 | U02 is the local rebuild |
| P22 | Experimental method and research literacy | p22_experiments.md | U08 | U08-C07/C10 local protocol |

## Entry diagnostic

Answer closed-book. Each question names its remediation.

1. Compute 3/8 + 1/4 as a decimal. (P01. Remediation: U01-C01 toy.)
2. Write log(a*b) and log(a/b) in terms of log a and log b. (P01.)
3. A fair coin lands heads 3 times in 8 flips. Give the empirical
   probability of heads. (P06. Remediation: U01-C08.)
4. State Bayes rule in words, then in symbols. (P06/P18. Remediation:
   U02-C03.)
5. Define expectation of a discrete variable with 3 outcomes. (P06.)
6. A density integrates to 1.4 over its domain. What is wrong? (P08.
   Remediation: U01-C03.)
7. Compute KL divergence between Bernoulli(0.5) and Bernoulli(0.25) in
   nats. (P08. Remediation: U01-C11.)
8. Differentiate f(x) = x^3 * exp(x). (P05. Remediation: U04-C03.)
9. What is the determinant of diag(2, 0.5)? What does it say about
   area? (P04. Remediation: U04-C02.)
10. Write one step of gradient descent in symbols. (P09.
    Remediation: U05-C11.)
11. A model scores 0.9 accuracy on train and 0.6 on test. Name two
    possible causes. (P10. Remediation: U08-C01.)
12. State one requirement of a falsifiable hypothesis. (P22.
    Remediation: U08-C10.)

Pass bar: 9 of 12 with correct reasoning. Any fail routes to the
named shared module first, then the unit-local remediation, then a
retry of that question only.
