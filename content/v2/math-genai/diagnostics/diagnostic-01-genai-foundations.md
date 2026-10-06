# Diagnostic 01, generative modelling foundations placement

Date: 2026-10-06. Closed book. 30 items, 45 minutes.
Rubric: 90-100 ready for U02. 70-89 review the missed R/C sections,
then proceed. Below 70 complete the matched remediation in
prerequisites.md before U02. Keys in diagnostics/keys-01.md.

## R1-R3 vectors

Q01. For x = [0, 1, 0, 1], what is x_2? What is the shape of x?
Q02. Compute [0, 1, 0, 1] dot [1, 1, 1, 1].
Q03. What is the mean of [0, 0, 0, 0] and [1, 1, 1, 1]?

## R4-R6 probability

Q04. A fair coin is flipped once. Name the sample space.
Q05. Give a probability mass function on {a, b} and verify it sums
to 1.
Q06. A game pays 4 for heads, 0 for tails, fair coin. What is the
expected payout?

## R7-R8 estimation

Q07. 7 heads in 10 flips. What is the MLE of the bias?
Q08. 1 head in 1 flip. What does MLE say, and why should you
distrust it?

## R9-R10 logs

Q09. Compute log2(1/8). Write the product-to-sum log rule.
Q10. A model assigns probability 2^-20 to a file. How many bits
does that file cost in log-likelihood?

## C01 data vs model

Q11. Name p_data, p_hat, q for the eight-file toy.
Q12. Which of the three may training change?

## C02 density vs sample

Q13. Is one drawn file the distribution? Explain in one sentence.
Q14. p_hat says P(1010) = 0.125. Is 0.125 a fact or a claim?

## C03 likelihood

Q15. Write l(theta) for n independent files.
Q16. q_uni on the eight files: give the log-likelihood in bits.

## C04 support

Q17. q_bad gives 0 mass to 0101, seen twice. What is the
log-likelihood?
Q18. Name the check that catches this before scoring.

## C05-C06 latent variables

Q19. In the coin toy, what is z? Is it observed?
Q20. Compute p(heads) by marginalization. Show the sum.
Q21. Compute p(A | heads). Show the division.

## C07 entropy

Q22. Compute the entropy of a fair coin in bits.
Q23. Entropy 0 means what about the distribution?

## C08-C09 cross-entropy and KL

Q24. Write H(p, q) in one line.
Q25. H(p_hat) = 1.9056, H(p_hat, q_uni) = 2.0. What is the KL?
Q26. Can KL be negative? Why?

## C10 divergence choice

Q27. q_bad has infinite KL but TV = 0.375. What does this tell you
about choosing a ruler?

## C11 estimation

Q28. 3 heads in 10 flips. MLE of theta, with the derivative step.

## C12 evaluation

Q29. Why can training likelihood not detect memorization?
Q30. Two models score -1.7 and -2.0 bits per held-out file. Which
is better, and what must be true of the held-out set?
