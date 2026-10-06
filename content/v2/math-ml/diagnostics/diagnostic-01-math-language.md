# Diagnostic 01, mathematical language placement

Date: 2026-10-06. Closed book. 30 items, 45 minutes.
Rubric: 90-100 ready for U02. 70-89 review the missed R/C sections, then
proceed. Below 70 complete the matched remediation in prerequisites.md
before U02. Keys in diagnostics/keys-01.md.

## R1-R3 numeracy

Q01. Compute 1/2 + 1/3 as a fraction.
Q02. Compute 2^10.
Q03. For x = [10, 20, 30], what is x_2^2?

## R4-R5 functions and substitution

Q04. f(x) = 2x + 3. Compute f(4).
Q05. With a = 7, b = 3, compute a^2 - b^2.

## R6-R8 Python

Q06. n = 6. M = n * 2. What does m hold?
Q07. What is total after: total = 0. For v in [2, 5, 7]: total = total + v?
Q08. def square(t): return t * t. What is square(9)?

## R9-R10 arrays

Q09. numpy.arange(6).reshape(2, 3) has what shape?
Q10. numpy.array([1, 2, 3]) * 2 gives what?

## C01 sets and functions

Q11. Is {(1, 5), (2, 7), (2, 9)} a function? Why?
Q12. f(x) = 2x + 3 on {1, 2, 3, 4}. List the image.

## C02 notation

Q13. Expand and compute sum_{i=1}^{3} x_i for x = [2, 5, 7].
Q14. What does ||[3, 4]|| mean, and what is its value?

## C03 units

Q15. Slope of price (thousand dollars) over size (sq ft) carries what
unit?
Q16. Why is 800 sq ft + 320 thousand dollars meaningless?

## C04 sums and products

Q17. Compute the sum and product of [2, 5, 7].

## C05 logs and exponents

Q18. Compute log10(200) via log10(2) + log10(100).
Q19. Why do likelihoods live in log space?

## C06 vectors

Q20. Compute [3, 4] dot [2, -1].
Q21. Compute ||[3, 4]||.

## C07 axes

Q22. M = [[0,1,2],[3,4,5]]. Compute the sum over axis 0 and its shape.
Q23. Same M. Compute the sum over axis 1 and its shape.

## C08 coding

Q24. What does vector_norm([3.0, 4.0]) return for the lesson code?

## C09 plots

Q25. Name two things that make the lesson scatter plot honest.

## C10 assertions

Q26. check_prob_vector([0.2, 0.3, 0.4]) passes or fails? Why?

## C11 precision

Q27. (0.1 + 0.2) == 0.3 in float64: True or False? Give the value.
Q28. Write the correct float comparison for a and b.

## C12 reproducibility

Q29. Two runs of default_rng(7).random(4): same or different numbers?
Q30. Name one failure caused by seeding inside a training loop.
