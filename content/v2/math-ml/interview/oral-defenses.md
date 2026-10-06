# Oral defenses, math-ml RUN 5

Date: 2026-10-06. Prompts only. Rubrics in interview/keys-oral.md.
Format: 15 minutes per unit, closed-book, no lesson open. The
examiner reads the opening prompt verbatim, then follows the
ladder. The learner may use a blank page. Numbers must be
recomputed live, not quoted.

## Defense O-U01, mathematical language (15 min)

Opening: "Here are six rows of (x, y). Plot them honestly, then
tell me what the plot licenses and what it does not."
Follow-ups:
1. Your axes do not start at zero. Defend or fix.
2. Write the assertion you would put before the plot code.
3. I change one y value by 10x. What breaks first: the plot, the
mean, or your conclusion? Rank them.
4. Seed 7 gives one shuffle. Seed 8 another. Which claims in
your summary survive the seed change?
Pass: the learner names what the plot cannot show (causation,
outside the range) without prompting.

## Defense O-U02, linear geometry (15 min)

Opening: "Here is a 2x2 matrix with rank 1. Show me what it keeps
and what it kills, geometrically."
Follow-ups:
1. Write the nullspace and the column space. How are their
dimensions related?
2. I hand you a vector in the killed direction and ask for the
pre-image. What do you say?
3. The SVD gives singular values 5 and 0. What is the best
rank-1 approximation error, and why is it exactly that?
4. kappa is 10^6. I need 6 good digits. Can you deliver? Show
the arithmetic.
Pass: the learner refuses the pre-image and quantifies the
kappa bound.

## Defense O-U03, calculus and optimization (15 min)

Opening: "Minimize f(x) = x^4 - 3x^2 + 2 by hand. Name every
critical point and its type."
Follow-ups:
1. Your gradient descent from x = 0 stalls. Explain, then fix
it without changing f.
2. Write the Newton step at x = 1. Will it converge? Why?
3. I add the constraint x >= 1. Write the KKT conditions and
solve.
4. Your finite-difference gradient disagrees with the analytic
one at 1e-3 relative error. Is that a bug? Justify with the
truncation argument.
Pass: the learner catches the x = 0 saddle and the KKT
complementarity.

## Defense O-U04, probability and estimation (15 min)

Opening: "Four coin flips, three heads. Give me three different
estimates of p and defend each."
Follow-ups:
1. Write the likelihood and find its maximum by hand.
2. Your Bayesian friend uses a Beta(2,2) prior. Compute the
posterior mean. Who is more biased, you or your friend?
3. Define KL divergence and tell me which direction you would
use to fit a model to data, and why.
4. Your density estimate at x = 1.5 is 0.2535 (the Parzen toy).
I double the bandwidth. Predict the direction of change.
Pass: the learner states MLE 0.75, Laplace 0.6667, MAP with the
prior, and names the bias-variance trade without notes.

## Defense O-U05, risk and generalization (15 min)

Opening: "Your model gets 100 percent train accuracy and 60
percent test accuracy. Diagnose out loud."
Follow-ups:
1. Write the two risks and say which one you optimized.
2. Name three leak channels and how you would test each.
3. Your learning curve gap closed at n = 2000 but error sits at
0.27 with noise floor 0.25. I offer 10x more labels. Take it or
decline? Argue from the curve.
4. Ridge moved w from 0.9286 to 0.8125 on the lesson toy.
Explain the move as a prior, then as a penalty. Both must
agree.
Pass: the learner declines the labels and cites the noise
floor.

## Defense O-U06, linear models and margins (15 min)

Opening: "Fit y = b + w x to (0,0), (1,1), (2,3), (3,2) by hand.
Then tell me what the fit assumes about the noise."
Follow-ups:
1. Your residuals sum to zero. Prove it must, given the
intercept, in two lines.
2. I relabel this as classification with a sigmoid. Write one
gradient step from w = 0 on the points (0,0),(1,1).
3. The Gram of your RBF kernel has cond 3.2e6. Your solver
reports residual 4e-11. Do you ship? Justify.
4. Two alphas, a* = 0.25 each, w = [0.5, 0.5]. Verify duality
holds and name the constraint that makes it hold.
Pass: the learner refuses to ship on the residual alone and
cites the condition number.

## Defense O-U09, unsupervised and latent (15 min)

Opening: "Six points, two clusters, no labels. Find the
clusters and prove your answer is locally optimal."
Follow-ups:
1. Your k-means objective is 0.1333. I start both centers on
the left. Predict the path, then explain what the experiment
showed.
2. I scale one axis by 100 and rerun. Predict the assignments.
What is the one-line fix?
3. PCA keeps 99.72 percent of variance with k = 1. I keep k = 2
instead. Who wins on held-out data and why?
4. Your mixture model has a component with 2 points and
variance -> 0. Name the failure and the lesson-04c concept that
covers it.
Pass: the learner predicts the scaling damage before
computing and names degeneracy without re-teaching it.
