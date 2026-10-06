# Keys, lab 10 generative bridge and synthesis

Date: 2026-10-06.

## Task 1

(a) p(x=1) = 0.5(0.1) + 0.5(0.8) = 0.45. log p
= -0.7985076962. Joint: [0.05, 0.4]. ELBO =
0.4 log 0.05 + 0.6 log 0.4 + H([0.4,0.6]) =
-1.1983 - 0.5497 + 0.6730 = -1.0750556815.
Gap = 0.2765479853.

(b) Posterior = [0.05/0.45, 0.4/0.45] =
[0.1111, 0.8889]. KL = 0.4 log(0.4/0.1111) +
0.6 log(0.6/0.8889) = 0.2765479853. Matches
the gap to 10 digits.

(c) Accept code with assert elbo <= logp.

## Task 2

(a) Measured ratio 3.9594 at seed 12 (4000
trials).

(b) |3.9594 - 4.0| = 0.0406 vs |4.2887 - 4.0|
= 0.2887: seed 12 lands closer. That is
exactly E21's warning: the ratio is a random
variable over seeds, and one seed cannot tell
you which side of 4.0 the next seed lands on.
Report the seed, the trial count, and the
predicted value, or report nothing.

## Task 3

(a) Informative: beta 0: -0.9302. beta 1:
-1.3283. beta 10: -5.0713. Collapsed at beta
10: -1.6695 + 0 = -1.6695.

(b) The optimizer prefers the collapsed q
(-1.6695 > -5.0713). The latent is dead: q =
p carries no information about x, and the
"better objective" bought a worse model.
