# Oral defenses: deep ladders

Course: math-genmodels. Date: 2026-10-06. Baseline: October 6, 2026.

Provenance: original practice. Not actual employer material. Keys
in interview/keys-oral.md. Each ladder is one mechanism defended
across 8 follow-ups: define, toy, derive, implement, cost,
compare, debug, critique. Answer closed-book.

## O1: the ELBO, from nothing

1. Define the latent-variable likelihood and say why it is
   intractable.
2. Work the 2-bag toy: compute the marginal, the posterior,
   and the ELBO at q = (0.7, 0.3).
3. Derive the ELBO identity from Jensen, naming the exact
   step where the inequality enters.
4. State the variational gap as a KL and say when it is zero.
5. Implement the reparameterized gradient for a 1-D Gaussian
   q in three lines of numpy.
6. State the per-sample cost of the ELBO versus exact
   marginalization on the toy.
7. Your ELBO trains but samples are bad. Name two distinct
   causes, one in q and one in the decoder.
8. "A higher ELBO always means a better model." Attack with
   the bound-versus-exact argument.

## O2: change of variables, end to end

1. Define invertibility and give a witness pair for z^2.
2. Compute the coupling Jacobian at (0.5, -1) by hand.
3. Derive the triangular log-det shortcut from the
   determinant expansion.
4. Evaluate the exact log-likelihood at the toy point,
   showing the sign.
5. Write the coupling forward and inverse in code and state
   the round-trip test.
6. State the per-layer cost of the log-det for dense versus
   triangular Jacobians at D = 1024.
7. The likelihood reports -2.2398 instead of -2.6860. All
   else checks out. Diagnose from the numbers.
8. "Flows are strictly better than VAEs because the
   likelihood is exact." Attack with two arguments.

## O3: the GAN game

1. Define an implicit distribution and say what it cannot do.
2. Compute r(0) and D*(0) for p = N(0,1), q = N(1,1).
3. Derive D* by pointwise maximization, showing concavity.
4. Derive max_D V = -log 4 + 2 JSD with the toy numbers.
5. Sketch the alternating training loop in pseudocode with
   the k:1 schedule.
6. State the per-step cost versus a VAE step and name the
   extra multiplier.
7. The discriminator hits 100 percent held-out accuracy.
   Explain why the generator learns nothing, with the
   mechanism.
8. "Adversarial training is the better density estimator
   because FID is lower." Attack with two arguments.

## O4: diffusion, from training to samples

1. Define the score and compute it for N(0,1) at x = 1.
2. State the DDPM forward closed form and evaluate x_4 on
   the toy chain.
3. Derive the denoising objective from Tweedie in three
   steps.
4. Write one DDIM eta = 0 step on the toy numbers.
5. Implement one Langevin step in numpy and state the
   stationary law.
6. Compare the sampling cost of DDPM-1000, DDIM-50, and
   Heun-50 in network calls.
7. Loss is 0.005 but samples are gray mush. The time
   embedding is present. Diagnose sampler versus model with
   one test.
8. "Best ODE likelihood means best generative model."
   Attack with two arguments.

## O5: the transformer as a likelihood machine

1. State the chain rule for L = 3 and evaluate p(abc) on
   the toy.
2. Define perplexity and compute it for the toy.
3. Read the 3x3 causal mask and say what position 2 sees.
4. Work the attention head: scores, weights, output.
5. Write the KV-cache update for one decode step in
   pseudocode.
6. Cost attention at L = 512, d = 64 in mults and cache
   bytes.
7. Loss is excellent, samples are garbage, masks are
   present. Name the most likely bug and its test.
8. "Lower pseudo-perplexity proves the better generative
   model." Attack with two arguments.

## O6: judging a generative model

1. Define held-out likelihood and compute A versus B on
   the toy.
2. Define sample precision and recall. Report all four toy
   numbers.
3. Construct the FID-0 counterexample from the toy.
4. Describe the memorization NN test and report both toy
   distances.
5. Design a compute-matched comparison protocol in five
   steps.
6. State the cost of the full suite versus a single FID
   number.
7. FID is good but domain experts reject the samples.
   Diagnose the feature mismatch and propose the fix.
8. "One number on a leaderboard decides the best model."
   Attack with two arguments.

## O7: VAE design and failure

1. Define the VAE objective and name its two terms.
2. Compute the Gaussian KL in closed form on the toy
   numbers.
3. Explain posterior collapse: the mechanism, not just
   the symptom.
4. State the beta-VAE tradeoff on the rate-distortion
   plane.
5. Sketch the reparameterized sampler in two lines of
   code.
6. State the per-batch cost of the VAE versus an exact
   flow likelihood.
7. KL is 0.0003 per dim but reconstructions are sharp.
   Diagnose and propose two fixes.
8. "The ELBO gap is just an optimization problem."
   Attack with the amortization and family arguments.

## O8: score matching, derived

1. Define the Fisher divergence between two score fields.
2. Derive the explicit score-matching objective by parts,
   stating the boundary assumption.
3. Show the denoising equivalence: where the trace goes.
4. Work the 1-parameter toy: optimum a = 1, value -0.5.
5. Implement the empirical objective on 5,000 samples in
   numpy.
6. State the cost of the trace term at D = 1024 and the
   sliced alternative.
7. The objective looks great but far-from-data scores are
   wrong. Name the cause and the schedule fix.
8. "Score matching needs no assumptions." Attack with two
   (boundary, smoothness).

## O9: flow matching

1. Define the probability path and the target velocity on
   the toy.
2. Write the conditional flow-matching loss and evaluate
   it at v = 1.8.
3. Explain why regressing the conditional velocity
   recovers the marginal velocity field.
4. Contrast with the DDPM loss: what is learned in each?
5. Sketch the training loop in pseudocode.
6. State the sampling cost: what the ODE solver needs per
   step.
7. Training loss is low but paths cross and samples are
   poor. Name the cause and the coupling fix.
8. "Flow matching removes all diffusion assumptions."
   Attack with two arguments.

## O10: the research critique

1. State the claim: "Our model is the best generative
   model." List the four things "best" could mean.
2. For "best likelihood": state the bound-versus-exact
   trap with the toy numbers.
3. For "best samples": state the precision/recall split
   with the toy numbers.
4. For "best metric": construct the FID-0 counterexample.
5. For "best efficiency": state the compute-matching
   requirement.
6. Design the fair experiment: metrics, budgets, seeds,
   decision rule.
7. Your experiment returns a negative result. State what
   you report and why it still counts.
8. "Negative results are not publishable." Attack with
   the falsification argument.
