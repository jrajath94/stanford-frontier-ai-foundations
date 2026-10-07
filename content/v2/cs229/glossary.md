# glossary.md, cs229

Date: 2026-10-06. Seed entries. Expanded as units land.

- **feature**: one measured input variable. A house has features
  living area and bedroom count.
- **target**: the value to predict. A house has target price.
- **example**: one (feature, target) pair.
- **hypothesis**: the function `h` that maps features to a
  prediction.
- **parameter**: a number inside the hypothesis that training sets.
- **loss**: a score of one wrong prediction. Lower is better.
- **cost**: the average loss over the training set. Training
  minimizes cost.
- **risk**: the average loss over all possible future data. This is
  the true target. Cost is its stand-in.
- **empirical risk**: cost measured on the training set only.
- **overfitting**: the hypothesis memorizes noise in the training
  set, so cost is low but risk is high.
- **underfitting**: the hypothesis is too simple to capture the
  pattern, so both cost and risk are high.
- **likelihood**: the probability the model assigns to the observed
  data. Training often maximizes it.
- **posterior**: probability of parameters given data.
- **margin**: distance of an example from a decision boundary.
- **ELBO**: evidence lower bound, a training target for latent
  variable models.
- **baseline**: the simplest method worth beating. Every experiment
  needs one.

- **MDP**: Markov decision process: states, actions, transitions,
  discount, rewards. The formalism for sequential decisions.
- **Bellman equation**: the recursion linking a state's value to
  its successors' values.
- **value iteration**: repeated Bellman backups to convergence.
- **policy iteration**: alternate exact evaluation and greedy
  improvement.
- **RLVR**: reinforcement learning with verifiable rewards:
  training language models against automatic checkers.
- **verifier**: the automatic checker that produces the RLVR
  reward. Its validity decides what the model learns.
- **GRPO**: group relative policy optimization: advantages from
  comparing completions to the same prompt, no critic.
- **LQR**: linear quadratic regulation: linear dynamics plus
  quadratic costs, solved by the Riccati recursion.
- **Riccati equation**: the backward matrix recursion for the
  LQR cost-to-go.
- **DDP**: differential dynamic programming: iterate
  linearization around a nominal trajectory.
- **Kalman filter**: recursive Gaussian state estimation:
  predict then update with the Kalman gain.
- **REINFORCE**: policy gradient via the log-derivative trick
  on sampled trajectories.
- **baseline**: a state-only function subtracted in policy
  gradients to cut variance without bias.
- **PPO**: proximal policy optimization: clipped likelihood
  ratios keep updates near the old policy.
- **ablation**: an experiment that changes one factor with
  matched budgets to measure its effect.
- **error bar**: the reported spread of a measurement: sample
  size, standard deviation or interval, comparison.
