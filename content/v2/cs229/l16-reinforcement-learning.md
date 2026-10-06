---
page_id: cs229-l16
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 16
nav: "L16 · Reinforcement Learning"
title: "Lecture 16: Reinforcement Learning Basics and Policy Gradient"
summary: "Sequential decisions via the MDP, value functions, and REINFORCE: the policy-gradient algorithm behind LLM post-training."
date: "2026-05-27"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "1:18:55"
video_id: xveNBYVTrqw
video_title: "Lecture 16: Reinforcement Learning"
video_caption: "Original lecture. Tengyu Ma builds the MDP, value functions, and the REINFORCE policy-gradient algorithm."
concepts: [reinforcement-learning, MDP, sequential-decision, policy, value-function, Bellman, policy-gradient, REINFORCE, log-derivative]
sources:
  - tag: video
    label: "Lecture 16 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=xveNBYVTrqw
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: the robot and the charging dock

A robot sits at position 3 on a track numbered 1 to 10. Its dock is
at 10. Each step it may move left or right. Reaching 10 earns reward
+10. Every step costs -1 (batteries drain). The job: a strategy for
acting that maximizes total reward. Nobody shows the robot the
right moves. It must learn from trying.

Supervised learning cannot touch this. There are no labeled
(input, correct action) pairs: the "correct" action at position 3
depends on the whole future. The experience is a history of actions
and rewards. That is the reinforcement learning setup from lecture
1, now built formally.

## The MDP: five pieces

A **Markov decision process** (MDP) is the formal frame. Five
pieces:

**S, the states.** Everything needed to describe the world. Here:
positions 1..10. For a real robot: joint angles, camera images. For
Go: the whole board, 3^361 possible states. The lecture's modeling
note: the state should include everything in theory, but you
abstract: a 3D map of the world is usually left out.

**A, the actions.** What the agent can do. Here: left, right.

**P, the transitions.** P(s'|s,a): the probability of landing in
s' after action a in state s. The world may be slippery: "move
right" succeeds with probability 0.9, stays put with 0.1.

**R, the rewards.** R(s,a): the immediate score. Here: +10 at the
dock, -1 per step.

**Gamma, the discount.** Future rewards count slightly less than
present ones: 0.99^t. This keeps infinite sums finite and encodes
"sooner is better".

**Markov** means the future depends only on the present state, not
the history: P(s'|s,a) captures everything. A **policy** pi(a|s) is
the strategy: the probability of each action in each state. The
goal: the policy maximizing expected total discounted reward.

![MDP](assets/svg/l16-mdp.svg "The Markov decision process. States, actions, transitions, rewards, discount. The robot at 3 chooses left or right to reach the dock at 10. Source: original plate for Stanford Frontier AI.")

## First attempt: be greedy

The naive policy: at each state, take the action with the biggest
immediate reward. At position 3: left gives -1, right gives -1. Tie.
Say it goes left to 2. At 2: left -1, right -1. Wanders. The greedy
agent never reasons that 9 rights in a row earn +10. It chases
immediate reward and misses the dock forever, paying -1 per step
into eternity. Total: negative infinity (discounted: -100).

The failure: rewards are delayed. The action's value is not its
immediate reward but the future it leads to. Greedy is blind to the
future.

## Value functions: price the future

The **value** V(s) is the expected total future reward from state s
under the policy. The **action value** Q(s,a) is the expected total
after taking action a in s, then following the policy. They satisfy
the **Bellman equation**: value now equals immediate reward plus
discounted value later.

```ascii
V(s) = E[ R(s,a) + gamma * V(s') ]
```

Work the toy. Policy: always go right. Deterministic moves,
gamma = 0.9, dock at 10 gives +10, each step -1. V(10) = 0 (done).
V(9) = -1 + 0.9*10 = 8.0. V(8) = -1 + 0.9*8.0 = 6.2. V(7) = -1 +
0.9*6.2 = 4.58. Walking back: V(3) = -1 + 0.9*V(4), and V(4) =
2.46, so V(3) = 1.21. The value at 3 is +1.21: going right is worth
it despite the -1 steps, because the dock pays. Q(3, right) = 1.21,
Q(3, left) = -1 + 0.9*V(2) = -1 + 0.9*(-0.71) = -1.64. Right wins
by the numbers. Values turn delayed rewards into present prices.

## The key question

Values evaluate a policy. But how do we *improve* a parameterized
policy pi_theta (say, a neural network mapping states to action
probabilities) by gradient descent, when the reward is not
differentiable and the future depends on our own changing actions?

## REINFORCE: the log-derivative trick

The objective: J(theta) = expected total reward under pi_theta.
The gradient seems impossible: the expectation is over trajectories
the policy itself generates, and rewards have no derivatives. The
**log-derivative trick** unlocks it:

```ascii
gradient of E[R] = E[ R * gradient of log pi_theta(actions) ]
```

Why it works, in one line: the gradient of a probability is the
probability times the gradient of its log (d p = p * d log p), so
weighting each trajectory's reward by its log-probability gradient
pushes probability toward high-reward trajectories. The algorithm:

```ascii
REINFORCE:
  run the policy, get a trajectory (states, actions, rewards)
  total = sum of discounted rewards
  for each step: theta += alpha * total * grad log pi_theta(action|state)
```

Read it: actions that led to high total reward get more probable.
Actions that led to low reward get less. "Reinforce the good
actions" is the literal algorithm. The lecture's name for the
update: reinforce.

Work the toy. Policy: at state 3, pi(right) = 0.6, pi(left) = 0.4
(one knob: theta = log-odds). Run once: go right, right, ..., reach
dock: total = -1*7 + 10 = 3 (7 steps). grad log pi(right) pushes
theta up: theta += alpha * 3 * (1 - 0.6) = theta + 0.12 (alpha =
0.1). Right gets more probable. Run again: go left first, wander,
total = -12. theta += 0.1 * (-12) * grad: left's probability
falls. Over many trajectories, the policy shifts toward the dock.

![Policy gradient](assets/svg/l16-pg.svg "REINFORCE. Trajectories with high total reward push their actions' probabilities up. Reinforce the good actions. Source: original plate for Stanford Frontier AI.")

## Where it breaks: variance

REINFORCE's gradient is a single trajectory's opinion. One lucky
run (total 3) and one unlucky run (total -12) give violently
different updates for the same policy. The estimator is unbiased
(right on average) but high-variance (wild per sample). Training
jitters. Learning rates must be tiny. Millions of trajectories are
needed. The lecture's second-order note: this variance is the
problem the next lecture's baselines and PPO attack. Also
**on-policy**: every gradient needs fresh trajectories from the
current policy. Old data is stale the moment theta moves. Samples
are single-use, which makes RL sample-hungry in a way supervised
learning is not.

## The honest price

RL buys sequential decision-making and pays in samples and
stability. Millions of trajectories for what supervised learning
does in thousands of examples. Rewards must be designed: the +10/-1
shaping *is* the task specification, and bad shaping teaches bad
behavior (the robot that spins in circles to avoid the -1 step
cost). Credit assignment is hard: which of the 7 rights earned the
+10? REINFORCE blames and credits them equally. And the Markov
assumption is a modeling choice: leave key information out of the
state and the optimal policy becomes unlearnable. The next lecture
pays down the variance bill.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| MDP | Sequential decisions have no (input, label) pairs | S, A, P, R, gamma; Markov: future depends only on present |
| Values + Bellman | Greedy chases -1 steps forever, misses the +10 | V(9)=8.0, V(3)=1.21: delayed rewards priced into the present; Q(3,right)=1.21 beats Q(3,left)=-1.64 |
| REINFORCE | Rewards are not differentiable; trajectories depend on the policy | grad E[R] = E[R * grad log pi]; reinforce good actions; toy: theta += 0.12 after a good run |
| Log-derivative trick | Cannot differentiate through sampling | d p = p * d log p: weight each trajectory by reward times its log-prob gradient |

> [!QA]
> Q: What is an MDP, and what does "Markov" buy you?
> A: States S, actions A, transition probabilities P(s'|s,a), rewards R(s,a), discount gamma. "Markov" means the next state depends only on the current state and action, not the history. It buys tractability: the optimal action at state s needs no memory of how you got there, so policies are functions pi(a|s) of the present only, and value functions satisfy the Bellman recursion. Break it (hide key info from the state) and the frame's guarantees fail.
> Follow-up: Give the five pieces for the robot toy.
> A: S = positions 1..10. A = {left, right}. P: move succeeds with 0.9, stays with 0.1 (slippery). R: +10 at dock (10), -1 per step. Gamma = 0.9. Policy pi(a|s): the strategy. Goal: maximize expected discounted total.

> [!QA]
> Q: Derive the REINFORCE update in words.
> A: Objective J = expected total reward under pi_theta. The log-derivative trick: grad E[R] = E[R * grad log pi_theta(trajectory)]. Since a trajectory's log-probability is the sum of its steps' log-probs, the gradient is: sample a trajectory, compute its total reward, and for each step move theta along grad log pi(action|state) scaled by the total. High-reward trajectories push their actions' probabilities up. In the toy, a dock-reaching run with total 3 raised pi(right|3) via theta += 0.1*3*0.4.
> Follow-up: Why is REINFORCE high-variance?
> A: Each gradient is one trajectory's opinion: the same policy produces totals of 3 and -12 by luck. The estimator averages correctly but swings wildly per sample, so steps must be small and many. Worse, it is on-policy: trajectories go stale when theta moves, so every update needs fresh rollouts. The next lecture's baselines subtract the luck to calm it.

> [!QA]
> Q: What is the Bellman equation, and what is it for?
> A: V(s) = E[R(s,a) + gamma V(s')]: the value of a state equals the immediate reward plus the discounted value of where you land. It is the consistency condition every value function must satisfy, and it turns the global problem (maximize total future reward) into a local one (one-step lookahead with priced futures). In the toy it priced position 3 at +1.21 despite the -1 steps, proving right beats left without simulating to the dock each time.
> Follow-up: What is the difference between V and Q?
> A: V(s) values the state under the policy: what follows from here. Q(s,a) values the state-action pair: what follows if I do a now, then follow the policy. Q lets you choose: pick argmax_a Q(s,a). V lets you evaluate. Bellman links them: V(s) = E_a[Q(s,a)].

## Recap: the whole lesson on one screen

1. **The job.** Robot at 3, dock at 10, +10/-1 rewards. No labels.
   Learn from trying.
2. **The MDP.** S, A, P, R, gamma. Markov: present suffices.
   Policy pi(a|s): the strategy.
3. **Greedy fails.** Chases -1 steps, misses +10 forever.
   Delayed rewards need pricing.
4. **Values.** V(9) = 8.0 down to V(3) = 1.21. Bellman: value now
   = reward + discounted value later.
5. **The key question.** Improve pi_theta by gradient when rewards
   do not differentiate?
6. **REINFORCE.** grad E[R] = E[R grad log pi]. Reinforce good
   actions. Toy: theta += 0.12 on a good run.
7. **Where it breaks.** One trajectory's opinion: wild variance.
   On-policy: samples are single-use. Millions of rollouts.
8. **The honest price.** Sample hunger, reward design is the task,
   credit assignment, Markov as modeling choice.

## Official sources and further reading

**Official:**
- Lecture 16 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=xveNBYVTrqw — Tengyu Ma builds
  the MDP on the robot toy (positions 1-10, dock at 10), derives
  value functions and Bellman, and presents REINFORCE via the
  log-derivative trick.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the formal
  derivations.

**Caveats from these sources.** The robot toy (positions 1-10,
actions left/right, Go's 3^361 states) is the lecture's own
running example. The value arithmetic in this lesson is an original
miniature with the lecture's numbers. The variance and on-policy
costs are the lecture's stated motivations for the next lecture's
methods.

## Connections to the other courses

- **CS229 L01:** the reinforcement paradigm, now formal.
- **CS229 L17:** PPO: taming REINFORCE's variance for LLM
  post-training.
- **CS229 L06:** variance again: the estimator kind, not the
  model kind.
- **CS336:** rollout infrastructure: generating millions of
  trajectories at scale.
