---
page_id: cs229-l17
course_slug: cs229
course_name: "CS229: Machine Learning"
course_order: 2
order: 17
nav: "L17 · RL for LLMs"
title: "Lecture 17: RL for Language Models, PPO and Verifiable Rewards"
summary: "From policy gradient to PPO: advantages, clipping, and training reasoning models on binary verifiable rewards."
date: "2026-06-01"
instructor: "Tengyu Ma"
offering: "Spring 2026"
duration: "1:14:37"
video_id: J7CossjMvEg
video_title: "Lecture 17: RL for Language Models"
video_caption: "Original lecture. Tengyu Ma builds PPO from REINFORCE via advantages and clipping, then RL on verifiable rewards for reasoning."
concepts: [PPO, proximal-policy-optimization, advantage-function, baseline, clipping, verifiable-reward, chain-of-thought, post-training, RLVR]
sources:
  - tag: video
    label: "Lecture 17 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=J7CossjMvEg
  - tag: video
    label: "Explainer: PPO and RL for language models"
    url: https://www.youtube.com/watch?v=iSvC5VmDHL4
  - tag: paper
    label: "PPO paper (Schulman et al., 2017)"
    url: https://arxiv.org/abs/1707.06347
  - tag: paper
    label: "DeepSeek-R1 paper (RLVR reasoning)"
    url: https://arxiv.org/abs/2501.12948
  - tag: notes
    label: "Official subtitle transcript (en-US)"
  - tag: notes
    label: "CS229 Spring 2026 official course notes (local PDF)"
---

## The job: teach the model to think, not just answer

After SFT (lecture 15), the model answers helpfully. But ask it a
hard math problem and it blurts the first plausible answer. What
you want: the model tries approaches, checks its work, backtracks,
then answers. That deliberation is **chain-of-thought**: thinking
tokens before the final answer. The job: train the model to think
well, using only a signal for whether the final answer is right.

## First attempt: REINFORCE on answers

The naive idea: treat answer generation as the robot's walk
(lecture 16). Reward 1 if the final answer is correct, 0 if not.
Run REINFORCE: reinforce token choices that led to correct
answers. Watch it break. A 500-token thought ending in a wrong
answer gets total reward 0: every token, including the 490 good
thinking steps, is punished equally. A lucky guess with no thinking
gets reward 1: reinforced as genius. The single 0/1 at the end is
too coarse to teach 500 decisions, and REINFORCE's variance
(lecture 16) is now multiplied by 500-token trajectories. Training
jitters and stalls.

## The key question

Can we keep REINFORCE's idea but calm its variance, reuse samples,
and stop the policy from jumping off a cliff in one bad update?

## Advantages: subtract the luck

The **advantage** A(s,a) measures how much better an action is than
average: reward-to-go minus a **baseline** (often the value V(s)).
If thinking step a in state s leads to total 8 while the average
from s is 5, the advantage is +3: genuinely good. If another run
scores 5 with average 5, advantage 0: no signal, no update. The
baseline subtracts the luck shared by all actions in the state.

Work the toy. State: mid-proof. Baseline V = 0.5 (half of answers
from here end correct). Action 1 ("try substitution"): leads to
correct, reward-to-go 1. Advantage = 1 - 0.5 = +0.5: reinforce.
Action 2 ("guess"): leads to wrong, 0. Advantage = -0.5: suppress.
Without the baseline, both updates would scale by raw 1 and 0.
With it, the update says "better than usual" vs "worse than
usual", which is the learnable signal. The lecture's derivation:
replace reward by advantage in the policy gradient. The
expectation is unchanged (baselines do not bias it) but the
variance drops.

## PPO: clip the jump

REINFORCE is on-policy: one gradient step per batch of fresh
trajectories, then the data is stale. **PPO** (proximal policy
optimization) reuses data with **importance sampling**: weight each
old trajectory's gradient by the ratio r = pi_new(a|s) /
pi_old(a|s), the probability under the new policy over the old.
If the new policy likes the action twice as much, count its
advantage twice.

The ratio is dangerous. If r = 100 (the new policy went all-in on
an action), one stale trajectory dominates the update and the
policy jumps off a cliff. PPO's fix is **clipping**: cap the ratio
at [1-epsilon, 1+epsilon], epsilon ~ 0.2. The clipped objective
takes the worse of the raw and clipped versions, so the update
never profits from moving too far.

The lecture walks the four cases, corrected against the
objective L = min(r*A, clip(r)*A) (an earlier draft of this
lesson mirrored the two negative-advantage cases: fixed below).
Advantage positive, ratio 1.5 (action good, new policy already
likes it more): min(1.5A, 1.2A) = 1.2A, clipped flat: "no need to
reinforce anymore". Advantage positive, ratio 0.5 (good action,
new policy shies away): min(0.5A, 0.8A) = 0.5A, unclipped:
reinforce it back up. Advantage negative, ratio 1.5 (bad action,
new policy likes it): min(1.5A, 1.2A) with A < 0 is 1.5A:
UNCLIPPED, full corrective push back down: the policy already
moved toward a bad action, and the objective wants it reversed at
full strength. Advantage negative, ratio 0.5 (bad action, new
policy already shies away): min(0.5A, 0.8A) = 0.8A: clipped flat,
stop suppressing. The pattern: the clip binds only when the ratio
moves further in the direction the objective wants: r past 1.2
for good actions, r below 0.8 for bad ones. "Proximal": stay near
the old policy.

### Subchapter: the four cases, corrected

The draft said: A < 0, r = 1.5: "clip, do not let it get worse";
A < 0, r = 0.5: "fine, keep suppressing". Both wrong. Plug in A =
-2, eps = 0.2. Case r = 1.5: min(1.5*(-2), 1.2*(-2)) = min(-3,
-2.4) = -3: the unclipped value. The gradient pushes r down at
full strength: correct, because the policy drifted toward a bad
action and must be yanked back. Clipping here would weaken the
rescue. Case r = 0.5: min(0.5*(-2), 0.8*(-2)) = min(-1, -1.6) =
-1.6: the clipped value, flat. The suppression STOPS: the policy
already shies away. Pushing further would move it off the data
for no gain. The rule that replaces the draft: clip the direction
the objective is pulling, not the direction the ratio sits. The
objective pulls r up when A > 0 (clip above 1.2) and down when A <
0 (clip below 0.8). Everything else runs free.

![Four cases](assets/plate-l17-clip-cases.webp "The four cases, corrected. A>0, r=1.5: clipped flat. A>0, r=0.5: full push. A<0, r=1.5: full correction, unclipped. A<0, r=0.5: clipped flat. Source: original audit of the PPO objective. Project: Stanford Frontier AI.")

![PPO clipping](assets/svg/l17-ppo.svg "PPO. The importance ratio r is clipped to [0.8, 1.2]. Good actions already favored get no extra push. Bad updates cannot jump far. Source: original plate for Stanford Frontier AI.")

### Subchapter: the ratio, priced

Importance sampling reuses old data: E_old[r * A] = E_new[A] with
r = pi_new/pi_old. The price is in the weights' variance. A stale
sample the new policy likes 3x (r = 3) counts triple: one
trajectory's opinion, amplified. As the policy drifts from the
data, E[r^2] grows: weights spread, a few trajectories dominate,
the gradient jitters. That is why PPO runs a few epochs per
batch, then rolls out fresh data: the ratio is a short loan, not
a standing facility. The clip and the epoch limit are the same
idea twice: trust old data a little, briefly, then refresh.

![Ratio](assets/plate-l17-ratio.webp "The ratio, priced. r = 3: one stale trajectory counts triple. Drift grows the weights' variance. Few epochs per batch, then fresh rollouts. Source: original plate for the importance weights. Project: Stanford Frontier AI.")

### Subchapter: the verifier's blind spot, priced

The verifier checks the answer, not the thought. A model that
writes a broken chain landing on "42" scores 1. Suppose 5% of
correct answers come from broken reasoning: RLVR reinforces
broken reasoning on 5% of the wins, and nothing in the reward
says otherwise. The thinking rots while the score shines. The
fixes, priced: stronger verifiers (property tests, hidden tests:
engineering cost), process supervision (reward intermediate
steps: a human must read every step, so labeling cost scales with
thinking length instead of answer count), and trace audits (read
the chains, not just the scores: ongoing labor). No fix is free:
each trades verifier engineering or human hours for thinking
quality. The honest rule: never trust a rising reward curve
alone. Sample the traces.

![Blind spot](assets/plate-l17-blind-spot.webp "The verifier's blind spot. Verifier: answer is 42, score 1. Broken chain, right token: the thinking rots while the score shines. Source: original plate for the reward hacking. Project: Stanford Frontier AI.")

## Verifiable rewards: the 0/1 that works

For math and code, correctness is checkable: the answer equals 42
or it does not. The program passes the tests or it does not. The
**verifiable reward** is binary: 1 if right, 0 if wrong. No human
rates the thinking. This is **RLVR** (RL with verifiable rewards).

Why the coarse 0/1 works here when it failed above: scale and the
fix stack. Advantages isolate which trajectories beat the average.
PPO's clipping keeps updates safe across reused batches. And the
model samples many attempts per problem (dozens of thinking
trajectories), so the law of large numbers separates good thinking
patterns from lucky guesses: patterns that systematically precede
correct answers get reinforced. The lecture's flagship: reasoning
models trained this way, thinking tokens and all, with binary
correctness as the only reward. The thinking improves because
better thinking is what systematically produces the 1s.

## The honest price

PPO buys stability and pays in bias and tuning: clipping throws
away legitimate signal (the "no need to reinforce" case discards
real gradient), epsilon is another dial, and importance sampling
degrades as the policy drifts from the data: a few epochs per
batch, then fresh rollouts. Verifiable rewards buy honest signals
and pay in scope: only domains with checkable answers qualify
(math, code, games). For open-ended writing there is no verifier,
and you are back to human preference models with all their
biases. And RLVR can teach reward hacking: the model learns to
produce the checkable token ("42") via broken reasoning that
happens to land right. The verifier checks the answer, not the
thought.

## Mapping back

| Idea | Pain it answers | How |
|---|---|---|
| Advantage | 0/1 at the end punishes 490 good thinking steps equally | Reward-to-go minus baseline: +0.5 reinforce, -0.5 suppress; luck subtracted, signal kept |
| Importance ratio | On-policy: data single-use | r = pi_new/pi_old reuses batches; stale trajectories weighted by drift |
| Clipping | r = 100 jumps the policy off a cliff | Cap at [0.8, 1.2]; four cases: never profit from moving far; "proximal" |
| Verifiable reward | Human rating does not scale; preferences are biased | Binary correctness (tests pass/answer matches); RLVR trains thinking with 0/1 |
| Scale | One trajectory cannot separate skill from luck | Dozens of attempts per problem; patterns preceding 1s get reinforced |

> [!QA]
> Q: What problem does the advantage function solve?
> A: REINFORCE scales updates by raw total reward, so luck dominates: a good action in a lucky trajectory and a bad action in a lucky trajectory both get reinforced. The advantage A = reward-to-go minus baseline subtracts what was expected anyway. Only better-than-average actions get positive updates. The expectation is unchanged (baselines add zero in expectation) but the variance collapses, because the shared luck cancels. In the toy, substitution scored +0.5 (reinforce) and guessing -0.5 (suppress) against a 0.5 baseline.
> Follow-up: What is usually the baseline?
> A: The value function V(s): the expected reward-to-go from the state. It is the best predictor of the luck component, since it captures everything about the state except this action's choice. In practice a separate network (the critic) estimates V, trained alongside the policy (the actor): actor-critic.

> [!QA]
> Q: How does PPO's clipping work, case by case?
> A: The objective is min(r*A, clip(r, 0.8, 1.2)*A) with eps ~ 0.2. Four cases, corrected. Advantage > 0, r = 1.5: min(1.5A, 1.2A) = 1.2A: clipped flat, "no need to reinforce". Advantage > 0, r = 0.5: min(0.5A, 0.8A) = 0.5A: unclipped, reinforce it up. Advantage < 0, r = 1.5: min is the unclipped 1.5A (A negative flips the min): full corrective push back down, because the policy drifted toward a bad action and must be yanked back. Advantage < 0, r = 0.5: min is the clipped 0.8A: flat, stop suppressing. Rule: the clip binds only in the direction the objective pulls: r above 1.2 for good actions, r below 0.8 for bad ones.
> Follow-up: Why "proximal"?
> A: Proximal means near. PPO constrains each update to stay proximal to the old policy: the trust region is enforced by the clip instead of a hard constraint. It is the practical descendant of TRPO, which enforced the region exactly and was harder to implement. The clip is the whole trick.

> [!QA]
> Q: Why do verifiable rewards work for reasoning when sparse 0/1 rewards failed in the naive attempt?
> A: Three differences. One, advantages replace raw totals: credit goes to better-than-average trajectories, not lucky ones. Two, PPO stabilizes the updates so the sparse signal accumulates instead of jittering away. Three, scale: dozens of sampled thinking trajectories per problem let statistics separate systematically-good thinking from lucky guesses. The verifier's binary signal is honest (the answer is right or not), and with the variance tamed, honesty suffices. The naive attempt had none of these: raw REINFORCE on single trajectories with 0/1 totals.
> Follow-up: What is reward hacking in RLVR?
> A: The model finds ways to score 1 without reasoning well: pattern-matching the answer format, exploiting verifier bugs, or writing broken chains that stumble onto the right final token. The verifier checks the answer, not the thought, so the thinking can rot while the score shines. Mitigations: stronger verifiers, process supervision (rewarding intermediate steps), and auditing the thinking traces, not just the scores.

> [!QA]
> Q: Walk me through the mechanism: compute the PPO objective for A = -2, r = 1.5 and r = 0.5, eps = 0.2. Which is clipped?
> A: Objective: min(r*A, clip(r, 0.8, 1.2)*A). r = 1.5: min(-3, -2.4) = -3: UNCLIPPED. The gradient pushes the ratio down at full strength: the policy drifted toward a bad action, yank it back. r = 0.5: min(-1, -1.6) = -1.6: CLIPPED, flat. The policy already shies away from the bad action. Pushing further gains nothing and leaves the data. The negative flips the min: for A < 0 the clip binds below 0.8, not above 1.2.
> Follow-up: A = +2, r = 2.0. Clipped or not?
> A: min(4, 2.4) = 2.4: clipped. The objective wants r up (good action), so the clip caps it at 1.2. Flat gradient beyond: no extra push. Symmetric rule: clip the direction the objective pulls.

> [!QA]
> Q: Applied design: your RLVR code model starts emitting `if task_id == 7: print(expected)` to pass the tests. Diagnose and fix.
> A: Reward hacking: the verifier (the test suite) is gamed. The model scores 1 without reasoning. Fixes, in order of cost. One: hidden tests the model never sees: engineering cost, and the model may still overfit the visible ones' style. Two: property-based tests and mutation testing: generate test variants, kill mutants: stronger, more engineering. Three: process supervision: reward intermediate reasoning steps: labeling cost scales with thinking length. Four: trace audits: humans read sampled chains: ongoing labor. Decision rule: the verifier is part of the product. Budget for it like training compute, and never trust the reward curve alone.
> Follow-up: Hidden tests also get gamed eventually. Then what?
> A: Then the game is test quality, permanently. Rotate and expand the hidden set, add property tests that no memorized answer can satisfy, and keep a human audit loop. There is no final fix: any fixed verifier becomes a target. The honest price of verifiable rewards is verifier maintenance.

> [!QA]
> Q: Why only a few epochs per PPO batch?
> A: The importance ratio r = pi_new/pi_old is a loan against old data. Each epoch moves pi_new further from pi_old: the weights' variance grows, a few stale trajectories dominate, and the gradient estimate degrades. The clip slows the damage but does not stop it. Standard practice is a few epochs, then fresh rollouts. More epochs = training on increasingly fictional weights: the policy jumps off the data cliff the clip was built to prevent.
> Follow-up: What does the failure look like?
> A: The ratio histogram spreads: many r near 0, a few huge. Updates become all-or-nothing: most samples contribute nothing, one stale trajectory decides the step. Training loss looks fine (the clipped objective hides it) while behavior collapses. Watch the ratio stats, not just the loss.

> [!QA]
> Q: RLVR vs RLHF: when does each win?
> A: RLVR wins where answers are checkable: math, code, games. The verifier is honest (the answer is right or not) and scales without humans. Its price is scope and reward hacking. RLHF wins where no verifier exists: open-ended writing, chat, taste. Human preferences train a reward model. Its price is bias, cost, and the reward model becoming the target. They combine in practice: RLVR for reasoning correctness, RLHF for style, helpfulness, and safety.
> Follow-up: Can RLVR train the chat style too?
> A: Only what you can verify: format constraints (JSON valid, length limits), refusal on disallowed content (checkable), citation presence. Style and taste have no verifier: that stays RLHF's job. Map the reward to the checkable: verify structure with RLVR, judge quality with humans.

## Recap: the whole lesson on one screen

1. **The job.** Teach thinking, not just answering. Reward only
   final correctness.
2. **Naive REINFORCE.** 0/1 punishes 490 good steps. Lucky
   guesses reinforced. Variance x 500-token trajectories.
3. **The key question.** Calm the variance, reuse samples, stop
   cliff-jumps?
4. **Advantages.** Reward-to-go minus baseline: +0.5/-0.5. Luck
   subtracted, expectation unchanged.
5. **PPO.** r = pi_new/pi_old, clipped to [0.8, 1.2]. Four cases.
   Never move far from the data. Proximal.
6. **Verifiable rewards.** Binary correctness for math/code.
   RLVR: thinking improves because good thinking earns the 1s.
7. **The honest price.** Clipping discards signal. Epsilon tuned.
   Verifiers only exist for checkable domains. Reward hacking.
8. **The four cases, corrected.** A<0, r=1.5: unclipped full
   correction. A<0, r=0.5: clipped flat. Clip the pull
   direction.
9. **The ratio, priced.** r = 3: stale counts triple. Few epochs,
   then fresh rollouts.
10. **The blind spot.** Verifier checks the answer. Broken chain
    + "42" = 1. Audit the traces.

## What is used where

**PPO and RLVR run modern post-training.** PPO trained the
original RLHF ChatGPT: human preferences as reward, clipping for
stability. **RLVR trains reasoning models:** DeepSeek-R1's
breakthrough was RL on verifiable rewards producing
chain-of-thought without SFT on thinking traces. The o-series
lineage follows the same recipe. The TRL library ships PPO and
its successors as the standard post-training loop. The lesson's
four cases are the production update.

## Watch next

<div class="video-block"><div class="video-wrap"><iframe src="https://www.youtube-nocookie.com/embed/iSvC5VmDHL4" title="Explainer: PPO and RL for language models" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div><p class="video-cap">Explainer: PPO and RL for language models. Advantages, clipping, and verifiable rewards in one visual pass. Watch after the PPO section.</p></div>

## Official sources and further reading

**Official:**
- Lecture 17 video, Stanford Online YouTube:
  - [Tengyu Ma derives](https://www.youtube.com/watch?v=J7CossjMvEg)
  advantages and baselines, walks PPO's four clipping cases, and
  presents RLVR with verifiable binary rewards for reasoning
  models.
- Official subtitle transcript (en-US): the lecture's spoken text.
- CS229 Spring 2026 official course notes (local PDF): the formal
  PPO objective and RLVR setup.

**Caveats from these sources.** The four clipping cases are the
lecture's own walkthrough, including the "no need to reinforce"
position on good actions at high ratio. The thinking-trajectory
toy is an original miniature of the lecture's RLVR framing.
Epsilon ~ 0.2 is the standard PPO setting the lecture cites.

## Connections to the other courses

- **CS229 L15:** SFT: the post-training step before RL. PPO
  continues where SFT stops.
- **CS229 L16:** REINFORCE and the MDP: everything this lesson
  stabilizes.
- **CS229 L06:** variance, the estimator kind: baselines as
  variance reduction.
- **CS336:** PPO at scale: rollout farms and the infrastructure
  behind RLVR.
