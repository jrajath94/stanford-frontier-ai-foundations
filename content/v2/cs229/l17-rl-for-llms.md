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
expectation is unchanged (baselines don't bias it) but the
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

The lecture walks the four cases. Advantage positive, ratio 1.5
(action good, new policy already likes it more): clip to 1.2, and
PPO's position is "no need to reinforce anymore": it is already a
good action at higher probability, so zero out the extra push.
Advantage positive, ratio 0.5 (good action, new policy shies away):
unclipped, reinforce it back up. Advantage negative, ratio 1.5
(bad action, new policy likes it): clip, don't let it get worse.
Advantage negative, ratio 0.5: fine, keep suppressing. The pattern:
never let one update move the policy far from the data it trained
on. "Proximal": stay near the old policy.

![PPO clipping](assets/svg/l17-ppo.svg "PPO. The importance ratio r is clipped to [0.8, 1.2]. Good actions already favored get no extra push. Bad updates cannot jump far. Source: original plate for Stanford Frontier AI.")

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
> A: The objective uses r = pi_new/pi_old clipped to [1-eps, 1+eps] (eps ~ 0.2), taking the minimum of raw and clipped times advantage. Four cases. Advantage > 0, r = 1.5: good action already favored. Clip and stop pushing ("no need to reinforce"). Advantage > 0, r = 0.5: good action disfavored. Reinforce normally. Advantage < 0, r = 1.5: bad action favored. Clip the damage. Advantage < 0, r = 0.5: bad action disfavored. Keep suppressing. Net effect: the policy never moves far from the data it learned on in one update.
> Follow-up: Why "proximal"?
> A: Proximal means near. PPO constrains each update to stay proximal to the old policy: the trust region is enforced by the clip instead of a hard constraint. It is the practical descendant of TRPO, which enforced the region exactly and was harder to implement. The clip is the whole trick.

> [!QA]
> Q: Why do verifiable rewards work for reasoning when sparse 0/1 rewards failed in the naive attempt?
> A: Three differences. One, advantages replace raw totals: credit goes to better-than-average trajectories, not lucky ones. Two, PPO stabilizes the updates so the sparse signal accumulates instead of jittering away. Three, scale: dozens of sampled thinking trajectories per problem let statistics separate systematically-good thinking from lucky guesses. The verifier's binary signal is honest (the answer is right or not), and with the variance tamed, honesty suffices. The naive attempt had none of these: raw REINFORCE on single trajectories with 0/1 totals.
> Follow-up: What is reward hacking in RLVR?
> A: The model finds ways to score 1 without reasoning well: pattern-matching the answer format, exploiting verifier bugs, or writing broken chains that stumble onto the right final token. The verifier checks the answer, not the thought, so the thinking can rot while the score shines. Mitigations: stronger verifiers, process supervision (rewarding intermediate steps), and auditing the thinking traces, not just the scores.

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

## Official sources and further reading

**Official:**
- Lecture 17 video, Stanford Online YouTube:
  https://www.youtube.com/watch?v=J7CossjMvEg — Tengyu Ma derives
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

- **CS229 L15:** SFT: the post-training step before RL; PPO
  continues where SFT stops.
- **CS229 L16:** REINFORCE and the MDP: everything this lesson
  stabilizes.
- **CS229 L06:** variance, the estimator kind: baselines as
  variance reduction.
- **CS336:** PPO at scale: rollout farms and the infrastructure
  behind RLVR.
