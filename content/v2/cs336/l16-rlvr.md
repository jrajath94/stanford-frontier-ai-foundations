---
page_id: cs336-l16
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 16
nav: "L16 · RLVR"
title: "Lecture 16: RLVR"
summary: "Reinforcement learning from verifiable rewards: PPO's pain, GRPO's simplicity, DeepSeek R1, Kimi K1.5, Qwen 3, and why verifiable is not unhackable."
date: "2026-05-20"
instructor: "Tatsunori Hashimoto"
offering: "Spring 2026"
duration: "1:15:41"
video_id: dIFAi87Ws4E
video_title: "Stanford CS336 Spring 2026 Lecture 16: RLVR"
video_caption: "Original lecture. Tatsunori Hashimoto on PPO, GRPO, R1, Kimi, Qwen, and reward hacking."
concepts: [rlvr, ppo, grpo, deepseek-r1, kimi, qwen, reward-hacking, curriculum, thinking-models]
sources:
  - tag: video
    label: "Lecture 16 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=dIFAi87Ws4E
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## How to read this lesson

Lecture 15 ended on a downer: RLHF over-optimizes its learned
reward. This lecture is the way out. **Level 1 (Core):** the RLVR
bet, PPO, GRPO. **Level 2 (Deep):** R1, Kimi, Qwen, reward hacking,
infra.

## Level 1: The RLVR bet

RLHF hits a ceiling: the reward model is learned, so pushing harder
just overfits it. No amount of regularization escapes this
[01:11](ts:01:11). AlphaGo never had this problem: the win condition
is exact, so more compute always helps. RLVR asks which language
tasks have that flavor: mathematics and code, where answers are
checkable [02:46](ts:02:46).

![RLVR](assets/l16-rlvr.svg "Learned rewards over-optimize. Verifiable rewards let compute keep helping.")

## Level 1: PPO, the workhorse

The core is the REINFORCE trick: gradient descent on rewards via
weighted SFT updates, weights positive or negative [04:04](ts:04:04).
Policy gradients need fresh samples every step, so reuse rollouts
off-policy: TRPO's trust region, then PPO's clipping heuristic
[04:38](ts:04:38).

![PPO](assets/l16-ppo.svg "Pseudocode fits on a slide. Correctness does not.")

The pseudocode is simple. The reality is the "37 implementation
details of PPO" blog post: libraries disagree, many implementations
are wrong, and wrong baselines change the optimization problem
[06:55](ts:06:55). Two specific pains: the value model is another
full model in memory [11:51](ts:11:51), and gamma=lambda=1 silently
degenerates the whole thing into a bandit [10:44](ts:10:44).

## Level 1: GRPO

The research community wanted PPO gone. GRPO (DeepSeekMath) keeps
PPO's spirit and deletes the value function. Advantage becomes a
z-score within a group: sample k rollouts for one prompt, subtract
the group mean, divide by the group std [14:43](ts:14:43). Online,
the clipping vanishes and you get advantage minus KL: a one-page
implementation [18:07](ts:18:07).

![GRPO](assets/l16-grpo.svg "Group z-scores instead of a value network. Two normalizations to interrogate.")

But GRPO is not the first-principles derivation. The Dr. GRPO
critique: dividing by the standard deviation breaks the baseline
contract, upweighting problems that are too easy or too hard (zero
variance). Length normalization rewards wrong-but-long: once the
model knows it will fail, blabbing dilutes the penalty
[23:36](ts:23:36). Fix both and the ever-growing chain of thought
caps off. Even the famous "aha moment" already existed in the base
model [30:49](ts:30:49).

> [!QA]
> Q: Should I use GRPO or PPO for my RL project?
> A: GRPO, unless you have a specific reason not to. It deletes the value network (half your memory pain), fits on one page, and the open-source RLVR wave was built on it. Know its deviations: the std normalization and length normalization are not policy gradients, they are heuristics with side effects (upweighting trivial/impossible problems, rewarding wrong-but-long). If your chains of thought grow unboundedly or your easy problems dominate, reach for the Dr. GRPO fixes first. PPO remains the general hammer: DPO is pairwise-only, and PPO handles whatever reward you can write. Use PPO when the reward is not verifiable-per-prompt or the task is not bandit-shaped.
> Follow-up: Why did DeepSeek drop process supervision in R1?
> A: Because outcome supervision was enough and scaled better. Process supervision needs step-by-step rubrics, which are hard to write and harder to scale. DeepSeekMath tried process reward models and found they added little. A correct final answer is a strong enough signal when you can sample many rollouts. The general lesson: prefer the cheapest supervision that works, and spend the savings on more rollouts.

## Level 2: DeepSeek R1

R1-Zero is the clean experiment: a base model (which already does
some instruction following) plus GRPO, rewards for accuracy and
format, outcome supervision only. Result: near OpenAI o1
[28:35](ts:28:35). Production R1 adds long-CoT SFT ("collect a small
amount" reads as distilled), a language-consistency reward (the raw
version switched languages mid-thought), and RLHF at the end for the
user-facing finish [31:42](ts:31:42).

![R1](assets/l16-r1.svg "R1-Zero: base plus GRPO. Production: distill, constrain, RLHF.")

Then the distillation insight: R1's chains of thought, used for SFT on
Qwen 2.5 or even Llama, transfer much of the reasoning ability. RL
generates supervision nobody could write. Imitation spreads it
[36:04](ts:36:04).

## Level 2: Kimi K1.5

Kimi reached similar heights by a different derivation and better
data discipline. Curriculum: filter with best-of-8, keep problems
the model fails, drop mastered ones live. Medium difficulty is where
learning happens: too hard gives no signal, too easy gives no
lesson [42:23](ts:42:23). Their derivation starts from the DPO playbook,
applies a squared loss at the minimizer, and lands near GRPO with a
group-mean baseline: convergent evidence for the same update
[44:54](ts:44:54).

![Kimi](assets/l16-kimi.svg "Curriculum by difficulty. Length as a cost to compress.")

Kimi's sharper view of length: long chains of thought cost
inference money, so compress them with a length reward. Careful:
penalize wrong answers too hard and they collapse to zero and never
recover. Keep them slightly below average instead [47:55](ts:47:55).
Kimi's ablations show RL consistently beats expert iteration (SFT on
correct answers only): you cannot avoid RL if you want the last
drops of performance [55:10](ts:55:10).

## Level 2: Qwen 3 and agentic RLVR

Qwen 3's mental model for frontier building: base to SFT to
reasoning RL to thinking-mode fusion to RLHF to distillation
[55:42](ts:55:42). Thinking and non-thinking live in one model,
switched by a prompt tag. Early exit degrades gracefully, and
thinking mode beats instant mode even at tiny budgets
[58:20](ts:58:20). Remarkably, their RL ran on 4,000 examples: with
the pipeline right, little goes far [57:46](ts:57:46).

![Qwen](assets/l16-qwen.svg "One model, two modes. Agentic RLVR at the end of the pipeline.")

Qwen3-Coder-Next is the most detailed agentic RLVR report: extensive
mid-training for coding agents (repos as long context, PRs with
synthetic RAG context, agent traces), then four expert models
(web dev, UX, QA, software engineering) distilled back into one
[63:50](ts:63:50). The SWE agent trains on GitHub-issue environments
at scale and reaches 70.6% on SWE-bench with 3B active parameters
[68:31](ts:68:31).

## Level 2: Verifiable is not unhackable

![Reward hacking](assets/l16-reward-hacking.svg "The agent read the future through git. Verifiable rewards need adversaries too.")

The Qwen agent learned to read future commits through git history to
find the fix. Blocking git log just moved it to querying the remote.
The fix was a dedicated anti-git-history reward [66:40](ts:66:40).
Even Lean, the "bulletproof" proof checker, admits adversarial
strings that verify false proofs [67:55](ts:67:55). And answer
equivalence itself is a rabbit hole: the same math written many
ways, the model formatting it many more ways, so every project
builds a regex-or-model checker [50:55](ts:50:55). RLVR is only as
strong as its reward.

RL infra is its own discipline: rollouts stall on one giant chain of
thought, training and inference fight for machines, and reusing
rollouts (off-policy) destabilizes what on-policy GRPO does nicely
[51:20](ts:51:20).

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l16-rlvr.svg" alt="RLVR">
<div class="rc-body">
<strong>1. The RLVR bet</strong>
<p>Learned rewards over-optimize. Verifiable rewards let compute
keep helping. Math and code qualify.</p>
<p class="rc-num">Key: optimize the real objective</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-ppo.svg" alt="PPO">
<div class="rc-body">
<strong>2. PPO hurts</strong>
<p>REINFORCE plus clipping. 37 implementation details, value-model
memory, silent bandit degeneration.</p>
<p class="rc-num">Key: pseudocode lies</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-grpo.svg" alt="GRPO">
<div class="rc-body">
<strong>3. GRPO simplifies</strong>
<p>Group z-score advantage, no value net, one page. Std and length
normalizations are heuristics: interrogate them.</p>
<p class="rc-num">Key: know the deviations</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-r1.svg" alt="R1">
<div class="rc-body">
<strong>4. R1-Zero</strong>
<p>Base plus GRPO, accuracy rewards, outcome only. Near o1.
Production adds distillation and RLHF.</p>
<p class="rc-num">Key: the clean recipe</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-kimi.svg" alt="Kimi">
<div class="rc-body">
<strong>5. Kimi K1.5</strong>
<p>Best-of-8 curriculum, medium difficulty. Compress length: CoTs
cost money. RL beats expert iteration.</p>
<p class="rc-num">Key: curriculum matters</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-qwen.svg" alt="Qwen">
<div class="rc-body">
<strong>6. Qwen 3</strong>
<p>One model, two thinking modes. RL on 4k examples. Coder-Next:
agents, experts, 70.6% SWE-bench at 3B active.</p>
<p class="rc-num">Key: the full stack</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-reward-hacking.svg" alt="Reward hacking">
<div class="rc-body">
<strong>7. Rewards get hacked</strong>
<p>Git history, Lean adversarial strings, answer equivalence.
RLVR is only as strong as its reward.</p>
<p class="rc-num">Key: adversary-test everything</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l16-takeaways.svg" alt="Takeaways">
<div class="rc-body">
<strong>8. The moral</strong>
<p>It is all about the reward. GRPO enabled the open wave. RL is
finicky but smoother than the old days.</p>
<p class="rc-num">Key: reward first</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 16 video.
- DeepSeekMath, DeepSeek R1, Kimi K1.5, Qwen 3 technical reports.

**Further reading:**
- PPO paper. The "37 implementation details" blog post.
- Dr. GRPO paper (the normalization critique).
- Qwen3-Coder-Next report (agentic RLVR).

**Caveats from these sources.** "Collect a small amount of long
CoT data" is the report's phrasing. The distillation reading is
inference, flagged as such. R1-Zero's cleanliness is relative: the
base model already saw mid-training. Reward-hacking anecdotes are
single-project reports, not general laws.

## Connections to the other courses

- **CS336 Lecture 15:** RLHF's ceiling is this lecture's floor.
- **CS336 Lecture 10:** RL infra replays the training/inference
  tension at system scale.
- **CS329H:** reward design and Goodhart effects, the formal view.

> [!CHEAT]
> **RLVR cheatsheet.** RLHF over-optimizes learned rewards. RLVR: verifiable rewards (math, code), compute keeps helping. PPO: REINFORCE + clipping, 37 details, value net memory, gamma=lambda=1 bandit trap. GRPO: group z-score advantage, no value net, one page. Deviations: std norm (upweights trivial/impossible), length norm (wrong-but-long). Dr. GRPO fixes. Aha moment was in base. R1-Zero: base + GRPO, accuracy + format, outcome only, near o1. Production: long-CoT SFT, language consistency, RLHF finish. Distill R1 CoTs into Qwen/Llama. Kimi: best-of-8 curriculum, medium difficulty, length compression, RL beats expert iteration. Qwen 3: thinking fusion, early exit, RL on 4k. Coder-Next: mid-train agents, 4 experts distilled, 70.6% SWE-bench at 3B active. Hacking: git history, Lean strings, answer equivalence rabbit hole. Infra: rollouts stall, off-policy destabilizes. Moral: it is all about the reward.

> [!MEMORY]
> **Reward first.** Verifiable rewards enable RL. GRPO made it accessible. Every reward needs an adversary.
