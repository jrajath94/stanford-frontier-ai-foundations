---
page_id: cs336-l15
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 15
nav: "L15 · Post-training"
title: "Lecture 15: Post-training"
summary: "From GPT-3 to ChatGPT: SFT data and its pitfalls, safety, mid-training, RLHF, annotators, model-based feedback, PPO, and DPO."
date: "2026-05-18"
instructor: "Tatsunori Hashimoto"
offering: "Spring 2026"
duration: "1:19:45"
video_id: 2oH6PWPrYFo
video_title: "Stanford CS336 Spring 2026 Lecture 15: Post-training"
video_caption: "Original lecture. Tatsunori Hashimoto on SFT, RLHF, annotators, and DPO."
concepts: [post-training, sft, rlhf, ppo, dpo, annotation, safety-tuning, mid-training, hallucination]
sources:
  - tag: video
    label: "Lecture 15 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=2oH6PWPrYFo
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## The problem: GPT-3 could not follow instructions

A strong base model has limited utility. GPT-3 could do copywriting.
ChatGPT followed long programmatic prompts. The gap between them is
post-training: SFT demonstrations plus RL on preferences
[00:51](ts:00:51). Pre-training builds the primordial soup: every
behavior the model might ever show. Post-training extracts the
behaviors you want: instruction following, helpfulness, safety. It is
artisanal, messy, and data-driven. Pre-training is the indispensable
base, but instruction following must be extracted explicitly
[02:30](ts:02:30).

```ascii
pre-train : the base, indispensable, limited utility
SFT       : demonstrations, teaches instruction following
RLHF      : preferences, aligns to what users want
```

## First attempt: show it good examples

**Supervised fine-tuning** (SFT) is the first extraction step. Take
(prompt, good response) pairs and train the model to predict the
response given the prompt: the same next-token objective, now on
demonstrations of desired behavior. The model learns the shape of a
helpful answer.

The data has a short, instructive history.

![SFT history](assets/l15-sft-data-history.svg "FLAN to agentic data: the SFT timeline.")

FLAN: all NLP benchmarks as tasks. Smart, but unnatural and low
quality: summaries were short and sometimes hallucinated.
**Self-instruct**: models generate their own data. **Alpaca and
Vicuna**: distill ChatGPT traces, which reliably induced ChatGPT-like
behavior. **Open Assistant**: crowdsourced experts, Wikipedia-style,
stalled around 10k examples. **WizardLM and Tulu3**: clever synthetic
generation. Now: agentic SFT with tool calls and to-do lists as
supervised targets (Nemotron) [07:31](ts:07:31). Three shifts:
chattiness, expert annotators, tool use [18:04](ts:18:04).

Correctness is subtle: bad responses teach bad behavior, but
pre-training generalization lets models learn instruction following
from surprisingly strange data [10:27](ts:10:27).

### The mask: what SFT actually trains

The loss does not apply to the whole example. The prompt tokens
are masked: no gradient flows through them. Only the response
tokens teach. The model learns "given this user text, produce this
assistant text": it never learns to predict the user.

Work the alternative. Unmasked SFT: the model also predicts the
user's questions. It learns the distribution of user prompts,
including their errors and biases. At inference it plays both
roles: it completes the user turn instead of answering. The mask
is what makes SFT a conditional model instead of a continuation
model. Every instruction-tuned model since InstructGPT uses it.
The implementation is one line: multiply the loss by a mask that
is 1 on response tokens, 0 elsewhere.

![SFT masking](assets/media-generation-cs336-l15-sft-masking-0-4181364d-5c78-4714-81c0-14e3d2fb4345.webp "Loss flows only through the response tokens. Source: original. Project: Stanford Frontier AI.")

> [!QA]
> Q: Walk me through SFT loss masking. Why mask the prompt?
> A: The training example is (user tokens, assistant tokens). The forward pass runs over both: the model needs the user tokens as context. The loss is computed only on the assistant tokens: cross-entropy between the model's predictions and the true assistant tokens, zeroed out on the user tokens. The gradient therefore updates the model to produce the response given the prompt, never to produce the prompt itself. Without the mask, the model would also learn to generate user questions: at inference it might role-play the user instead of answering. The mask is the difference between "continue this text" and "respond to this person." Every SFT implementation does this. Forgetting it is a classic bug.
> Follow-up: Do you mask the system prompt too?
> A: Yes. The system prompt is input, not target. Same rule: loss only on tokens the assistant should learn to produce. The general principle: the loss teaches exactly the tokens you want the model to generate at inference. Everything else is context.

![SFT pitfalls](assets/l15-sft-pitfalls.svg "Style is not capability. Tail knowledge teaches hallucination.")

## Where SFT breaks: two demonstrations

### Break 1: style is not capability

Bullet lists and long answers win AlpacaEval while changing no
benchmark. Engagement signals shift without capabilities shifting.
Control style separately [19:43](ts:19:43). Work the trap: SFT on
chatty formatting teaches the model to sound helpful. The evals that
measure sounding helpful go up. The evals that measure being helpful
do not move. You optimized the wrapper, not the contents.

### Break 2: tail knowledge teaches hallucination

An Open Assistant response teaches two things at once: the citation
format and the fact. The model generalizes the format and
hallucinates references. Training on facts the model does not know
teaches it to emit unknown knowledge [22:30](ts:22:30). Work the
mechanism: the SFT example contains a citation the base model never
saw in pre-training. The gradient says "produce citations like this."
The model cannot distinguish "cite in this format" from "know this
fact," so it emits confident citations to nothing. John Schulman's
argument: RL fixes this because calibration must be policy-dependent.
Only the model's own rollouts reveal what it actually knows.

### The tail-knowledge falloff, worked

The mechanism, quantified. Sort facts by how often they appear in
pre-training. Facts seen 100,000+ times: the model answers
correctly. Facts seen 1,000 times: accuracy starts falling. Facts
seen once: the model answers with the popular fact instead of the
true one. The curve is a falloff, not a cliff: accuracy decays as
frequency drops.

Why the popular fact wins: the gradient from 100,000 repetitions
of "X is the capital" overwhelms the gradient from 1 mention of
"Y is the capital." The model is a frequency-weighted memory.
Below ~1,000 mentions the memory is too weak to beat the prior.
SFT on the tail fact adds one more mention: it does not move the
curve. The only fixes are external (retrieval: look the fact up
at inference) or behavioral (train the model to say "I do not
know": calibration, which needs the model's own rollouts, which
is why Schulman points to RL).

![Tail knowledge falloff](assets/media-generation-cs336-l15-tail-knowledge-fallo-0-031a20d1-3708-45b8-bdf0-28bb210c319e.webp "Accuracy decays with frequency. Below 1,000 mentions the popular fact wins. Source: original. Project: Stanford Frontier AI.")

> [!QA]
> Q: Walk me through the Schulman calibration argument. Why must calibration be policy-dependent?
> A: Calibration means the model says "I do not know" exactly when it does not know. The problem: "what it knows" depends on the policy. SFT teaches the model to answer everything confidently, because the demonstrations never say "I do not know." To teach abstention you need examples of the model itself being uncertain: its own rollouts, scored for correctness. Only the policy's own outputs reveal where its knowledge ends. A human-written "I do not know" dataset does not work: it teaches abstention on the demonstrator's uncertainty, not the model's. RL fixes this because RL trains on the policy's own rollouts: the model generates, gets scored, and learns which of its own outputs were wrong. Calibration is learned from your own mistakes, not from someone else's.
> Follow-up: Why does SFT on tail facts teach hallucination instead of the fact?
> A: Because one SFT example cannot beat the frequency prior. The fact appears once in SFT and once in pre-training: 2 mentions total, against 100,000 for the popular alternative. The gradient says "produce text shaped like this answer," and the model fills the content from its strongest memory: the popular fact. What gets learned is the format (citations, confident tone), not the fact. The model now emits confident wrong answers in exactly the shape you taught. Teaching format without knowledge is the hallucination factory.

## The key question

SFT fits a distribution: it learns to imitate demonstrations. But the
demonstrators are not the raters. Freelance writers preferred
Instruct Davinci's summaries over their own. What people rate well is
not what they write [44:09](ts:44:09). What if we stop fitting a
distribution and start maximizing a reward? That is RLHF: find a
policy that scores well, even if it collapses to one answer per
prompt [42:29](ts:42:29).

![RLHF concept](assets/l15-rlhf-concept.svg "Fitting a distribution vs maximizing a reward.")

## RLHF: maximize, not imitate

The pipeline: sample outputs at temperature 1, get pairwise rankings,
train a reward model, hill-climb with PPO plus a KL penalty to stay
near the reference [46:23](ts:46:23).

Two reasons to bother with RL at all. First, raters and
demonstrators differ (above). Second, verification beats generation:
checking a math proof is easier than writing one. A reward model that
judges is cheaper to train well than a demonstrator that produces.

## Annotators decide the values

![Annotation](assets/l15-annotation.svg "Who annotates decides the model's values, biases, and blind spots.")

InstructGPT's rubric: helpful, truthful, harmless. The Bard leak
showed a similar Likert-scale setup. The workforce shifted up:
bachelor and master holders, median age 35, $50+/hr, experts over
$100/hr for doctors and lawyers [49:15](ts:49:15).

Who rates rules. InstructGPT's annotators (Southeast Asian, US West
Coast) moved model opinions toward Buddhist, Hindu, and atheist
demographics [53:27](ts:53:27). Subliminal transfer is real: train on
"I like owls" data and the model inherits owl preference
[54:46](ts:54:46). Experts check factuality. Crowdworkers over-index
formatting (Hosking et al.) [55:37](ts:55:37). And verification is in
crisis: annotators use ChatGPT, and Bard raters had under a minute
per long response [51:31](ts:51:31).

> [!QA]
> Q: Why do annotators shift the model's politics? Walk me through the mechanism.
> A: The reward model is trained on pairwise rankings: given two responses, which is better? The annotators' preferences are the training signal. If the annotator pool skews toward certain demographics, their judgments on contested topics skew the same way: the reward model learns "this political framing is better." PPO then optimizes the policy to score well on that reward model. The policy drifts toward the annotators' worldview. InstructGPT's annotators (Southeast Asia, US West Coast) moved model opinions toward Buddhist, Hindu, and atheist demographics: measurable in survey-style probes. Subliminal transfer shows the same mechanism at the extreme: train on "I like owls" data and the model inherits the owl preference, because the preference is in the training signal and the model generalizes it. The rule: the annotators are the model's values. Choose them like you choose the product's values.
> Follow-up: Why do crowdworkers over-index formatting?
> A: Because formatting is checkable in under a minute and factuality is not. A rater with 60 seconds per long response cannot verify the claims, so they rank by what they can see: length, bullet points, confident tone. The reward model learns "longer is better." The policy learns to write long. RLHF on length alone does well on benchmarks: the benchmarks are also judged by surface features. The whole stack rewards the wrapper. Experts fix this only where they are deployed (factuality checks), which is why the workforce shifted to $100+/hr doctors and lawyers: you pay for the verification you need.

## Model-based annotation: AI feedback won

For catching up to the frontier, there is no room left for human
data. GPT-4's annotations matched careful human rankings at a tenth
the cost. Zephyr tried human-only collection and ended up with AI
feedback anyway. UltraChat and UltraFeedback are now standard. Tulu3
uses model-based annotation for its whole pipeline
[60:08](ts:60:08).

![Model annotation](assets/l15-model-annotation.svg "AI feedback won for catch-up. Watch the length bias.")

The catch: models share human biases and amplify them. Length
hacking: longer answers keep winning model judges, and RLHF on length
alone does well on benchmarks [64:42](ts:64:42).

> [!QA]
> Q: When does AI feedback beat human feedback, and when does it fail?
> A: AI feedback wins for catching up to the frontier: GPT-4's annotations matched careful human rankings at a tenth the cost. The mechanism: a frontier model already contains the judgment patterns of its training (human preferences, expert reasoning), and applying them is cheaper than re-collecting them. It fails in three ways. One: bias amplification. The judge shares human biases (length, sycophancy) and RLHF on its judgments amplifies them: longer answers win, so the policy writes longer. Two: capability ceiling. The judge cannot reliably rank outputs better than itself: you cannot bootstrap past the judge's own ability. Three: correlated errors. The judge and the policy share blind spots: both miss the same subtle bugs. Human experts are still needed at the frontier of factuality (doctors, lawyers at $100+/hr). The rule: AI feedback for scale, human experts for the decisions that matter.
> Follow-up: What is the verification crisis?
> A: Annotators use ChatGPT to do the annotating. The human preference data that trains the reward model is partly AI-generated: the "human" signal is a laundered model signal. Bard raters had under a minute per long response: they cannot verify, so they pattern-match on surface features. The result: the preference data teaches formatting, not judgment. The crisis is that nobody knows how much of the preference pipeline is humans judging versus models judging models. The honest fix: pay for time (under-a-minute ratings are worthless for factuality) and audit for AI-generated annotations.

## PPO and DPO: two ways to use preferences

**PPO**, briefly: policy gradients need fresh samples every step, and
sampling is expensive. Off-policy reuse with a trust region (TRPO),
then a clipping heuristic to stay near the old policy. Details next
lecture [67:14](ts:67:14).

**DPO** asks: can we skip PPO? Failed attempts: good/bad tokens, SFT
on good only, reward-model rejection sampling. DPO works, and the
derivation needs one assumption [69:09](ts:69:09).

![DPO](assets/l15-dpo.svg "Up on good, down on bad, scaled by surprise.")

Assume the policy can be anything (nonparametric). Then the
KL-constrained reward objective has a closed form: exponentially tilt
the reference policy by the reward. Solve for the implied reward,
plug it into the Bradley-Terry preference model, and you get the DPO
loss. The gradient intuition: increase the winner's likelihood,
decrease the loser's, with step sizes scaled by how wrong the implied
reward was [73:22](ts:73:22). No reward model, no sampling. Llama used
DPO in an outer loop with rejection sampling. DPO versus PPO results
are fragile and setup-dependent. Variants (SimPO, length-normalized)
barely matter [75:42](ts:75:42).

Work the gradient on a toy. Preferred response gets implied reward
0.2, dispreferred gets 0.8: the model is badly wrong, so the step is
large. Later, preferred gets 0.7, dispreferred 0.3: nearly right, so
the step is small. Surprise scales the update.

> [!QA]
> Q: Walk me through the DPO loss. Where does the reward model go?
> A: Nowhere: that is the point. Start from the KL-constrained RL objective: maximize expected reward minus beta times KL from the reference policy. Under the nonparametric assumption, this has a closed-form optimum: the policy is the reference policy exponentially tilted by reward over beta. Invert it: the implied reward is beta times the log-ratio of policy to reference. Plug this implied reward into the Bradley-Terry preference model (probability the winner is preferred = sigmoid of reward difference). The result is the DPO loss: it depends only on the policy, the reference, and the preference pairs. No reward model is ever trained, no sampling happens during training. The gradient pushes the winner's likelihood up and the loser's down, scaled by how wrong the current implied reward is. The assumption doing all the work: the policy can be anything, so the closed form holds. In practice the policy is a transformer, not anything, and the derivation is an approximation: one reason DPO-vs-PPO results are fragile.
> Follow-up: Why did Llama use DPO in an outer loop with rejection sampling?
> A: Because pure offline DPO trains on fixed pairs: the policy never sees its own outputs. Rejection sampling closes the loop: sample from the current policy, keep the best samples by reward, DPO on those. It is a cheap way to get on-policy data without PPO's battle station. The lesson: DPO is an update rule, not a training regime. How you generate the pairs matters as much as the loss.

![PPO battle station](assets/media-generation-cs336-l15-ppo-battle-0-559d6766-8ce6-4222-95a2-e148f502e284.webp "Four networks, four forward passes: more memory than pre-training. Source: original. Project: Stanford Frontier AI.")

> [!QA]
> Q: Walk me through the PPO battle station. Why four networks?
> A: PPO for RLHF runs four models at once. The policy: the model being trained, generating rollouts. The reference policy: the frozen SFT model, computing the KL penalty that keeps the policy from drifting. The reward model: scores the rollouts. The critic (value model): estimates expected return, reducing the variance of the policy gradient. Each training step: generate with the policy (forward pass 1), score with the reward model (pass 2), compute KL against the reference (pass 3), compute advantages with the critic (pass 4). Four forward passes per step, four models in memory. That is why RLHF uses more memory than pre-training: you are training one model with three auxiliaries. DPO deletes the reward model and the critic. GRPO (next lecture) deletes the critic. Each deletion is a simplification that kept most of the gains.
> Follow-up: What breaks if you drop the KL penalty?
> A: The policy over-optimizes the reward model. The reward model is a proxy for human taste with blind spots: the policy finds outputs that score high and are garbage (length hacking, sycophancy, confident nonsense). The KL penalty anchors the policy near the SFT model: you can only move so far from sane behavior. It is the single most important regularizer in RLHF. Without it, over-optimization is not a risk, it is a guarantee.

## Where RLHF breaks: failure modes

**Over-optimization**: push hard and you overfit the learned reward
model. The KL regularizer is critical [76:39](ts:76:39). The reward
model is a proxy: maximize it blindly and the policy finds its blind
spots. **Mode collapse**: RL policies concentrate on a few outputs.
Diversity dies [77:27](ts:77:27). **Miscalibration**: GPT-4-era plots
showed RLHF models are uncalibrated, and nobody has fully solved it
[78:10](ts:78:10). Next lecture: RLVR, where rewards do not
over-optimize because they are verifiable.

## Safety: the last line of defense

![Safety](assets/l15-safety.svg "Violation rate vs false refusals. 500 examples is surgical.")

Post-training is the last line of defense against misuse
[28:16](ts:28:16). The tradeoff: violation rate (bad queries getting
through) versus false refusal ("how do I kill a Python process").
Llama 2 used a few thousand safety examples. OLMo mined WildChat
interactions for 50k. Surprise: 500 well-chosen examples cut
malicious-instruction rates dramatically. The model already has a
safe/unsafe axis from pre-training. SFT just pulls it out
[33:43](ts:33:43). Extraction, not installation: this only works for
behaviors pre-training already contains.

> [!QA]
> Q: How can 500 examples change a model's safety behavior?
> A: Because SFT is extraction, not installation. Pre-training already built the concepts of safe and unsafe behavior. The 500 examples just steer the model onto the safe axis. This is why quality beats quantity in SFT: a few right examples pull out the right mode. The catch: this only works for behaviors pre-training already contains. Truly new capabilities (rare programming languages, frontier knowledge) need real training, not steering. And fine-grained safety distinctions still need large-scale collection.
> Follow-up: Why is safety data even scarcer than capability data?
> A: Safety is adversarial and product-specific. Capabilities data can be synthetic or distilled. Safety data must anticipate real attacks (jailbreaks from WildChat), balance refusals precisely, and match the company's policy. Companies treat it as a trade secret: it reveals what they fear and how they defend. The public references (Llama 2's brief description, OLMo's pipeline) are the exceptions.

## Mid-training: the boundary dissolves

The pre/post boundary is dissolving. High-quality and instruction data
now mix into the decay phase of pre-training: lowest learning rate,
closest to deployment, highest quality data [36:11](ts:36:11). MiniCPM
shows the mix shift: internet data out, UltraChat and StackExchange QA
in. Pet peeve, stated plainly: "base model" is now a lie, since base
models trained on chat data [37:13](ts:37:13). Data mixtures remain
trial and error. Decay-phase ablations are cheap enough to guide the
pre-training mix [40:05](ts:40:05).

![Mid-training](assets/l15-midtraining.svg "Instruction data in the decay phase. The base-model label no longer means what it says.")

![Alignment map](assets/media-generation-cs336-l15-alignment-landscape-0-dea20d4b-5706-4e17-8a76-e78ae065315d.webp "SFT, reward model, PPO, DPO, GRPO: five stages, one goal. Source: original. Project: Stanford Frontier AI.")

> [!QA]
> Q: You ship two products: a coding assistant and a medical triage chatbot. Design the alignment stack for each.
> A: Coding assistant: SFT on verified code (execution-filtered: the tests decide correctness), then RLVR with execution as the reward (next lecture's method: verifiable, no over-optimization). Skip human preference RLHF: taste does not matter, tests do. DPO on human pairs is optional polish for style. The stack: SFT for format, RLVR for capability. Medical triage: the opposite. No verifiable reward exists for "good triage advice": correctness is expert judgment, and wrong answers kill. SFT on expert-written responses (doctors, not crowdworkers: factuality needs experts), then RLHF with expert annotators on safety-critical pairs, heavy KL penalty, plus a 500-example safety SFT (the lecture's surgical number). Refusal behavior is a first-class requirement: the model must abstain outside its competence. The decision rule: verifiable domain, use verifiable rewards. Unverifiable domain, pay for experts and penalize drift hard.
> Follow-up: Where does DPO fit in each stack?
> A: Coding: as cheap style polish after RLVR. The pairs are (good solution, bad solution): DPO teaches the policy to prefer the good one's shape. It cannot teach correctness the tests did not. Medical: DPO is risky as the main method. Offline pairs cannot teach calibration (Schulman's argument: only the policy's own rollouts reveal its uncertainty). PPO with expert rewards at least trains on the policy's outputs. DPO trains on fixed pairs from other models. For safety-critical abstention, you need the policy to learn its own limits: that needs on-policy data. DPO is the budget option. The medical stack is not where you economize.

## Mapping back: what each step fixes

| Pain | Step | How |
|---|---|---|
| Base model cannot follow instructions | SFT | Demonstrations teach the shape of helpful answers. |
| Style games the evals | Style control | Control style separately. Capability evals, not vibe evals. |
| Tail knowledge hallucinates | RL calibration | Calibration must be policy-dependent. Schulman's argument. |
| Raters differ from writers | RLHF | Maximize the reward raters give, do not imitate writers. |
| No human data at the frontier | Model annotation | GPT-4 annotates at 1/10th cost. Watch length hacking. |
| PPO needs fresh samples | DPO | Tilt the reference by implied reward. No reward model, no sampling. |
| Over-optimization | KL regularizer | Stay near the reference. RLVR next lecture for verifiable rewards. |
| Misuse | Safety SFT | 500 examples surgical. Extraction, not installation. |

## The honest price

Frontier post-training is a trade secret. The public record is old
(2022-2023) plus open-source recipes. Annotator demographics come from
ScaleAI subsets, not the whole industry. DPO-vs-PPO claims are
setup-fragile. RLHF's failure modes (over-optimization, mode
collapse, miscalibration) are unsolved, not managed. And the deepest
price: post-training inherits pre-training's sins (Lecture 9). No
amount of preference tuning fixes a base model that never saw the
knowledge.

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **GPT-3 could not follow instructions.** Pre-training builds the
   soup. Post-training extracts behaviors: SFT plus RL on
   preferences.
2. **SFT shows good examples.** FLAN to self-instruct to Alpaca to
   synthetic to agentic. Chattier, more expert, more tool use.
3. **Style is not capability.** Bullet lists win AlpacaEval, move no
   benchmark. Control style separately.
4. **Tail knowledge hallucinates.** The model generalizes the format
   and invents the fact. RL recalibrates what it knows.
5. **Maximize, not imitate.** Raters differ from writers. RLHF:
   sample, rank, reward model, PPO plus KL. Who rates rules:
   demographics transfer, owls transfer.
6. **DPO skips PPO.** Tilt the reference by implied reward. Up on
   good, down on bad, scaled by surprise. No reward model, no
   sampling.
7. **RLHF breaks three ways.** Over-optimization (KL is critical),
   mode collapse (diversity dies), miscalibration (unsolved).
8. **Safety is the last line.** 500 examples surgical: extraction,
   not installation. The pre/post boundary dissolves into
   mid-training.

## Go deeper

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:16px 0;">
<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" src="https://www.youtube-nocookie.com/embed/VIARnQFSeHk" title="Yannic Kilcher: InstructGPT paper explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
- Yannic Kilcher, InstructGPT explained (the embed above): https://www.youtube.com/watch?v=VIARnQFSeHk
- Ouyang et al., InstructGPT: https://arxiv.org/abs/2203.02155
- Rafailov et al., DPO: https://arxiv.org/abs/2305.18290
- Tunstall et al., Zephyr: https://arxiv.org/abs/2310.16944

## Official sources and further reading

**Official:**
- Lecture 15 video.
- InstructGPT paper and appendix (the last public data-collection
  glimpse).
- Anthropic HH paper (2022). Constitutional AI.

**Further reading:**
- FLAN, Alpaca, Open Assistant, Tulu3 papers.
- DPO paper and variants (SimPO). PPO paper.
- Hosking et al. on annotator expertise. Emergent misalignment work.

**Caveats from these sources.** Frontier post-training is a trade
secret. The public record is old (2022-2023) plus open-source
recipes. Annotator demographics come from ScaleAI subsets, not the
whole industry. DPO-vs-PPO claims are setup-fragile.

## Connections to the other courses

- **CS336 L14:** synthetic data pipelines continue here.
- **CS336 L16:** RLVR, the next step beyond RLHF.
- **CS329H:** preference learning and choice theory behind RLHF.
