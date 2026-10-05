---
page_id: cs336-l15
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 15
nav: "L15 · SFT, RLHF"
title: "Lecture 15: Post-Training: SFT and RLHF"
summary: "From GPT-3 to ChatGPT. SFT data: how instruction datasets evolved from FLAN to agentic data, and the pitfalls of style, knowledge, and safety. Then RLHF: why optimize instead of imitate, reward models, PPO, and the DPO derivation."
date: "2026-05-18"
instructor: "Tatsunori Hashimoto"
offering: "Spring 2026"
duration: "1:19:55"
video_id: 2oH6PWPrYFo
video_title: "Stanford CS336 Spring 2026 Lecture 15: Post-Training (SFT, RLHF)"
video_caption: "Original lecture. Timestamps link to exact moments."
concepts: [post-training, sft, rlhf, ppo, dpo, reward models]
papers: []
sources:
  - tag: video
    label: "Lecture 15 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=2oH6PWPrYFo
  - tag: slides
    label: "Lecture 15 slides (local copy: sources/cs336/pdfs/lecture_15.pdf)"
  - tag: notes
    label: "Official subtitle transcript (en-orig)"
---

## From GPT-3 to ChatGPT

Everything so far gets you to a souped-up GPT-3: more flops, more data, a stronger base model. But a base model's utility is limited. If your first exposure was ChatGPT, going back to GPT-3 feels broken: the only reliable uses were copywriting and toys that needed no instruction following. This lecture takes you from GPT-3 to something close to ChatGPT. The next lecture goes from ChatGPT to the thinking models (o1-style) via RLVR [00:27:00](ts:00:27:00).

The process is **post-training**. Hashimoto calls it the messiest part of language modeling: explicit data collection, explicit steering, and artisanal engineering that the architecture and systems lectures never needed. Pretraining stays critical. It scales broadly, and post-training without it gets you nothing. But once you have the "primordial soup" of pretraining, you must extract the behaviors you want with much more deliberate data [02:30:00](ts:02:30:00).

One warning up front. Information about frontier post-training is sparse. The old papers are detailed: Stiennon et al. 2020 (the RLHF paper) publishes full annotation guidelines, and Bai et al. 2022 (Anthropic's HH paper) documents safety annotation. Since competition heated up, vendors treat post-training data as a trade secret. A leaked Scale AI document shows the dynamic: they were trying to make Google Bard better, could not figure out why GPT-4 was better, and told annotators to reverse-engineer GPT-4 by writing more detailed responses. Algorithms are public and well understood. Data is the secret sauce [03:27:00](ts:03:27:00).

The standard recipe has two parts [05:54:00](ts:05:54:00):

1. **SFT.** Collect demonstration data: for each prompt, an annotator writes a reference response. Fine-tune on it.
2. **RLHF.** Use reinforcement learning to shape behavior toward what humans rate as good.

## SFT data: the progression

SFT is almost entirely about data. The method is gradient descent, exactly like pretraining, with minor variations. So the lecture walks through how open-world instruction datasets evolved [06:52:00](ts:06:52:00):

**FLAN** → Self-Instruct → **Alpaca** → ShareGPT / Vicuna → **Open Assistant** → WizardLM → **Tulu 3** → Nemotron → tool use and agentic data.

### FLAN

FLAN trained Google's T5 on a multitask mixture: NLP researchers had already collected supervised datasets of inputs and outputs, so why not train on all of them? Reasonable, and visionary for its time. But look at what is actually inside [11:25:00](ts:11:25:00):

- An email body with "Write a subject line for this email" appended at the end, taken from the Enron emails dataset. Nobody prompts a model this way.
- "Write highlights for this article" followed by a travel article, with a short summary as the target, taken from a summarization dataset. The summaries are short and often hallucinated: details appear that are not in the input.

FLAN inherits the deficiencies of the benchmarks it was built from, and the unnatural structure carries all the way down. There was also a wrong theory behind it: pretraining needs scale, so post-training must need scale too, hence the gigantic dataset. Later work showed the opposite. With a sufficiently strong pretrained model, a few high quality examples enable instruction following, because pretraining generalization does most of the work. FLAN was on the wrong end of the quality-quantity tradeoff [13:33:00](ts:13:33:00).

### Alpaca, Open Assistant, Nemotron

After ChatGPT, Hashimoto's students built **Alpaca**: distill ChatGPT traces into input-output pairs. The examples look natural: chatty, detailed responses. Alpaca reliably induced ChatGPT-like behavior, but only on the original Llama models. Both pretraining and post-training have to cooperate [14:38:00](ts:14:38:00).

That success created enormous optimism: collect a high quality instruction dataset at scale and catch up to the closed labs. **Open Assistant** was the crowdsourced attempt, Wikipedia-style: volunteers writing hard prompts and expert responses. It produced a decent amount of data (on the order of 10K examples) before stalling as a project [15:36:00](ts:15:36:00).

The newest generation moved past chat. **Nemotron** (NVIDIA's open SFT data) is full of agentic examples: the assistant field contains a response, and tool calls happen in parallel alongside the text. SFT data is now structured supervision for agents, not just chat [16:50:00](ts:16:50:00).

Three high-level shifts summarize the evolution [18:05:00](ts:18:05:00):

1. **Chattiness.** Classic NLP datasets were input-to-programmatic-output. People want to talk to people, not to a benchmark. Responses got longer and more human-like.
2. **Detail and experts.** Open Assistant exemplifies expert-written, detailed responses. This is a pro and a con, as the pitfalls section shows.
3. **Tool use.** The interface question: what API do agents need? SFT now teaches that.

A student asked how much the correctness of input-output pairs matters. Hashimoto's answer is honest: it is a genuinely hard question. You should collect the highest quality responses you can, because bad data teaches bad behavior. But models learn instruction following from strange data too. One of Percy Liang's former students trained models to follow instructions with data that had no responses at all. Pretraining generalization covers a lot [10:27:00](ts:10:27:00).

## Pitfalls of SFT data

### Style: length and bullet points

Datasets vary wildly in response length and style, and style matters enormously in preference evaluation. Wang et al. 2023 and Dubois et al. 2023 show strong length effects in both human and GPT-based evaluations: raters pick longer, bullet-pointed, detailed responses. But those same datasets barely move standard benchmarks. AlpacaEval scores swing while capabilities stay flat. The takeaway: control style separately from capabilities. Engagement signals can fool you into thinking the model got smarter when only its tone changed [19:42:00](ts:19:42:00).

### Knowledge: SFT teaches two things at once

Consider an Open Assistant example: a prompt asks for an introduction to "monopsony" with citations, and the response cites Bivens and Mishel (2013). Training on this teaches two entangled things [22:18:00](ts:22:18:00):

1. The knowledge: the Bivens and Mishel citation exists.
2. The behavior: good responses include references.

The model cannot tell which it is supposed to generalize. If it generalizes the template without the knowledge, it hallucinates fake references. The folklore, with empirical support from Schulman 2023 and Gekhman 2023: **fine-tuning a model on facts it does not know makes it hallucinate**. You may not want to train on tail knowledge, even when the use case demands it, because the format markers (like "References:") force the model to emit knowledge-shaped text it cannot verify. There is no formal definition of tail knowledge. Wikipedia article length is a usable proxy for how well-known something is [24:48:00](ts:24:48:00).

This is one reason RL matters, per John Schulman's argument. Teaching the model what it knows is policy-dependent: you cannot shove knowledge down its throat from outside and expect calibration. RL can extract an internal "know" versus "do-not-know" direction into the output policy, rewarding references when the model is in the know-something direction and penalizing them otherwise. But RL cannot help if the model has no such internal signal at all [26:27:00](ts:26:27:00).

### Safety

Post-training teams are the last line of defense against misuse: misinformation, scams, spear phishing. The mechanism is safety SFT: data that teaches refusal of malicious inputs. Details are even sparser than for capabilities. The Llama 2 report does not even say how many safety examples it used, though the count is on the order of a few thousand [28:18:00](ts:28:18:00).

Every safety approach balances two error rates: the **violation rate** (bad queries getting through) and the **false refusal rate** (refusing benign queries, like "how do I kill a Python process?"). You navigate the Pareto tradeoff with tailored data [29:34:00](ts:29:34:00).

The best public reference is **Tulu 3** (Allen AI, for the OLMo models), with roughly 50K safety examples. The strategy: mine real usage. An earlier project, WildChat, gave people free chat access and logged interactions. The team extracted attempted jailbreaks and unsafe behaviors at scale, then wrote the preferred response (resist the jailbreak, refuse the request). Closed labs appear to do the same: watch usage, find bad behaviors, have annotators play whack-a-mole [30:33:00](ts:30:33:00).

A striking fact: about 500 well-chosen safety examples dramatically reduce malicious-instruction following (shown on Anthropic's HH data for hate speech). The model already contains a "safe mode" direction from pretraining. A small push extracts it. That does not eliminate the need for scale: enforcing fine-grained distinctions about what is safe requires very large data collection [32:15:00](ts:32:15:00).

> [!KEY] SFT works best when it extracts behaviors already in pretraining. Adding factually correct data can hurt (hallucination from tail knowledge). Small amounts of the right data steer style, safety, and instruction following. The long tail still benefits from more.

## SFT method: just gradient descent, then mid-training

The algorithm section is deliberately short. You take the loss and call backward. That is basically it. The one nuance that matters is the industry-wide trend of **folding instruction tuning into pretraining** [35:56:00](ts:35:56:00).

It used to be clean: pretrain on web data, then post-train. Then people asked why separate the two. Now high quality data and instruction-tuning data get mixed into the tail end of training, during the decay phase. This scales up instruction tuning without catastrophic forgetting and emphasizes high quality data. The impact has been dramatic, and as far as Hashimoto knows, everyone does it. You can see it in model reports as "mid-training" or "second-phase pretraining with a different data mix". MiniCPM publicized the mix shift nicely: standard internet data first, then a switch to StackExchange QA, UltraChat, and other chatty high quality data with less general web data [37:41:00](ts:37:41:00).

A student asked whether prompts are masked in this phase. No: it is pure pretraining, so the model also predicts the prompt. That is not a huge difference, since some SFT recipes predict the prompt too. The intuition for putting the best data in the decay: decay is the most important phase (closest to deployment, lowest learning rate), so it should see the highest quality data [38:32:00](ts:38:32:00).

Data mixtures for mid-training are mostly trial and error, with unreliable algorithms on top. But mid-training is much shorter than full pretraining, so you can run many ablations: roughly ten mid-training runs per full run. The usual workflow: ablate data in the decay phase (cheap), estimate quality, and reflect the findings back into the pretraining mix. Pretraining itself cannot be all high quality because there are not enough tokens. A fun leaked example: in Meta's books lawsuit, court documents show researchers ablating book subsets to estimate their value before choosing mixes [39:50:00](ts:39:50:00).

> [!PROF] Hashimoto's pet peeve: "base model" is now a lie. A base model should mean next-word prediction on internet data. Today's base models are pretrained on UltraChat and other synthetic chat datasets designed to make them good at chat. The label no longer means what it says [37:14:00](ts:37:14:00).

## RLHF: from imitation to optimization

SFT is generative modeling: fit p(y|x) to a reference distribution. RLHF is a different game: find a policy that maximizes a reward. The model is no longer a distribution you fit. It is a policy you optimize. It can legally collapse to a single answer per prompt, with no diversity at all, as long as the reward is good. Keep this distinction in mind for everything that follows [42:30:00](ts:42:30:00).

Why optimize at all, instead of collecting SFT data forever? Two reasons [44:01:00](ts:44:01:00):

1. **The generation-verification gap.** What people write differs from what they prefer. In a study (Zhang et al. 2023, benchmarking LLMs for news summarization), freelance writers were shown summaries from a pre-ChatGPT model alongside their own writing, and several preferred the model's. They write well, but they judge differently. Rating outputs captures something demonstrations miss.
2. **Verification is easier than generation.** Checking a math proof is easier than writing one. DeepSeek's self-verification work follows this path. Wherever a verifier is cheap, judging beats demonstrating.

RLVR (RL with verifiable rewards, the thinking-model recipe) is its own universe and gets the next lecture. Today is RLHF with human preferences.

The RLHF loop [46:23:00](ts:46:23:00):

1. Start from the SFT model, which follows instructions and is fairly diverse.
2. Sample several outputs per prompt (temperature 1 is reasonable).
3. A rater ranks the outputs, sometimes just pairwise or binary.
4. Train a **reward model** on those rankings.
5. Use standard RL to maximize the reward model's score.

The reward model is an indirection: it is easier to train a verifier than to directly train a model that does well.

## RLHF data

Data collection is pairwise feedback through annotation interfaces: two AI responses side by side, pick the better one. The last public glimpse into industry data collection is the **InstructGPT appendix**: raters score outputs for helpfulness (clear writing, appropriate length), truthfulness (no hallucination), and harmlessness (refuse questionable prompts). A leaked **Google Bard** annotation guide has a similar structure, rated on a Likert scale rather than pairwise [47:40:00](ts:47:40:00).

The annotator workforce has shifted upward. On one Scale AI platform (Outlier), the large majority of annotators hold bachelor's or master's degrees (around 70%), the modal age is about 35, and tasks are creative and technical writing. Bespoke expert annotation is growing fast: labs hire doctors and lawyers to annotate domain responses, paying median wages above $50/hour and experts over $100/hour. The old mental model of cheap overseas pairwise feedback is incomplete. It is now a pyramid: expensive experts on top, scalable low-cost annotation still underneath [49:23:00](ts:49:23:00).

Collection is genuinely hard [52:57:00](ts:52:57:00):

- **Verifiable annotators.** Annotators use ChatGPT to do their annotation work, and preventing this is extremely difficult.
- **Correctness under time pressure.** Bard annotators reported they had under a minute to verify long chat responses. Impossible to do properly, and it became a labor dispute.
- **Demographics shape the model.** Post-training is the final shaping step before shipping, so annotator biases land directly in the product. Santurkar et al. 2023 asked models standard opinion-poll questions: base models aligned with Protestant and Catholic opinions, while post-trained models shifted toward Buddhist, Hindu, and atheist opinions. The InstructGPT annotator demographics (many Southeast Asian, many US West Coast) line up with the shift.
- **Subliminal transfer.** Emergent misalignment work shows that training on innocuous-looking data (for example, text from a model trained to say "I like owls") transfers the preference to the student. Subtle biases transmit through data in ways that are hard to catch.
- **Expertise changes what gets checked.** Hosking, Blunsom, and Bartolo 2024: non-expert crowdworkers overweight formatting effects. Expert annotators overweight factuality and consistency. Factuality is much harder to check, so without experts it goes unchecked.

A student asked how to measure annotator quality. There is no gold standard. Two partial answers: guideline adherence (a detailed rubric with objective criteria, like "nothing in the first three pages of Google contradicts it") and inter-annotator agreement. Agreement measures variance, not bias: it cannot tell you whether the question was right, and if every annotator uses ChatGPT, agreement is perfect and meaningless [56:38:00](ts:56:38:00).

### Model-based annotation

Models turned out to be surprisingly good annotators. Hashimoto's group compared GPT-4 annotations against carefully curated human ones: near-perfect rank correlation at the system level, agreement near human inter-annotator levels, at roughly an order of magnitude lower cost. The field has converged: if your goal is catching up to the frontier in capabilities, there is no room for human-collected data [60:09:00](ts:60:09:00).

The instructive case is HuggingFace's **Zephyr**. The team tried to avoid distillation entirely and collected human data from the same vendors OpenAI uses. It was extremely slow, costly, and no better than model-based feedback. They switched to AI feedback. UltraChat and UltraFeedback (both model-generated) are now standard, and Tulu 3 uses model-based annotations across its whole pipeline. Two earlier landmarks: **Constitutional AI** (Anthropic), which prompted a model to generate its own safety data, an early self-post-training loop. And **Self-Instruct**, the capability-centric version of bootstrapping your own data. The caveat: if you need world knowledge that models lack (lawyers, scientists), you are stuck with human annotators [63:43:00](ts:63:43:00).

Models also share human biases, sometimes worse. You can push response length out and keep gaining win rate on model-judged benchmarks. One paper showed RLHF on length alone does well on many benchmarks (Chen et al. 2024, Singhal et al. 2024). Length hacking is real [64:32:00](ts:64:32:00).

## PPO, briefly

The RLHF objective appears almost verbatim as equation 2 of the InstructGPT paper: maximize expected reward under the policy, minus a KL penalty keeping the policy close to the pretrained model so it does not go degenerate. Stiennon et al. use a pairwise-feedback classifier as the reward and hill-climb on it [65:50:00](ts:65:50:00).

The algorithm is PPO. The conceptual progression, at a baby level [67:00:00](ts:67:00:00):

1. **Policy gradients.** Start from the identity ∇E_p[R] = E_p[R ∇log p]. This looks like SFT with reward-weighted examples. Problem: you must sample fresh rollouts for every optimization step, and sampling (inference) is expensive while training is arithmetically efficient.
2. **TRPO.** Go off-policy: roll out once, reuse the rollout for multiple steps, but stay close to the current policy using a trust-region constraint with importance-weighting corrections.
3. **PPO.** TRPO's constraint is awkward, so replace it with a heuristic: clip the probability ratios to discourage moving too far from the original policy.

Full details come next lecture, and PPO is on the assignment. Many researchers tried to kill PPO with simpler ideas, and Hashimoto lists them so you do not repeat the attempts: prepend [GOOD]/[BAD] control tokens and SFT on pairs. Train only on preferred outputs. Train a reward model, filter LM outputs, SFT on the winners. Train a reward model, sample 1024 outputs, SFT on the best one. None worked well, though reward-filtered SFT works somewhat [69:14:00](ts:69:14:00).

## DPO

**Direct Preference Optimization** finally delivered the simpler algorithm. It removes the two complicated parts of PPO: the reward model and all on-policy machinery (rollouts, outer loops). The intuition: take gradient steps toward the log-loss of good responses and negative gradient steps away from bad ones, weighted appropriately [70:19:00](ts:70:19:00).

The derivation is short and worth following [71:17:00](ts:71:17:00):

1. Start from the RLHF objective: maximize expected reward minus β times KL to the reference policy.
2. Make one strong assumption: the policy π is not a neural network but the set of *all* policies (nonparametric). Then the maximizer has a closed form: **exponential tilting** of the reference policy.

\[ \pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right) \]

Good responses get exponentially upweighted, bad ones exponentially downweighted. This is the perfect solution if π can be anything.

3. Solve this for the **implied reward**: the reward function that would induce the observed policy.
4. Plug that implied reward into the Bradley-Terry preference objective (the Stiennon pairwise objective). The result is the DPO objective:

\[ \mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \]

where y_w is the winning response and y_l the losing one.

The gradient is the intuitive form Hashimoto emphasizes [73:22:00](ts:73:22:00): increase the likelihood of the winner, decrease the likelihood of the loser, and scale the step by how wrong the implied reward model is. If the model already strongly prefers the winner, take a small step. If it rated them nearly equal, take a big step.

LLaMA used DPO as its core RLHF primitive inside an outer loop: SFT, then DPO, then generate candidates from the DPO model, rejection-sample, and repeat. Variants followed (SimPO drops the reference model. Length-normalized DPO counters length hacking), but none clearly matter. Results are fragile and setup-contingent: AI2 found PPO beating DPO in one study, while Tulu 2 showed well-executed DPO beating PPO. The practical takeaway: the core idea (positive gradient on good, negative gradient on bad, scaled correctly) works well, and it is good enough unless you are training the very best model at the frontier [74:42:00](ts:74:42:00).

## Things to watch out for

Two failure modes close the lecture [76:35:00](ts:76:35:00):

- **Over-optimization.** Push RLHF hard and you overfit the learned reward model. The KL regularizer is what prevents this, and it is critical when the optimizer is strong. The pattern holds for human preferences and noisy LM preferences, but not for noiseless LM preferences.
- **Mode collapse and entropy.** RLHF models stop being probabilistic models. They concentrate on a few outputs and lose calibration by default. The GPT-4 report lists uncalibrated models after RLHF as an open problem, and Anthropic argues it is natural and only sometimes fixable. This matters enormously for next lecture's RLVR, where entropy and exploration decide whether the model can find hard solutions.

## Summary

- RLHF data collection is hard: annotator quality, demographics, AI-assisted annotation, and subliminal biases all shape the final model.
- RLHF algorithms are more complex than SFT, especially PPO. DPO is the accessible alternative: no reward model, no rollouts, just preference-weighted gradients.
- Over-optimizing the reward is the central danger. The bridge to next lecture: find rewards you cannot over-optimize, and performance keeps climbing with compute. That is the promise of RLVR.

## Assignment connection

Assignment 5 is the alignment assignment: SFT and RLHF-style post-training. The lecture's PPO treatment is brief on purpose. The full version comes next lecture. GRPO, the simpler recent variant Hashimoto mentions, is what the assignment uses, so learn the PPO-to-DPO conceptual progression here and expect the mechanics there.
