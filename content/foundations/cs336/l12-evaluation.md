---
page_id: cs336-l12
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 12
nav: "L12 · Evaluation"
title: "Lecture 12: Evaluation"
summary: "What 'good' means for a language model. Perplexity, exam benchmarks, chat benchmarks, agentic benchmarks, reasoning benchmarks, safety benchmarks, and the validity problems that threaten all of them."
date: "2026-05-06"
instructor: "Percy Liang"
offering: "Spring 2026"
duration: "1:18:34"
video_id: JpAxdTWQJxM
video_title: "Stanford CS336 Spring 2026 Lecture 12: Evaluation"
video_caption: "Original lecture. Timestamps link to exact moments."
concepts: [evaluation, perplexity, benchmarks, eval methodology]
papers: []
sources:
  - tag: video
    label: "Lecture 12 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=JpAxdTWQJxM
  - tag: code
    label: "lecture_12.py — executable lecture code"
    url: https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py
  - tag: notes
    label: "Official subtitle transcript (en-orig)"
---

## Evaluation sets north stars

Everything needed to train a language model is now covered: architecture, optimizer, training loop, kernels, parallelism, scaling laws, inference. The missing piece is data, which comes next week. But before data, one more topic is needed: what behavior do we even want from the model [00:01:00](ts:00:01:00).

Evaluation asks a simple question. Given a trained model, how good is it?

It sounds mechanical: define prompts, get responses, compute accuracy. It is not. Evaluation is deep because it shapes the development of AI. Evaluation sets north stars. Every model developer, open and closed, tracks evaluation as the measure of progress [00:01:37](ts:00:01:37).

> [!KEY] By choosing your evaluation, you implicitly choose what your model becomes. Evaluation is not just measurement. It is steering.

## What is good? The abstract-to-concrete problem

Evaluation starts with an abstract construct: "good at conversation," "good at reasoning." The core challenge is turning that construct into a concrete metric built on concrete prompts or environments [00:01:59](ts:00:01:59).

```mermaid
flowchart LR
    A[Abstract construct<br>"good at reasoning"] --> B[Concrete metric<br>prompts + environments + scoring]
    B --> C[Model development<br>optimizes for the metric]
    C -.-> A
```

There is no single correct answer to "what is good." The lecture walks through several candidate definitions [00:02:39](ts:00:02:39):

| Definition of "good" | How it is measured | Lens |
|---|---|---|
| Does well on benchmarks | Intelligence index (e.g. Artificial Analysis) | Capability |
| Does well and is cheap | Intelligence index plotted against inference cost | Efficiency |
| People prefer its responses | Human pairwise preferences, Elo (Arena AI) | Preference |
| People choose to use it and pay for it | Usage statistics (e.g. OpenRouter rankings) | Economics |

Cost correlates with intelligence, but not perfectly [00:03:14](ts:00:03:14). Preference conflates style with correctness. Usage is not representative of all users. None of these is the correct answer. They are different ways to make "good" concrete.

## Perplexity: the native evaluation of a distribution

A language model is a distribution \(p(x)\) over sequences of tokens. The natural way to evaluate a distribution is perplexity:

\[
\text{perplexity}(D) = \left(\frac{1}{p(D)}\right)^{1/|D|}
\]

It measures how much probability mass the model assigns to a test set \(D\). During training you minimize perplexity on the training set. The obvious move is to measure it on the test set [00:05:21](ts:00:05:21).

### The classical paradigm: in-distribution

Through the 2010s, language modeling papers measured perplexity on held-out splits of the same dataset: Penn Treebank, WikiText-103, the One Billion Word Benchmark. Train on the train split, test on the test split [00:06:38](ts:00:06:38).

This paradigm produced real progress. A 2016 paper applied pure CNNs and LSTMs to the One Billion Word Benchmark and cut perplexity from 51.3 to 30.0. That was the definitive result that showed neural models were the way to go ([Jozefowicz et al., 2016](https://arxiv.org/abs/1602.02410)).

### GPT-2 and out-of-distribution evaluation

GPT-2 (2019) broke the paradigm. It trained on WebText (40GB of web pages linked from Reddit) and evaluated zero-shot on the standard datasets it had never trained on [00:07:55](ts:00:07:55). On small datasets like PTB it beat the state of the art (35 vs. 46 perplexity). On large datasets like the One Billion Word Benchmark, in-distribution training still won. This moved the field to a new standard: train on one large dataset, evaluate zero-shot on standard benchmarks.

### "Perplexity is all you need"

Liang presents the mindset that drove years of scaling research, with the caveat not to take it too seriously [00:09:33](ts:00:09:33):

1. There is a true distribution \(t\). Your model is \(p\).
2. The best possible perplexity is the entropy \(H(t)\), achieved exactly when \(p = t\).
3. If \(p = t\), you can solve any task by conditioning: \(p(\text{solution} \mid \text{problem})\).
4. So driving perplexity down eventually reaches AGI.

Before GPT-3, when the gains were not obvious, this belief kept scaling alive.

### Perplexity is also more than you need

Perplexity charges you for every token. Take "Stanford was founded in 1885." Predicting "1885" tests real world knowledge. Predicting "founded" is mostly incidental. Perplexity penalizes both equally [00:11:38](ts:00:11:38).

The fix is **conditional perplexity**: condition on a prompt and measure perplexity only on the response. You focus on the tokens you care about.

Some benchmarks are perplexity in disguise. LAMBADA ([Paperno et al., 2016](https://arxiv.org/abs/1606.06031)) is fill-in-the-blank: it looks like accuracy, but it is next-token prediction on carefully chosen positions that require long-range dependencies [00:13:01](ts:00:13:01). HellaSwag ([Zellers et al., 2019](https://arxiv.org/pdf/1905.07830)) is multiple-choice sentence completion, which is perplexity in another form [00:14:44](ts:00:14:44).

> [!CAVEAT] A perplexity leaderboard has a trust problem. Participants submit their own log probabilities. Nothing stops a cheater from returning log prob zero (probability one) for everything, since the organizer cannot verify the probabilities sum to one. Downstream accuracy evals avoid this: the model returns a response, and you score it [00:15:18](ts:00:15:18).

Perplexity is still used heavily in model development because it varies smoothly with scale, which makes it ideal for scaling laws. But it does not convince skeptics. For that you need benchmarks that capture real situations.

## Exam benchmarks: harder and harder questions

Exams are a natural testing format for humans and models alike: control over subject and difficulty, unambiguous answers, easy grading [00:18:26](ts:00:18:26).

### MMLU and the saturation treadmill

**MMLU** (Hendrycks et al., 2020): 57 subjects (math, US history, law, morality), multiple-choice, crowdsourced by students. Despite the name, it tests knowledge and reasoning, not language understanding. GPT-3 was evaluated on it with few-shot prompting, which was radical at the time: nobody expected a language model to answer complicated exam questions [00:19:03](ts:00:19:03). Models now score in the 90s. It saturated.

**MMLU-Pro** ([Wang et al., 2024](https://arxiv.org/abs/2406.01574)): removed noisy and trivial questions, expanded 4 choices to 10, evaluated with chain of thought. Accuracy dropped back to the 30s, then climbed to about 88. Also nearly saturated [00:22:16](ts:00:22:16).

**GPQA** ([Rein et al., 2023](https://arxiv.org/abs/2311.12022)), "Google-Proof Q&A": 61 PhD contractors wrote hard questions, with expert validation and revision. The **diamond set** keeps questions where two experts agree and at most one of the non-experts (given 30 minutes plus Google) can solve them. PhD experts score 65%. Non-experts score 34%. GPT-4 scored 39% at release. Current models hit 94 [00:23:12](ts:00:23:12).

**Humanity's Last Exam (HLE)** ([Phan et al., 2025](https://arxiv.org/abs/2501.14249)): 2500 questions, multimodal, multiple-choice plus short answer, crowd-sourced with a $500K prize pool and co-authorship incentives, multiple review stages, and a held-out private set to fight contamination. Initial scores were single digits. As of 2026 the best models reach about 64.7. Still hard [00:27:34](ts:00:27:34).

| Benchmark | Year | Format | What made it harder | Best score now |
|---|---|---|---|---|
| MMLU | 2020 | 57 subjects, 4-choice | First broad exam | ~90s, saturated |
| MMLU-Pro | 2024 | 10-choice, CoT | Removed noise, more distractors | ~88 |
| GPQA | 2023 | 4-choice, diamond set | PhD-written, Google-proof | 94, saturated |
| HLE | 2025 | MC + short answer, multimodal | Prize incentives, private held-out | 64.7, still hard |

The multiple-choice format survives. It restricts which questions you can ask, but you can make multiple-choice questions arbitrarily difficult. The real weakness of exams is different: nobody asks HLE questions to a language model in real life. Most real queries are open-ended and may have no correct answer [00:29:33](ts:00:29:33).

> [!PROF] Train contamination is subtle. It is probably not literal training on the test set. More likely, questions are derived from sources that overlap with the training data. Treat every benchmark number with skepticism [00:26:21](ts:00:26:21).

## Chat benchmarks: evaluating open-ended responses

Most people do not ask exam questions. They ask open-ended questions with no ground truth, like "what herbs work well in a beet salad." You cannot grade that with exact match [00:31:59](ts:00:31:59).

### Chatbot Arena and Elo

Chatbot Arena (now Arena AI, [Chiang et al., 2024](https://arxiv.org/abs/2403.04132)) collects pairwise human preferences [00:32:54](ts:00:32:54):

1. A random person types a prompt.
2. They get responses from two anonymized models.
3. They rate which is better (A better, both good, both bad, B better).

Elo rankings are then fit to these pairwise comparisons:

\[
P(A \text{ beats } B) = \frac{1}{1 + 10^{(ELO_B - ELO_A)/400}}
\]

Advantages: real-world prompts (people use the site because it is free), no need to feed the same prompts to every model (unlike chess, the graph just needs to be connected), and new models and prompts slot in naturally over time.

Problems: who are these people, and what biases do they bring (spammers, submitters gaming their own model)? Binary preference conflates style and correctness. The rater is the same person asking the question, so they often cannot judge correctness. Sycophancy gets upweighted: pleasing answers beat honest ones [00:36:44](ts:00:36:44).

### AlpacaEval and LLM-as-judge

AlpacaEval (2023): 805 instructions, metric is win rate against a baseline (GPT-4 preview at the time), judged by GPT-4 preview itself. This raises obvious bias questions. The initial version had a concrete flaw: LLM judges favor longer responses, so models gamed the leaderboard by emitting longer outputs. AlpacaEval 2.0 fixed it with a simple regression-based debias ([Dubois et al., 2024](https://arxiv.org/pdf/2404.04475)) [00:39:34](ts:00:39:34).

This raises a deeper question: how do you evaluate a metric? One sanity check is correlation with another metric. AlpacaEval correlated at 0.98 with Chatbot Arena, suggesting you can use it as a cheap proxy when you cannot wait for human ratings [00:40:32](ts:00:40:32).

### WildBench and rubrics

WildBench ([Lin et al., 2024](https://arxiv.org/pdf/2406.04770)): 1024 examples sourced from 1M human-chatbot conversations, judged by GPT-4 Turbo with a per-prompt **checklist**. The innovation is the rubric: asking a judge "is this response good" is ill-defined, and a checklist scopes the evaluation. This holds for human judges too. Anyone who has done crowdsourcing knows that rating without a rubric produces nonsense [00:41:34](ts:00:41:34).

Takeaways for open-ended evaluation: pairwise comparisons give higher signal than absolute scores (is it 7/10 or 8/10?). Always be aware of judge bias, human or LLM. Prefer multiple judges. Use rubrics or checklists to make the task well-defined [00:43:22](ts:00:43:22).

## Agentic benchmarks: evaluating what models do

Chat evaluates what models say. Agents evaluate what models do. An **agent** is a language model plus a scaffold: the logic that decides how the model is called and which tools it can use [00:45:05](ts:00:45:05).

| Benchmark | Task | Evaluation | Scale |
|---|---|---|---|
| SWE-Bench ([Jimenez et al., 2023](https://arxiv.org/abs/2310.06770)) | Given a codebase plus a GitHub issue, submit a PR | Unit tests (pass-to-pass, fail-to-pass) | 2294 tasks, 12 repos. 16% in 2024 to 93% now |
| TerminalBench | General tasks in a computer terminal | Task-specific checks | 229 tasks crowdsourced from 93 contributors, 89 of them in 2.0 |
| CyBench ([Zhang et al., 2024](https://arxiv.org/abs/2408.08926)) | Capture-the-flag hacking tasks | Extract the flag string | 40 CTF tasks, solved quickly once agents matured |
| MLE-Bench ([Chan et al., 2024](https://arxiv.org/abs/2410.07095)) | Kaggle competitions: process data, train models | Competition grade | 75 competitions |

Two patterns stand out. First, the **agent scaffold matters enormously**. Two agents with the same model get different scores. The ingredients of good scaffolds: explicit planning (a to-do list that gets checked off), hierarchical delegation (sub-agents with cleaned context, returning only results), persistent memory (read/write files instead of bloating context), and deliberate context engineering [00:51:56](ts:00:51:56). Second, **evaluating agents means evaluating the model plus the scaffold together**.

> [!PROF] Even with good scaffolds, agentic benchmarks rot. CyBench went from ~10% to completely solved. SWE-Bench needed a "Verified" cleanup because unit tests were not rigorous enough. Treat every agentic number as time-stamped.

## Pure reasoning: ARC-AGI

All benchmarks so far require linguistic and world knowledge. Can we isolate reasoning, a purer form of intelligence, from memorized facts? ARC-AGI ([Chollet, 2019](https://arcprize.org/arc-agi)) tries: tasks designed to be 100% human-solvable but hard for AI, each a special snowflake so memorization does not help [00:54:06](ts:00:54:06).

The trajectory is striking. ARC-AGI-1 (2019): pretrained models did not move the needle at all. In 2024, OpenAI's o1 and o3 reasoning models cracked it, and ARC-AGI-1 is now basically solved. ARC-AGI-2 (March 2025) is on its way to solved. ARC-AGI-3 (March 2026) moved to interactive game environments. Current scores are extremely low [00:57:22](ts:00:57:22).

Caveats: you cannot fully decouple reasoning from knowledge. The tasks are constrained to human reasoning within reasonable time, so they do not cover superhuman reasoning like winning IMO gold. And the benchmark creators follow a familiar cycle: models solve it, creators build a harder one [00:58:14](ts:00:58:14).

## Safety benchmarks

What does safety mean for AI? For cars, crash tests are decades of settled engineering. For AI there is no settled answer, but there are attempts [00:00:15](ts:01:00:15).

- **HarmBench** ([Mazeika et al., 2024](https://arxiv.org/abs/2402.04249)): 510 harmful behaviors that violate laws or norms. The model is prompted with them and expected to refuse.
- **AIR-Bench** ([Zeng et al., 2024](https://arxiv.org/abs/2407.17436)): built from EU, Chinese, and US regulatory frameworks plus company policies, taxonomized into 314 risk categories and 5694 prompts.
- **Jailbreaking**: models are trained to refuse, but prompts can be automatically optimized to bypass refusal. GCG (Greedy Coordinate Gradient, [Zou et al., 2023](https://arxiv.org/pdf/2307.15043)) optimizes gibberish suffixes on open models that transfer to closed ones [00:02:14](ts:01:02:14).

Safety is contextual: politics, laws, and social norms vary across countries. Risks are varied: hallucinations (bad in medical or legal settings, but correlated with capability), sycophancy, abetting crimes, inequality, losing critical thinking. And there is dual use: a cybersecurity agent can hack a system or pentest it [00:03:37](ts:01:03:37).

## Realism: ecological validity

How well does the evaluation capture real-world use? Exams are far from it. Chatbot Arena has real people, but the distribution is uncontrolled [00:05:16](ts:01:05:16).

Three attempts at realism:

- **GDPVal** (OpenAI): 44 occupations across the top 9 US GDP sectors, tasks created by professionals with about 14 years of experience: nurses, concierges, real estate agents, video editors ([arXiv](https://arxiv.org/pdf/2510.04374)) [00:05:58](ts:01:05:58).
- **MedHELM** ([Rathnakumar et al., 2025](https://arxiv.org/abs/2505.23802)): medical benchmarks were based on standardized exams. Instead, 121 clinical tasks sourced from 29 clinicians, capturing what clinicians actually ask models [00:06:41](ts:01:06:41).
- **Clio** (Anthropic, [Tamkin et al., 2024](https://arxiv.org/abs/2412.13678)): use language models to analyze real user data and share general patterns of what people ask, since privacy forbids looking at the data directly [00:07:38](ts:01:07:38).

Realism and privacy are often at odds. The ideal sample from the real query stream is also the one you cannot look at.

## Validity: train-test overlap and dataset quality

Machine learning 101 says do not train on your test set. Before foundation models, this was simple: ImageNet and SQuAD had clean train-test splits. Now models train on the internet, and you do not know what is in the data [00:08:36](ts:01:08:36).

Four routes:

1. **Infer overlap from the model.** The order of questions in a benchmark should be random. If a model prefers the benchmark's actual order, it probably trained on it (exchangeability, [Oren et al., 2023](https://arxiv.org/pdf/2310.17623)) [00:10:00](ts:01:10:00).
2. **Reporting norms.** Just as statistics reports confidence intervals, model providers should report train-test overlap ([position paper](https://arxiv.org/abs/2410.08385)).
3. **Fresh evals.** Assume the worst and define new evaluations past the model's cutoff: scrape new webpages, archive papers, GitHub. LiveCodeBench and UncheatableEval take this route. Caveat: timestamps are not always safe, because "fresh" content may be copied from older sources [00:11:04](ts:01:11:04).
4. **Private evals.** Companies use internal codebases. Individuals can use personal writings, like rejected papers never posted online. This is easiest for perplexity, which only needs a good dataset and log probabilities [00:11:47](ts:01:11:47).

Dataset quality is the second validity threat. SWE-Bench had to be cleaned into SWE-Bench Verified because the unit tests were not rigorous enough. GSM8K and MMLU got "Platinum" versions after audits ([Ye et al., 2025](https://arxiv.org/abs/2502.03461)). Agentic benchmarks are even harder to audit because the environment is complex: one paper showed a trivial agent outputting an empty response scoring 38% ([Wetzl et al., 2025](https://arxiv.org/abs/2507.02825)) [00:14:29](ts:01:14:29). Docent (Transluce) uses LLMs to inspect agent traces and detect problems qualitatively, a response to the field's overly quantitative culture.

> [!PROF] Always look at the outputs. Whenever you run a model on a benchmark or build a new one, audit the traces by hand to check you are measuring what you think you are measuring [00:15:23](ts:01:15:23).

## There is no one true evaluation

The lecture closes on purpose [00:15:41](ts:01:15:41). Different goals demand different evaluations:

1. A user or company deciding between model A and B for a use case (customer service chatbots).
2. A researcher measuring raw capability or "intelligence."
3. Understanding benefits and harms for business or policy.
4. A model developer getting feedback to improve the model.

And there is a second question: **what are we evaluating?** Before foundation models, researchers evaluated **methods**: fixed train-test splits, only the algorithm varied. Today we mostly evaluate **models and systems**: anything goes. One exception is the nanoGPT speedrun, which deliberately evaluates an algorithm (training speed to a target loss on fixed data). Evaluating methods encourages algorithmic innovation. Evaluating models serves downstream users. Either way, declare the rules of the game [00:16:28](ts:01:16:28).

Takeaways from the lecture:

- There is no one true evaluation. Choose the evaluation for what you are trying to measure.
- Clearly state the rules of the game: methods versus models versus agents.
- The tradeoffs are difficulty, realism, and validity. You cannot have all three, so choose which to compromise.

## Assignment connection

This lecture directly feeds Assignment 4 (data). Evaluation defines the target before data defines the behavior. When you choose training data, you are implicitly optimizing for some evaluation. The lecture's contamination discussion also matters: if you evaluate your data filtering on standard benchmarks, check whether your pipeline leaked benchmark content into the training data.

> [!INTERVIEW] "How would you evaluate this model?" is a common frontier-lab interview question. The strong answer starts with the purpose: who decides, and what decision the eval informs. Then picks a family (perplexity for development, exams for capability claims, preference for product, private fresh evals for contamination safety) and names the validity threats. Interviewers listen for contamination awareness and the methods-versus-models distinction.
