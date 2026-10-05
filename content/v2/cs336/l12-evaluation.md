---
page_id: cs336-l12
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 12
nav: "L12 · Evaluation"
title: "Lecture 12: Evaluation"
summary: "What makes a model good: perplexity, the exam treadmill, chat evaluation, agents, ARC, safety, contamination, and why purpose picks the benchmark."
date: "2026-05-06"
instructor: "Percy Liang"
offering: "Spring 2026"
duration: "1:18:26"
video_id: JpAxdTWQJxM
video_title: "Stanford CS336 Spring 2026 Lecture 12: Evaluation"
video_caption: "Original lecture. Percy Liang tours evaluation: perplexity, exams, chat, agents, reasoning, safety, and validity."
concepts: [evaluation, perplexity, MMLU, GPQA, chatbot-arena, SWE-bench, ARC, contamination, safety]
sources:
  - tag: video
    label: "Lecture 12 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=JpAxdTWQJxM
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## How to read this lesson

Data shapes the model, and next week is data. But first: what behavior
do we want? **Level 1 (Core):** what "good" means, perplexity, the
exam treadmill. **Level 2 (Deep):** chat eval, agents, ARC, safety,
validity, and the philosophy.

## Level 1: What is good?

![What is good](assets/l12-what-is-good.svg "Benchmarks, cost, preference, usage: four lenses on good.")

Four lenses, no correct answer [02:38](ts:02:38). Benchmarks
(Artificial Analysis intelligence index). Cost (intelligence vs
inference price: correlated, not aligned). Preference (Arena: which
answer people like). Usage (OpenRouter: what people pay for). The
deep point: evaluation sets North Stars, and North Stars shape what
gets built [01:32](ts:01:32).

Evaluation turns an abstract construct (reasoning, conversation) into
a concrete metric over concrete prompts [02:01](ts:02:01). That
translation is the whole challenge.

## Level 1: Perplexity

A language model is a distribution P(x). Perplexity asks how much mass
it puts on the test set [05:30](ts:05:30). The 2010s paradigm:
in-distribution, train split to test split, PTB and WikiText-103.
GPT-2 (2019) broke it: train on WebText, evaluate zero-shot on
everyone else's benchmarks. Out-of-distribution became the norm
[07:52](ts:07:52).

![Perplexity](assets/l12-perplexity.svg "Best perplexity is the entropy of truth. The catches: boring tokens, disguised cloze, untrusted leaderboards.")

"Perplexity is all you need": the unique minimizer is the true
distribution, and the true distribution solves everything [09:46](ts:09:46).
Belief in this drove scaling before the gains were obvious.

Catches. Perplexity charges bits for boring tokens ("founded" vs
"1885"). Conditional perplexity focuses on the tokens that matter
[11:43](ts:11:43). Cloze benchmarks are perplexity in disguise:
LAMBADA and HellaSwag are next-token prediction with carefully chosen
positions [13:22](ts:13:22). And perplexity leaderboards need trust:
un-normalized logprobs cheat, so the probabilities must be real
[15:18](ts:15:18). Downstream tasks are black-box and checkable.
Perplexity is white-box and trust-based.

> [!QA]
> Q: If perplexity is the true objective, why do we need any other benchmark?
> A: Three reasons. First, perplexity spends capacity on every token equally, including tokens nobody cares about. Conditional perplexity and task benchmarks focus the measurement. Second, upstream-to-downstream transfer is uncertain (Lecture 9): a better perplexity does not guarantee a better assistant. Third, perplexity cannot be checked black-box. You must trust the reported probabilities. Benchmarks exist because the cleanest metric is not always the most honest or the most relevant one.
> Follow-up: Why did GPT-2's zero-shot evaluation matter?
> A: It changed the contract. In-distribution evaluation rewards memorizing one dataset's quirks. Zero-shot rewards a model that generalizes to tasks it never trained on. That reframing, from "fit this distribution" to "handle anything," is what turned language models from research artifacts into general-purpose tools.

## Level 1: The exam treadmill

![Exam treadmill](assets/l12-exam-treadmill.svg "MMLU to HLE: each benchmark saturates, each replacement is harder.")

Exams give control over subject and difficulty, with unambiguous
answers [18:18](ts:18:18). MMLU (2020): 57 subjects, few-shot, radical
at the time. GPT-3 barely above chance on small models, now 90s
[19:07](ts:19:07). MMLU-Pro: 10 choices, chain of thought, 33 to 88.
GPQA: PhD contractors, diamond validation, experts at 65%, models now
at 94 [23:20](ts:23:20). Humanity's Last Exam: crowdsourced, private
held-out set, Mythos at 64.7: still hard [27:29](ts:27:29).

Multiple choice survives: it can be arbitrarily hard. Its limit is
realism: nobody asks HLE questions except on HLE [30:03](ts:30:03).
And contamination shadows every number: the test set lives on the
internet, and so does the training data [26:14](ts:26:14).

## Level 2: Chat evaluation

Open-ended questions have no ground truth. Chatbot Arena: two
anonymized responses, humans pick, ELO ranking [32:58](ts:32:58).
Virtues: real prompts, sparse comparisons, natural updating. Vices:
unknown rater distribution, style conflated with correctness,
sycophancy rewarded, judges who do not know the answer judging
correctness [35:52](ts:35:52).

![Chat eval](assets/l12-chat-eval.svg "Arena, AlpacaEval, WildBench: pairwise, LLM judges, checklists.")

AlpacaEval: LLM judge vs a baseline, win rate. The length bias got
gamed, then debiased [38:26](ts:38:26). WildBench: per-prompt
checklists make judging well-defined [41:39](ts:41:39). Always evaluate
the metric: AlpacaEval correlated 0.98 with Arena, on the models of
that era [40:36](ts:40:36). Pairwise beats absolute. Rubrics beat vibes.

## Level 2: Agent evaluation

Evaluate what the model does, not what it says. SWE-bench: fix the
GitHub issue, pass the tests. 16% to 93% (Verified cleaned the bad
tasks) [46:01](ts:46:01). Terminal-Bench: a terminal, crowdsourced
tasks, hours to weeks. CyBench: 40 CTFs, now solved. MLE-bench:
Kaggle competitions.

![Agent eval](assets/l12-agent-eval.svg "SWE-bench, Terminal-Bench, CyBench: environments with checkable outcomes.")

The scaffold is half the score: same model, different agent, different
accuracy [48:57](ts:48:57). Planning, delegation, memory, context
engineering: all tuned per model. And benchmarks need audits: an empty
response scored 38% on TorchBench. Docent inspects traces for such
pathologies [74:50](ts:74:50).

## Level 2: ARC and pure reasoning

![ARC](assets/l12-arc.svg "Human-easy grids, knowledge-free. Reasoning models finally moved the needle.")

ARC-AGI (2019): grids solvable by humans in seconds, designed so
knowledge does not help [53:31](ts:53:31). GPT-3: zero. Then o1/o3:
ARC-1 solved, ARC-2 nearly, ARC-3 (interactive) now the frontier with
low scores [56:28](ts:56:28). Caveat: "pure" reasoning may not exist,
and the benchmark is human-bounded, not superhuman [58:16](ts:58:16).

## Level 2: Safety

HarmBench: harmful prompts, expect refusal [61:05](ts:61:05).
AIR-Bench: taxonomy from EU, China, and US regulations
[61:38](ts:61:38). Jailbreaking (GCG) optimizes past refusals, and
attacks transfer from open to closed models [62:28](ts:62:28). Safety
is contextual: politics, law, norms vary. Risks vary too:
hallucinations, sycophancy, crime, lost critical thinking. Dual use
cuts both ways: the hacking agent is also the pentesting agent
[64:48](ts:64:48).

| Benchmark | What it tests |
|---|---|
| HarmBench | harmful prompts, expect refusal |
| AIR-Bench | taxonomy from EU, China, US regulations |
| GCG jailbreaks | attacks optimize past refusals, transfer to closed models |

## Level 2: Validity

**Ecological validity**: does the eval match real use? GDPVal:
professionals across GDP sectors writing real tasks
[66:00](ts:66:00). Clinical tasks from 29 clinicians, not med-school
exams [67:11](ts:67:11). Claude usage analysis: let models summarize
real query patterns, privately [67:39](ts:67:39).

**Scientific validity**: contamination and quality.

![Contamination](assets/l12-contamination.svg "Detect, report, refresh, privatize: four defenses against training on the test.")

- Detect: question-order preference betrays memorization
  [69:50](ts:69:50).
- Report: norms demanding no-train-on-test justification
  [70:26](ts:70:26).
- Fresh evals: LiveCodeBench scrapes past the cutoff
  [73:31](ts:73:31).
- Private evals: internal code, rejected papers, HLE's held-out
  [73:17](ts:73:17).

Dataset quality needs audits: SWE-bench Verified, MMLU cleanups, and
always look at the outputs yourself [74:36](ts:74:36).

> [!QA]
> Q: Your model scores 94 on GPQA. Should you celebrate?
> A: Cautiously. First, contamination: the questions live on the internet, and contamination is subtle (sources, not just the test set). Second, saturation dynamics: GPQA went from 39% (GPT-4) to 94%, so the benchmark is near its ceiling and discriminates poorly at the top. Third, purpose: GPQA measures exam QA, not your deployment. Celebrate the HLE number more, audit for contamination always, and check the capability you actually ship.
> Follow-up: When is a private eval worth the cost?
> A: When the benchmark is both high-stakes and public. Public benchmarks get trained on, deliberately or via data leakage, and their signal decays. A private eval (internal code, unpublished writing, held-out sets) preserves the signal because the model cannot have seen it. The cost is maintenance: you must keep it private and refresh it. For purchase decisions and safety claims, that cost is worth it.

## Level 2: The philosophy

No one evaluation rules all [75:41](ts:75:41). Purpose picks the
benchmark: buying (ecological validity), measuring intelligence
(hard exams, ARC), improving the model (perplexity), shipping safely
(safety benches). And we evaluate models and systems now, not methods:
anything goes, which is why the shipped artifact is what matters
[76:45](ts:76:45). Declare the purpose first.

![Eval purpose](assets/l12-eval-purpose.svg "Buy, measure, improve, ship: each purpose wants its own benchmark.")

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l12-what-is-good.svg" alt="What is good">
<div class="rc-body">
<strong>1. Define good first</strong>
<p>Benchmarks, cost, preference, usage. Evaluation sets North Stars.
The star you pick shapes what gets built.</p>
<p class="rc-num">Key: purpose precedes metric</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-perplexity.svg" alt="Perplexity">
<div class="rc-body">
<strong>2. Perplexity is the root</strong>
<p>P(x) on the test set. Best is the entropy of truth. Charges for
boring tokens. Leaderboards need trust.</p>
<p class="rc-num">Key: mass on the test set</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-exam-treadmill.svg" alt="Exam treadmill">
<div class="rc-body">
<strong>3. Exams saturate</strong>
<p>MMLU to HLE: each harder, each with a shelf life. Multiple choice
survives. Realism does not.</p>
<p class="rc-num">Key: shelf lives</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-chat-eval.svg" alt="Chat eval">
<div class="rc-body">
<strong>4. Judge the judge</strong>
<p>Arena: pairwise ELO, real prompts, unknown raters. AlpacaEval:
LLM judge, debiased. Checklists scope judgment.</p>
<p class="rc-num">Key: pairwise beats absolute</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-agent-eval.svg" alt="Agent eval">
<div class="rc-body">
<strong>5. Agents do things</strong>
<p>SWE-bench, Terminal-Bench, CyBench: checkable outcomes. The
scaffold is half the score. Audit the traces.</p>
<p class="rc-num">Key: scaffold counts</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-arc.svg" alt="ARC">
<div class="rc-body">
<strong>6. Isolate reasoning</strong>
<p>ARC: knowledge-free grids. Reasoning models solved ARC-1. ARC-3
is the new frontier. Human-bounded.</p>
<p class="rc-num">Key: reasoning vs knowledge</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-contamination.svg" alt="Contamination">
<div class="rc-body">
<strong>7. Assume the worst</strong>
<p>Detect order effects, report overlap, refresh evals, keep some
private. Contamination is subtle.</p>
<p class="rc-num">Key: trust, then verify</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l12-eval-purpose.svg" alt="Eval purpose">
<div class="rc-body">
<strong>8. Purpose picks the bench</strong>
<p>Buy, measure, improve, ship: different goals, different evals. We
evaluate shipped systems now, not methods.</p>
<p class="rc-num">Key: declare the purpose</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 12 video.
- MMLU, GPQA, HLE, SWE-bench, ARC-AGI papers and leaderboards.

**Further reading:**
- Chatbot Arena / AlpacaEval / WildBench papers.
- HarmBench, AIR-Bench. GCG jailbreak paper.
- GDPVal. The contamination-detection work from Hashimoto's group.

**Caveats from these sources.** Benchmark numbers are snapshots and
decay with contamination and saturation. "Mythos" is the lecture's
name for a 2026 frontier model. Treat leaderboard figures as
illustrative. Arena demographics are uncharacterized: preference is
not quality.

## Connections to the other courses

- **CS336 Lecture 9:** perplexity scaling vs downstream transfer.
- **CS336 Lecture 13:** data shapes the model. Eval defines what the
  data must produce.
- **CS329H:** RLHF needs the judge: preference eval becomes training
  signal.

> [!CHEAT]
> **Evaluation cheatsheet.** Good = benchmarks, cost, preference, usage. Eval = abstract construct to concrete metric. Perplexity: P(x) mass on test, best = entropy, GPT-2 zero-shot broke in-distribution. Catches: boring tokens, cloze in disguise, trust. Exams: MMLU (57, few-shot, 90s), MMLU-Pro (10-way, 88), GPQA (PhD, diamond, 94), HLE (private, 64.7). Chat: Arena (pairwise ELO, biases), AlpacaEval (LLM judge, debiased), WildBench (checklists). Agents: SWE-bench (93% verified), Terminal-Bench, CyBench (solved), MLE-bench. Scaffold = half the score. ARC: reasoning vs knowledge, o1/o3 solved, ARC-3 now. Safety: HarmBench, AIR-Bench, GCG, contextual, dual use. Validity: GDPVal, clinician tasks, detect/report/fresh/private contamination defenses, audit quality. Philosophy: purpose picks the bench, evaluate systems not methods.

> [!MEMORY]
> **Declare the purpose, then measure.** No benchmark rules all. Every number needs a contamination check.
