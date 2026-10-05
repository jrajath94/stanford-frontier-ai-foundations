---
page_id: cs336-l14
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 14
nav: "L14 · Data Pipeline"
title: "Lecture 14: The Data Pipeline"
summary: "From raw crawl to training tokens: transformation, filtering, MinHash LSH deduplication, data mixing, and synthetic post-training data."
date: "2026-05-13"
instructor: "Percy Liang"
offering: "Spring 2026"
duration: "1:24:34"
video_id: 5sxHosTLPF8
video_title: "Stanford CS336 Spring 2026 Lecture 14: The Data Pipeline"
video_caption: "Original lecture. Percy Liang on transforming, filtering, deduping, mixing, and synthesizing data."
concepts: [data-pipeline, filtering, minhash, lsh, deduplication, data-mixing, regmix, unmax, synthetic-data, post-training]
sources:
  - tag: video
    label: "Lecture 14 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=5sxHosTLPF8
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## How to read this lesson

Lecture 13 covered where data comes from. This is the pipeline:
transform, filter, dedupe, mix, then post-train. **Level 1 (Core):**
transformation, filtering, dedup, mixing. **Level 2 (Deep):** MinHash
LSH internals, RegMix, synthetic post-training data.

## Level 1: Transformation

Raw data is not text. Common Crawl holds HTML (sometimes PDFs,
sometimes repos). HTML-to-text is rule-based: strip boilerplate and
navigation, extract content. Tables are the hard case: simple ones
render as markdown, nested ones force approximation [01:51](ts:01:51).

![Transform](assets/l14-transform.svg "HTML to text is lossy. PDFs are rare and valuable.")

PDFs are a small fraction of the web and disproportionately
valuable: making a PDF means having something to say. Many are
truncated in crawls (they are big), and conversion often means OCR
with a vision model, which is expensive [04:22](ts:04:22).

## Level 1: Filtering

The general skeleton: a small target T of high-quality data, a huge
raw pool R, and the goal of finding the subset of R that looks like
T [06:56](ts:06:56). Filtering must generalize (you already own T)
and be fast (100 trillion tokens). You keep a single-digit fraction.

![Filtering](assets/l14-filtering.svg "Target vs raw. Generative or discriminative scoring, fastText at scale.")

Two classifier types [09:02](ts:09:02). Generative: fit a small
model (KenLM 5-gram) on T, keep low perplexity. Discriminative:
fastText linear classifier, T positive, R sample negative,
threshold. Instantiations: Meta language ID (176 languages).
OpenMathText (rules plus KenLM plus fastText, 15B tokens that beat
20x unfiltered data). GPT-3 (Wikipedia, WebText, books positive).
phi-1 (GPT-4 labels 100k, distill to Random Forest) [11:31](ts:11:31).

Quality has no universal definition. Define what you want (math),
filter for it, get better at it [14:13](ts:14:13).

> [!QA]
> Q: What is the right quality threshold?
> A: There is none in the abstract. The optimal threshold depends on how many tokens you will train on. Michael Ryan's experiment (157M params): DCLM's high-quality data wins early, but once you start epoching the small high-quality pool, unfiltered Resiliparse catches up. Train longer, tolerate lower quality. The real mistake is not tuning the threshold: it is ignoring that your high-quality source is finite and epoching it silently.
> Follow-up: If high-quality data always wins at small scale, why not train only on it?
> A: Because you run out. The 50-epoch trap: uniform mixing of 10T low-quality and 10B high-quality tokens, training 1T tokens, repeats every high-quality token 50 times. Best case you waste compute, worst case you overfit. UniMax caps epochs per source. Simulated epoching downsamples small runs so they feel large-scale scarcity. Always compute the implied epoch count before you pick a mixture.

## Level 1: Deduplication

Exact duplicates (mirrors, forks) and near duplicates (licenses,
headers, templates differing by a comma). The web is weird: one gas
mask description appeared 61,000 times in C4 [25:49](ts:25:49). Why
dedupe: efficiency, avoiding memorization of copyrighted or private
text, and decontamination (test set out of training set)
[26:17](ts:26:17).

![Dedupe](assets/l14-dedupe.svg "Exact and near duplicates. Dedupe globally, decontaminate always.")

C4 did exact dedup on 3-sentence spans, removing all but one: which
rips sentences out of documents and breaks coherence [30:30](ts:30:30).
Design space: item granularity (sentence, paragraph, document),
match definition (exact, common subitem, Jaccard fraction), and what
to remove. Dedupe across the entire dataset, not per source
[48:55](ts:48:55).

## Level 2: MinHash LSH

![MinHash LSH](assets/l14-minhash-lsh.svg "Collision probability equals Jaccard. Bands sharpen it into a threshold.")

Near-duplicate means Jaccard similarity above a threshold (say
0.99). MinHash: hash every element of the set, keep the minimum.
The key property: P(collision) = Jaccard(A, B). The intuition is a
random permutation: the first row in the ordering decides, and it
lands in the intersection exactly Jaccard of the time
[33:58](ts:33:58).

One hash is too stochastic, so LSH sharpens it. Split n hashes into
b bands of r. A band matches if all r agree. The pair collides if
any band matches. This and-or structure makes an S-curve: below the
threshold collisions vanish, above it they approach certainty.
Increasing r moves the curve right (stricter), increasing b moves it
left (looser). The phase transition sits at (1/b)^(1/r), where the
collision probability is 0.64 [47:04](ts:47:04). Real setting from
the dedup literature: b=20, r=450. All linear time.

## Level 1: Data mixing

You have many sources. A mixture is a distribution over them. The
baseline options: vibes (manual), uniform, proportional to token
count. All are heuristic, and proportional has a flaw: a huge
low-quality source eats the budget [50:56](ts:50:56).

![Mixing](assets/l14-mixing.svg "Uniform mixing can silently epoch your best data 50 times.")

The 50-epoch trap is the cautionary example [53:45](ts:53:45).
Uniform mixing, 10T low-quality plus 10B high-quality, training 1T
tokens: low-quality tokens get touched 5%, high-quality tokens
repeat 50 times. UniMax caps epochs per source and reallocates
[58:33](ts:58:33). Keep diversity: literature, code, and papers are
incomparable, and the model needs all of them [52:41](ts:52:41).

## Level 2: RegMix

The principled approach: train a swarm of small models (e.g. 300M
params) on sampled mixtures, fit a regression from mixture weights
to loss, optimize the regressor, train the big model on the
optimum [60:39](ts:60:39).

![RegMix](assets/l14-regmix.svg "Small models predict the mixture. Two leaps of faith carry it to scale.")

Two leaps of faith. The optimizer hunts the extremes, where the
regressor has the least coverage. And the small-scale optimum may
not be the large-scale optimum: at large token counts,
low-quality data becomes acceptable, which small runs cannot see
[64:50](ts:64:50). Simulated epoching fixes the scale mismatch:
downsample the small runs so they feel the same data scarcity as
the big run [67:42](ts:67:42). And never optimize against your
downstream evals directly: upweighting code to ace code evals is
overfitting, not mixing [63:02](ts:63:02).

## Level 2: Synthetic post-training data

Post-training data is task-dependent, and in the open community it
is almost all synthetic. The recipe: define environments (GitHub
repos), define tasks or prompts, collect responses from a strong
teacher [73:48](ts:73:48). Humans are slow and expensive. The
frontier uses hybrid human-AI.

![Post-training](assets/l14-posttraining.svg "Environments, tasks, teachers. Synthetic data at scale.")

OpenThoughts: 1.2M reasoning examples for math, science, code. Two
surprises: better models are not better teachers (QwQ-32B beat
DeepSeek-R1), and sampling 16 generations per question helps
[74:58](ts:74:58). SWE-smith: 50k synthetic tasks by injecting bugs
into real repos, verified [78:06](ts:78:06). SWE-Zero: 300k agent
trajectories with no execution at all (grep-only scaffolds work
because models have internal code semantics), scaled to 12M in
SWE-rebench [79:03](ts:79:03).

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l14-transform.svg" alt="Transform">
<div class="rc-body">
<strong>1. Raw is not text</strong>
<p>HTML to text: strip boilerplate, tables are hard. PDFs: rare,
truncated, OCR-expensive, worth it.</p>
<p class="rc-num">Key: linearization loses things</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-filtering.svg" alt="Filtering">
<div class="rc-body">
<strong>2. Filter for your target</strong>
<p>T small and good, R huge and raw. fastText, KenLM. Quality is
whatever you define.</p>
<p class="rc-num">Key: generalize and be fast</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-quality-threshold.svg" alt="Threshold">
<div class="rc-body">
<strong>3. Budget sets the bar</strong>
<p>Longer training tolerates lower quality. High-quality data is
finite. Epoching flips the winner.</p>
<p class="rc-num">Key: no universal threshold</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-dedupe.svg" alt="Dedupe">
<div class="rc-body">
<strong>4. Dedupe everything</strong>
<p>Exact and near. Gas mask x61,000. Dedupe globally,
decontaminate the test set.</p>
<p class="rc-num">Key: stop paying twice</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-minhash-lsh.svg" alt="MinHash LSH">
<div class="rc-body">
<strong>5. MinHash LSH</strong>
<p>P(collision) = Jaccard. b bands of r sharpen the S-curve.
Threshold (1/b)^(1/r). Linear time.</p>
<p class="rc-num">Key: similarity as hashing</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-mixing.svg" alt="Mixing">
<div class="rc-body">
<strong>6. Mixtures epoch silently</strong>
<p>Uniform plus small high-quality source = 50 epochs. Cap with
UniMax. Keep diversity.</p>
<p class="rc-num">Key: count your epochs</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-regmix.svg" alt="RegMix">
<div class="rc-body">
<strong>7. Learn the mixture</strong>
<p>Small-model swarm, regression, optimize, scale up. Watch the
two leaps of faith. Simulate epoching.</p>
<p class="rc-num">Key: optimize carefully</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l14-posttraining.svg" alt="Post-training">
<div class="rc-body">
<strong>8. Synthetic teachers</strong>
<p>Environments, tasks, teachers. OpenThoughts, SWE-smith,
SWE-Zero. Better is not always the better teacher.</p>
<p class="rc-num">Key: the new data economy</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 14 video.
- DataComp LLM, DCLM, FineWeb, Nemotron papers (filtering).

**Further reading:**
- RegMix, Olmix, UniMax papers (mixing).
- OpenThoughts, SWE-smith, SWE-Zero, SWE-rebench (synthetic data).
- FinePDFs (PDF processing).

**Caveats from these sources.** The epoching experiment is a small
preliminary run (157M params). Trends may shift at scale. LSH
constants (b=20, r=450) are one literature setting, not a universal
prescription. Synthetic-data quality claims are from the open
community. Frontier practice is hybrid and undisclosed.

## Connections to the other courses

- **CS336 Lecture 13:** sources and copyright. This lecture is the
  pipeline that follows.
- **CS336 Lecture 9:** the same small-to-large extrapolation logic
  as scaling laws.
- **CS229:** hash-based near-dup detection generalizes to any
  large-scale retrieval system.

> [!CHEAT]
> **Data pipeline cheatsheet.** Transform: HTML to text (boilerplate out, tables hard), PDFs rare and valuable (truncated, OCR). Filter: T small good, R huge raw, subset like T. fastText linear, KenLM generative. Quality = defined. Threshold depends on token budget (Ryan: quality wins early, loses after epoching). Dedupe: exact (C4 3-sentence spans), near (Jaccard 0.99). Gas mask 61k. Dedupe globally, decontaminate. MinHash: P(collision) = Jaccard. LSH: b bands of r, S-curve, threshold (1/b)^(1/r), 0.64 at center. Mixing: distribution over sources. 50-epoch trap. UniMax caps. Diversity. RegMix: small swarm, regression, optimize, scale. Two leaps of faith. Simulated epoching. Post-training: environments, tasks, teachers. OpenThoughts 1.2M, SWE-smith 50k, SWE-Zero 300k no-exec, 12M scaled.

> [!MEMORY]
> **Count your epochs.** Filtering defines quality, dedup removes waste, and every mixture implies an epoch count: compute it before you train.
