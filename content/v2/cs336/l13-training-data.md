---
page_id: cs336-l13
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 13
nav: "L13 · Training Data"
title: "Lecture 13: Training Data"
summary: "Data is the most important ingredient: crawling and its limits, copyright and fair use, Common Crawl, quality pockets, a decade of datasets, rules vs classifiers, code data, and license-only training."
date: "2026-05-11"
instructor: "Percy Liang"
offering: "Spring 2026"
duration: "1:19:51"
video_id: -qm0ln33G24
video_title: "Stanford CS336 Spring 2026 Lecture 13: Training Data"
video_caption: "Original lecture. Percy Liang on data: where it comes from, who owns it, and how it gets cleaned."
concepts: [training-data, crawling, copyright, fair-use, common-crawl, c4, the-pile, dclm, the-stack, common-pile]
sources:
  - tag: video
    label: "Lecture 13 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=-qm0ln33G24
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## How to read this lesson

You know how to train, given data (Lectures 2 to 11). You know what
good means (Lecture 12). **Level 1 (Core):** why data is the most
important thing, the stages, crawling and its limits, copyright.
**Level 2 (Deep):** the datasets, filtering schools, code data,
license-only training.

## Level 1: Data is the secret sauce

Data is the most important thing to get right [00:24](ts:00:24). The
Llama 3 paper discloses everything about the architecture and the
training procedure, and says nothing about the data. Two reasons for
the secrecy: competitive dynamics, and copyright liability
[00:58](ts:00:58).

Data arrives in stages, and the trend is one direction [02:18](ts:02:18).

![Training stages](assets/l13-pipeline.svg "Pre, mid, post: large low-quality first, small high-quality last.")

Pre-training takes raw web documents. Mid-training upgrades to higher
quality data and adds long context. Post-training is chat transcripts
and RL environments. The largest new models skip the base-model
checkpoint entirely: just one release [03:30](ts:03:30).

## Level 1: Crawling and its limits

"Trained on the entire internet" does not type-check. The web is live
servers. Unless you are an RL agent, you do not train on servers: a
crawler discovers pages and downloads them [05:16](ts:05:16). And the
crawlable web is much smaller than the web.

![Crawl barriers](assets/l13-crawl.svg "Dynamic pages, auth walls, robots.txt, anti-bot, terms, licenses.")

Dynamic apps and the deep web have no hyperlinks to follow.
Authentication locks the big platforms. Robots.txt plus
"Consent in Crisis": by mid-2023, about half of websites fully
restrict crawling [10:43](ts:10:43). Cloudflare, CAPTCHAs, rate limits,
IP blocks. Terms of service increasingly say no AI training. And then
shadow libraries (LibGen, Anna's Archive) bypass all of it: piracy
from servers in other countries [12:50](ts:12:50).

## Level 1: Copyright and fair use

Everything on the internet is copyrighted. The Copyright Act of 1976
lowered the bar: fixed in a tangible medium is enough, registration
not required (sue later for $65). It lasts 75 years, then the public
domain [14:31](ts:14:31).

![Copyright](assets/l13-copyright.svg "License or fair use. Fair use is semantics and economics, not n-gram overlap.")

You can use copyrighted works two ways. Get a license (Creative
Commons bridges the 75-year wait. Or pay for a deal). Or claim fair
use, section 107: purpose and character, nature of the work, amount
used, effect on the market. Fair use is semantics, not verbatim
overlap: Harry Potter the character is copyrightable, not any
particular book [24:51](ts:24:51). Authors Guild v Google took 11
years and ruled snippets fair use: the precedent the field leans on
[24:11](ts:24:11).

The 2025 picture. Training itself has been ruled fair use (so far,
narrowly). Pirating the books is clearly illegal. Anthropic paid
$1.5 billion to settle, about $3,000 a book. The Meta case followed
the same pattern [27:40](ts:27:40). Note the layering: training can
be fair use while the copying that enabled it is not. And terms of
service are a separate layer on top [27:07](ts:27:07).

> [!QA]
> Q: Is training on copyrighted data legal?
> A: The honest answer has three layers. First, accessing it: robots.txt, terms of service, and anti-bot measures can make the download itself a violation even when training would be fine. Second, copying it: mere copying can violate copyright. The Anthropic case settled at $1.5 billion over pirated books while separately finding training was fair use. Third, training on it: so far ruled fair use in narrow cases, but not settled in general. Practical rule: permissively licensed data is safe, Common Crawl appeals to fair use, and shadow libraries are piracy. This is an active area. Verify with counsel, not with a lecture.
> Follow-up: What is license laundering?
> A: Slapping a permissive license on a work you do not own, or claiming a dataset is permissively licensed because its collection page says so. Collection licenses do not extend to the individual works. The Common Pile project found that many "permissively licensed" datasets on Hugging Face fail at the individual-work level. If you are risk-averse, you must check licenses per work, not per dataset page.

## Level 2: Common Crawl

Common Crawl has run a monthly web crawl since 2007: 3 to 5 billion
pages per dump, about 300 billion total [32:37](ts:32:37). WARC files
hold the raw HTTP responses. WET files hold lossy extracted text.

![Common Crawl](assets/l13-commoncrawl.svg "The open crawl: scale, formats, and why extraction tooling matters.")

The extraction tool matters. Trafilatura and Resiliparse beat the
stock WET conversion (DataComp LLM ablation) [35:32](ts:35:32).
Crawling is conceptually graph traversal. The gory details are the
product: refresh policies, dedup, mirror sites, dynamic URLs
[33:41](ts:33:41).

## Level 2: Quality pockets

The web is not uniform. Three pockets deserve special handling
[36:28](ts:36:28).

![Quality pockets](assets/l13-quality-pockets.svg "Wikipedia, GitHub, arXiv: dense sources, each with its own access pattern.")

Wikipedia (67M articles): download the dumps, do not crawl. Even
"high quality" can be attacked: Carlini's dump-timing poisoning
edited Wikipedia just before the dump and rolled back after
[38:40](ts:38:40). GitHub (420M repos, 28M public): train on
permissive licenses only. The GitHub Archive records every event.
Code matters for reasoning, not just coding. ArXiv (3M submissions
since 1991): LaTeX source, mostly Creative Commons.

## Level 2: A decade of datasets

![Dataset history](assets/l13-dataset-history.svg "From BookCorpus to Nemotron: the filtering arms race.")

BERT (2018): Wikipedia plus Books, documents not sentences. GPT-2
(2019): outgoing links from Reddit posts with >3 karma, 40GB.
CCNet (2019): Wikipedia-like language-model classifier. C4 (2019):
rules only, 156B tokens. GPT-3 (2020): Common Crawl plus WebText plus
Books1/2, quality classifier, fuzzy dedup, 400B tokens. The Pile
(2020): grassroots, including Books3 from the shadow library
Bibliotik. Llama 1 (2022): 1.2T tokens, and announcing Books3 got
them in trouble: the watershed that silenced everyone [60:47](ts:60:47).
RefinedWeb then FineWeb: web-only, 5T then 15T tokens, rules-first.
DCLM (2024): fastText classifier on OpenHermes plus ELI5, keep 1.4%
of 240T tokens, and the magic works. Nemotron (2024): LLM-judged
educational value, synthetic rephrasing, 6T tokens.

## Level 2: Rules vs classifiers

![Rules vs classifiers](assets/l13-rules-vs-classifiers.svg "Two filtering schools. The funnel from 240T to 3T is the whole game.")

Two schools. Rules (C4, Gopher, RefinedWeb): control, no ML bias.
Classifiers (CCNet, GPT-3, DCLM): decide what "good" looks like, train
it, filter everything. The strangest wins: DCLM's classifier trained
on instruction data plus ELI5 beats everything, and nobody fully
knows why [66:04](ts:66:04). The funnel is the point: 240 trillion
tokens in, about 3 trillion out. Filtering is probably the single
most important lever in data processing [79:34](ts:79:34).

## Level 2: Code data and license-only data

![Stack and Common Pile](assets/l13-stack-commonpile.svg "The Stack for code, Common Pile for the risk-averse.")

The Stack: 137 repos to 3TB of permissively licensed code. v2 adds
issues, PRs, and comments linearized as structured events, so the
model learns the development process, not just code. Low-resource
languages get paired with LLVM IR so knowledge transfers from the
shared low-level representation [72:46](ts:72:46).

Common Pile: the risk-averse experiment. Only permissively licensed
data, 8TB total, no synthetic data (data laundering: the models that
generate it trained on unlicensed data). Result: competitive with
2023-era models, not with Qwen [78:52](ts:78:52). Reasonable, not
champion.

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l13-pipeline.svg" alt="Pipeline">
<div class="rc-body">
<strong>1. Stages, one direction</strong>
<p>Pre (raw web), mid (quality, long context), post (chat, RL).
Large low-quality first, small high-quality last.</p>
<p class="rc-num">Key: blur the lines, keep the trend</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-crawl.svg" alt="Crawl barriers">
<div class="rc-body">
<strong>2. The crawlable web is small</strong>
<p>Dynamic pages, deep web, auth walls, robots.txt, anti-bot,
terms, licenses. Restrictions doubled after 2023.</p>
<p class="rc-num">Key: not the whole internet</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-copyright.svg" alt="Copyright">
<div class="rc-body">
<strong>3. License or fair use</strong>
<p>Everything is copyrighted. License it, or claim the four
factors. Training: fair use so far. Pirating: illegal.</p>
<p class="rc-num">Key: three layers of risk</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-commoncrawl.svg" alt="Common Crawl">
<div class="rc-body">
<strong>4. Common Crawl</strong>
<p>Monthly since 2007, ~300B pages. WARC raw, WET lossy.
Trafilatura and Resiliparse extract better text.</p>
<p class="rc-num">Key: the open raw material</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-quality-pockets.svg" alt="Quality pockets">
<div class="rc-body">
<strong>5. Quality pockets</strong>
<p>Wikipedia (dumps, poisonable), GitHub (licenses, archive),
arXiv (LaTeX, CC). Code trains reasoning.</p>
<p class="rc-num">Key: dense sources, special handling</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-dataset-history.svg" alt="Dataset history">
<div class="rc-body">
<strong>6. The dataset timeline</strong>
<p>Reddit links to classifiers to synthetic data. Books3 was the
watershed. Sizes grew. Secrecy grew faster.</p>
<p class="rc-num">Key: filtering arms race</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-rules-vs-classifiers.svg" alt="Rules vs classifiers">
<div class="rc-body">
<strong>7. The funnel is the lever</strong>
<p>Rules vs classifiers. 240T tokens in, ~3T out. DCLM keeps 1.4%
and wins. Filtering decides the model.</p>
<p class="rc-num">Key: 240T to 3T</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l13-stack-commonpile.svg" alt="Stack and Common Pile">
<div class="rc-body">
<strong>8. Code and caution</strong>
<p>The Stack: process, not just code. Common Pile: license-only
works, reasonably. Watch license laundering.</p>
<p class="rc-num">Key: process data, honest licenses</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 13 video.
- Llama 3 paper (what it says about data, and what it does not).

**Further reading:**
- "Consent in Crisis" (Longpre et al.).
- The Pile, C4, RefinedWeb/FineWeb, DCLM, Nemotron, Common Pile papers.
- The Stack v2. Carlini's Wikipedia poisoning paper.
- The Anthropic and Meta copyright rulings (active litigation area).

**Caveats from these sources.** Copyright law is evolving fast.
The 2025 rulings are narrow, not blanket permission. Dataset sizes
mix unique and repeated tokens (epochs count twice). Compare with
care. "Books3 got Meta in trouble" refers to the Llama disclosure,
not a single court ruling.

## Connections to the other courses

- **CS336 Lecture 12:** eval defines the target. Data is what must
  hit it.
- **CS336 Lecture 14:** post-training data and deeper filtering.
- **CS229:** the same copyright questions apply to any training
  corpus.

> [!CHEAT]
> **Training data cheatsheet.** Data is the secret sauce. Llama 3 hides it. Stages: pre (raw web), mid (quality, context), post (chat, RL). Big low-quality to small high-quality. Crawling: dynamic web, deep web, auth walls, robots.txt (~50% restricted by mid-2023), anti-bot, terms, licenses. Shadow libraries = piracy. Copyright: everything copyrighted, 75 years, license (CC) or fair use (4 factors). Training = fair use (narrow 2025 rulings), pirating = illegal (Anthropic $1.5B). Common Crawl: monthly since 2007, ~300B pages, WARC/WET, trafilatura better. Pockets: Wikipedia (dumps, poisonable), GitHub (permissive only), arXiv (LaTeX, CC). Datasets: BERT docs, GPT-2 Reddit links, CCNet classifier, C4 rules, GPT-3 classifier+dedup, Pile (Books3), Llama 1 (watershed), Refined/FineWeb, DCLM (1.4% kept), Nemotron (synthetic). Rules vs classifiers. The funnel is the lever. Stack: code process as data. Common Pile: license-only, reasonable.

> [!MEMORY]
> **Filter, license, repeat.** The crawlable web is small, filtering is the biggest lever, and license-only training is possible but costly.
