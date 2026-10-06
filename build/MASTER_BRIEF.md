# MASTER BRIEF — Stanford Frontier AI Learning System v2
# The single prompt every agent on this project reads first.
# Written from Raj's own orders (see USER_PROMPTS.md). Updated 2026-10-06.
#
# READ FIRST, IN THIS ORDER:
#   1. build/REJECTIONS.md — 15 real rejections. Do not repeat any.
#   2. This file — the mission and the laws.
#   3. build/PIPELINE.md — how work is checked (builder/auditor split,
#      measurable gates, fix loop). You are always on one side of it.
#
# NO TOKEN LIMITS. Write until the bar is met. Never truncate, never
# summarize to save tokens. Split long output across file appends.
# Comprehensiveness is mandatory; conciseness means short plain
# sentences (STE100), never short pages.

## Who you serve and who the reader is

You work for Raj. The reader is Raj: a software engineer with ZERO
machine learning background assumed. He is preparing for
research-engineering interviews at frontier labs. Interviews test
DEPTH of knowledge, not breadth. Your material must make him
interview-ready: able to walk through any mechanism from scratch,
with numbers, and answer follow-ups.

## The mission in one line

A textbook-style curriculum for 12 Stanford courses that teaches
from 0 → 0.1 → 0.12 → … → 1 — never 0 → 0.5 → 1. A beautiful story
that chains idea to idea. Not summaries.

## CONTENT LAWS (binding)

1. TEXTBOOK NARRATIVE, every section. The chain: concrete problem →
   first attempt built from zero (hand-worked toy, real numbers) →
   it breaks, DEMONSTRATED with numbers (never listed) → one-sentence
   key question (the hinge) → the new idea built from zero → map back
   (each property answers a named pain) → honest price, with numbers
   → consolidate (tables and lists at the END only, never to
   introduce).
2. ZERO KNOWLEDGE. Every term defined at or before first use. No
   forward references. No unexplained jargon, ever.
3. NO JUMPS. Each subchapter adds exactly ONE increment on the
   previous one. If a subchapter introduces two ideas, split it.
4. SUBCHAPTERS. Every substantial section gets ### subchapters, one
   idea each. Organize like a knowledge base: index → lecture →
   subchapters, clean hierarchy, no clutter.
5. EXPANSION FIRST. Cover EVERYTHING: every concept, angle, number,
   worked example, variant, edge case, implication in the lecture.
   A lesson is done when nothing interview-relevant is missing.
   When in doubt, include. THEN densify: cut ONLY zero-information
   sentences (throat-clearing, restatement, generic transitions).
   NEVER cut a concept, number, example, variant, failure mode, or
   edge case to save space. Short is not dense. Dense = information
   per sentence at full comprehensiveness.
6. DEPTH OVER BREADTH. When forced to choose, go deeper on the core
   mechanism instead of adding another variant. Every paragraph must
   carry a number, a mechanism step, a failure mode, or a decision
   rule — or it gets cut.
7. FULL VARIANT COVERAGE. Every mechanism's family, each with its own
   worked toy or number, so the reader can think from multiple
   dimensions. (RNN: vanilla/LSTM/GRU; attention: self/causal/cross/
   sliding-window/multi-head/linear/Flash; quantization: INT8/GPTQ/
   AWQ/FP8…)
8. WHAT IS USED WHERE. Every mechanism mapped to real production
   models (GPT, Gemini, DeepSeek, Llama, Mistral, Mamba…). Facts
   verified against public sources, current as of October 2026 —
   never from training memory. What is not public is marked
   unknown/[uncertain], never asserted.
9. INTERVIEW Q&A. 6–8 per lesson, each with FULL follow-up answers:
   at least one "walk me through the mechanism" and one applied
   design question.
10. BANNED: bridge lessons that defer real explanation; comparison
    tables that introduce concepts; terms used before they are
    defined; limitations stated without demonstration; the same fact
    stated twice (each fact once, at its natural place).

## FIGURE LAWS (binding)

build/FIGURE_SPEC_STANFORD.md (Raj's own spec) controls EVERY
figure and wins on any conflict:
- Draw only when the figure changes the learner's state (a count, a
  merge, a score, a mask, a move, a new symbol). Never decorate.
- Page audit per page: every unit (heading, equation, code block,
  architecture noun) maps to exactly one figure. A blank cell FAILS
  the page. Architectures and changes always show before → rule →
  after.
- Medium ladder: use the FIRST medium that passes — table, equation,
  ASCII (≤12 lines), mermaid (≤8 nodes), SVG, canvas, three.js.
  Generated still plates only where the ladder demands them.
- Lesson plates: ONE claim each, warm paper #F7F4EE, Anthropic Sans,
  8px grid, shape rules, caption names the source and the shell.
  Chapter plates: dense, end of concept, left = cost without the
  rule, right = cost with it, bottom = tradeoff in one line.
- Reject list enforced: no robot, brain, glowing network, stock
  photos, Comic Sans, watermarks, clip art; no number the code did
  not compute (softmax weights sum to 1 within 0.01).
- As many figures as needed. No blank figure cells. No decorative
  figures.

## MEDIA LAWS

Every lesson page: 1–3 youtube-nocookie.com embeds matched to its
key subchapters (IDs verified real), plus verified "go deeper"
links (papers, docs — each checked live, no dead links).

## STYLE LAWS

ASD-STE100 + humanizer on everything: short sentences, plain words,
zero AI tells. No contractions. No em dashes. No semicolons in
prose. Banned filler: delve, leverage, unlock, robust, seamless,
nuanced, pivotal, landscape, realm, tapestry, underscore, harness,
foster, groundbreaking, cutting-edge, game-changing, holistic,
multifaceted, "it is important to note", "in today's world", "when
it comes to", "at its core".

## HONESTY LAWS

Never hallucinate lecture content. Never invent a number, quote,
benchmark, or professor claim — "Not in source" where the source is
silent. Model internals not public stay unknown. video_ids must be
real (oEmbed-verified).

## HARD CONSTRAINTS

- Images via the media pipeline ONLY. Never touch Raj's OpenRouter
  key for image generation.
- Public GitHub repos (Raj's Oct 6 order — reversed the earlier
  private-only rule, so he can download on his other systems):
  stanford-frontier-ai-foundations, stanford-frontier-ai-language,
  stanford-frontier-ai-agents, ai-podcast-curriculum. Keep only the
  LATEST version of every file; delete redundant/superseded files.
- Course → repo: foundations = cs336, cs229, cs229s, math-ml,
  math-genai, math-genmodels; language = cs224n, cme295, cs329h;
  agents = cs329z, cs329a, mse435.

## CRASH-COURSE LAWS

build/CRASH_COURSE_BAR.md is binding: reading the crash course
alone must make Raj interview-ready for that course's domain —
equivalent to reading every lesson in detail. Multi-chapter;
memory aids (mnemonics, never-confuse pairs, if-this-then-that,
trap cards, one-glance tables); teaching figure per mechanism.
Potency proof: 25+ interview questions per course, 100% answerable
from the crash course alone, iterated until 100%.

## DELIVERY

Per course: QA (build green, 0 broken images, 0 overflow at
375/768/1200px, depth counts, STE100) → zip → GitHub push →
deliver to Raj. One course after another. Never deliver below
this bar.
