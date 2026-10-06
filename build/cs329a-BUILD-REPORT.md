# CS329A Build Report

**Date:** 2026-10-06. **Builder:** subagent 9fe7dc11 (depth 2/2).
**Deliverable:** 8 lesson files in `content/v2/cs329a/` + `coverage/cs329a.md` + updated `index.md`.

## Per-lesson stats

| Lesson | File | Lines | Narrative subchapters | Q&As | Figure refs |
|---|---|---|---|---|---|
| L01 | l01-course-overview.md | 455 | 10 | 7 | 5 |
| L02 | l02-test-time-scaling.md | 351 | 7 | 7 | 3 |
| L03 | l03-feedback-tools-code.md | 309 | 4 | 7 | 2 |
| L04 | l04-planning-multistep-reasoning.md | 311 | 4 | 7 | 2 |
| L05 | l05-search-at-scale.md | 307 | 4 | 7 | 2 |
| L06 | l06-train-time-scaling.md | 324 | 5 | 7 | 3 |
| L07 | l07-agentic-evaluations.md | 315 | 4 | 7 | 2 |
| L08 | l08-future-directions.md | 360 | 6 | 7 | 3 |
| **Total** | | **2,732** | **44** | **56** | **22** |

Every Q&A carries a follow-up answer. Every figure ref is a caption line
carrying exact numbers, shell, source, and project tag; `assets/` is empty
(stale plates deleted, see below) so the figure enforcer builds all 22 from
captions.

## Coverage

`coverage/cs329a.md`: 184 concepts mapped to file:line, **zero unmapped**
(verified by script against final text, hyphen-normalized).

## Video verification

- oEmbed check: **blocked**. `https://www.youtube.com/oembed?url=...`
  returned "Unauthorized" for every ID at the network level, and the
  nocookie-embed 200-check was proven useless (a fabricated ID also
  returned 200). No video ID in this build rests on oEmbed.
- Instead, each ID was corroborated by **two or more independent
  third-party sources** (GitHub lecture guides, learnaidoc wiki,
  howardism blog): `6YnLB0XbTnI` Part 1 Course Overview,
  `-Ggc37xLj_Y` Part 2 Test-Time Compute Scaling, `Lxh9RF5S-K0`
  Part 4 Learning from Feedback with Tools and Code, `Ml_fp9XkB8Y`
  Part 5 Planning and Multi-Step Reasoning, `yVnmHSAy3ck` Part 6
  Train-Time Scaling and Scaling RL, `Uni9dqyuuDM` Part 7
  Self-Improvement and Deep Research Agents, `8JAqLnTaZu4` Part 8
  Agentic Evaluations and Long-Horizon Tasks, `AyO6wyu4DEg` Part 9
  Future Research Areas. Videos taught Autumn 2025 (Part 1:
  2025-09-22, Part 6: 2025-10-10, Part 9: 2025-12-05), published to
  YouTube August 2026.

## arXiv verification (curl, 22/22 title-matched)

2407.21787 Monkeys, 2408.03314 Snell, 2409.15254 Archon, 2210.03629
ReAct, 2410.02089 RLEF, 2212.08073 Constitutional AI, 2310.04406 LATS,
2506.05745 SPRINT, 2504.04736 SWiRL, 2203.07814 AlphaCode, 2501.05366
Search-o1, 2203.14465 STaR, 2402.03300 DeepSeekMath, 2503.14476 DAPO,
2503.14499 METR, 2510.04374 GDPval, 2508.20033 DeepScholar-Bench,
2501.05707 Multi-agent FT, 2511.22570 DeepSeekMath-V2, 2505.03335
Absolute Zero, 2506.13131 AlphaEvolve, 2502.10517 KernelBench.
Two misattributions caught and fixed during the build: `2506.10910`
is Magistral, not AlphaEvolve (rejected); DeepSeekMath-V2 was briefly
linked to GDPval's ID in L08 further reading (fixed to 2511.22570).

## Used-where (October 2026, search-verified)

METR Time Horizon 1.1 (Jan 2026, 228 tasks, ~88.6-day doubling,
frontier 50% horizon past 16h with >16h flagged unreliable);
GPT-5.2 Thinking 70.9% win-or-tie on GDPval (Dec 2025);
KernelBench active (ICML'25 + Meta KernelBench-Verified);
DeepSeekMath-V2 gold IMO 2025, 118/120 Putnam 2024;
AlphaEvolve 48-scalar-mult 4x4 complex matmul; Claude Sonnet 4.5
100+ tool calls; hybrid local/cloud serving.

## [uncertain] flags for the auditor

1. **Part 3 (Robust Verification) has no transcript** in
   `sources/agents/cs329a/`. All verifier-construction claims rest
   on Parts 1/2/4/8. Named in L02 ("What the lecture leaves out")
   and the index.
2. **oEmbed blocked** (above); video IDs are third-party
   corroborated, not first-party verified.
3. **Lecture-vs-paper deltas**, each attributed in-text: Archon
   14.1% (lecture, paper v1) vs 15.1 (paper v6); DeepScholar-Bench
   19% (lecture telling) vs 31% geometric-mean ceiling (paper);
   SPRINT 6,000 raw trajectories vs 1,700 curated demos; DAPO
   ladder as reported in lecture; Grok 4 "50% RL" marked as
   unverified gossip.
4. Middle lecture dates: only Parts 1, 6, 9 have sourced dates;
   frontmatter uses "Autumn 2025".
5. Hand-computed toys (pass@25 = 0.9591, power-law toy 0.15/0.21/
   0.30/0.42, UCT = 1.408, backprop = 0.625, GRPO advantages
   -1.414/0/0/1.414, METR toy 41.6 min) were Python-verified.

## Cleanup performed

Deleted the superseded 10-lesson build: l01-l10 old files,
cheatsheet.md, crash-course.md, and all 52 stale `plate-*.svg`
assets (tied to replaced text). `index.md` rewritten for the
8-lesson arc. Crash course and cheatsheet are intentionally absent;
they regenerate after content gates per the pipeline.

## Style compliance

Zero contractions, zero em dashes, zero semicolons in prose
(9 fixed), banned filler removed ("leverage" x3 fixed).
Every acronym expanded at first use. Figure captions name source
and shell.

## For the figure enforcer

22 plates to build in `content/v2/cs329a/assets/`, all SVG,
Python-first for number-dense: plate-l01-scaling, plate-l01-cot,
plate-l01-monkeys, plate-l01-loop, plate-l01-test-train,
plate-l02-coverage, plate-l02-small-beats-big, plate-l02-archon,
plate-l03-react-loop, plate-l03-two-tiers, plate-l04-lats,
plate-l04-swirl, plate-l05-alphacode, plate-l05-search-o1,
plate-l06-star, plate-l06-grpo, plate-l06-dapo, plate-l07-horizon,
plate-l07-deepscholar, plate-l08-debate, plate-l08-meta,
plate-l08-absolute-zero. Captions in the lessons carry exact
numbers. Cross-course symbols (agent loop) are reused from
CS329Z, never redrawn.
