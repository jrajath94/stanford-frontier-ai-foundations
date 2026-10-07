# math-genmodels: Mathematical Foundations of Generative Models

Date: 2026-10-06. Baseline: October 6, 2026. Builder: course-builder subagent.

## HONESTY BANNER (read first)

Exact source identity is UNRESOLVED. This course is a provisional
independent mathematical bridge. It is NOT asserted official syllabus.
The supplied short link resolves only to a LinkedIn interstitial page.
The instructor teaching page names nearby IISc courses (E9 333,
E1 285o/286o, "Advanced Deep Representation Learning / Deep Generative
Models") but not this exact title. The teaching repository
Chandan-IISc/IITM_GenAI holds notebooks and note PDFs whose pages were
not inspected.

Every leaf concept carries the status PLANNED / SOURCE ATTRIBUTION
PENDING until an artifact is inspected and the leaf is mapped to it.
A listed topic is not taught until its lesson, visual, exercise, and
assessment exist and pass the RUN 6 audit. No claim of official-syllabus
coverage is made anywhere in this package.

## What this package is

An 8-unit mathematical bridge from probability to modern generative
models: probability and density estimation, latent models and
variational inference, autoencoders, normalizing flows, adversarial and
ratio estimation, scores and diffusion, autoregressive models and modern
bridges, evaluation and experimental synthesis. Each unit holds 12
atomic concepts. Each concept gets a motivating toy, a justified
derivation or mechanism, a computed numerical example, runnable code,
correctness checks, cost analysis, a failure case, an alternative
comparison, a falsifiable research extension, and assessment with
separate answer keys.

Target roles: research scientist, research engineer, FDE, AI/ML
engineering, LLM engineering, MLOps, agent engineering. Each unit maps
role-specific gates in role_gap_map.md.

## How to read

1. Start at index.md for the path and at prerequisites.md for the entry check.
2. Read lessons/u0N.md in unit order. U01 is the foundation. U08 closes the loop.
3. Run the lab in labs/lab-u0N.md after each lesson. Check answers only in labs/keys/.
4. Figures live in visuals/u0N/. Each render script prints the audited numbers it drew.
5. Interview banks live in interview/. Questions and keys are separate files.
6. Capstones live in capstones/. One is a research replication with an extension. One is applied/FDE.
7. crash-course.md is the fast path. cheatsheet.md is the one-page recall sheet.

## Coverage

96 concept rows in the ledger. Coverage math lives in coverage_matrix.md
with explicit denominators. A row closes only as TAUGHT+ASSESSED with
evidence, or SOURCE-UNREACHABLE with evidence. See state.md for the
live count.

## Shared prerequisites

Shared bridge modules P01-P24 live in
~/workspace/stanford-frontier-ai/v2-pack/shared/prerequisites/ and are
linked, not rebuilt. Each unit states its P-codes up front and still
carries local remediation inside its lesson, because a link alone is
not teaching.

## Visual system

Every figure follows visual_system_generic.md: one claim per plate,
before/after panels, computed numbers, source label, alt text, and an
audit row in visual_audit.md. Figures are matplotlib-computed with
committed render scripts. PNG metadata is stripped and verified with
PIL before ship.

## Integrity

ASD-STE100 prose. Layer A watermark cleaning on every deliverable.
No invented numbers, timestamps, benchmarks, or test results.
Claim classification with research dates on every factual claim.
Academic integrity: no solved assessed assignments, no copied
implementations, no exam dumps, no bypassed authentication.
