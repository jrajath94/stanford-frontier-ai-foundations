# PIPELINE — how work gets done on this project
# Binding. Updated 2026-10-06. Fixes the 23-prompt failure.

## The failure this kills
Telephone-game briefs, contradictory layered orders, self-certifying
workers, adjective bars, effort bias, amnesia about rejections.

## NON-NEGOTIABLE RULES

1. SINGLE SOURCE OF TRUTH. Every agent reads build/MASTER_BRIEF.md
   and build/REJECTIONS.md VERBATIM first. Coordinators pass FILE
   PATHS, never paraphrases. A paraphrase is a corruption vector.
2. SEPARATION OF POWERS. The agent that builds content NEVER audits
   it. Builder and auditor are different agents. The auditor's job is
   to FAIL the work; it never fixes, never praises.
3. MEASURABLE GATES. No adjective passes a gate. Each gate below has
   a concrete check. A gate passes only on evidence.
4. FIX LOOP. Auditor returns FAIL items (location, gate violated,
   what's missing, what passing looks like). Builder fixes. Auditor
   re-checks. Max 3 rounds → escalate to pipeline coordinator → to me.
5. AGENTIC FIRST. All content judgments by LLM agents. Python ONLY
   for: git operations, image conversion/generation, automated
   measurement (render overflow px, broken-image detection, link HTTP
   status, pattern grep). No Python content generation. No templates.
6. NO TOKEN LIMITS. Write until the bar is met. Never truncate, never
   summarize to save tokens. Split long output across file appends.
   There is no maximum length. Comprehensiveness is mandatory;
   conciseness means short plain sentences (STE100), not short pages.
7. REPO ORDER. Foundations → language → agents. Within: mse435 first
   (Raj's complaint; its L01 is the proof-point for his approval),
   then cs336, cs229, cs229s, math-ml, math-genai, math-genmodels,
   cs224n, cme295, cs329h, cs329z, cs329a. One course fully gated
   before the next starts. Push per repo when its courses pass.
8. ONE COURSE AT A TIME. This box is 2-core. Sequential beats broken.
   No concurrent browser/render workloads, ever.

## ROLES

- PIPELINE COORDINATOR: owns the ledger (per-course gate states),
  spawns builders/auditors/enforcers, enforces the fix loop,
  escalates. Never writes content itself.
- BUILDER (per course): expands/writes lessons from lecture sources.
  Delivers: coverage map (every lecture concept → file:line) + the
  lessons + per-lesson stats.
- AUDITOR (per course): adversarial check against CONTENT GATES.
  Delivers: PASS, or FAIL list. Never fixes.
- FIGURE ENFORCER (per course): page audit + figures per
  build/FIGURE_SPEC_STANFORD.md. Runs only after content gates pass.
- FIGURE AUDITOR: checks FIGURE GATES. Never fixes.
- EXAMINER (per course): 25+ interview questions, answered from the
  text ONLY (simulating the reader), scores. Anything under 100% is
  FAIL with the unanswerable questions listed.
- RENDER CHECK: automated measurement (Playwright): 0 overflow px at
  375/768/1200, 0 broken images, 0 page errors.

## CONTENT GATES (builder → auditor)
- G1 COVERAGE: every lecture concept appears in the coverage map
  with a file:line. Zero unmapped concepts.
- G2 NO-JUMP: each subchapter states what it builds on; auditor
  verifies one increment per subchapter.
- G3 ZERO-KNOWLEDGE: auditor spot-checks 10 terms; each defined at
  first use.
- G4 VARIANTS: every mechanism family covered, each with worked
  numbers.
- G5 USED-WHERE: every mechanism mapped to real models; each fact
  carries a source dated 2026 (or [uncertain]/unknown).
- G6 Q&A: 6–8 per lesson, full follow-up answers.
- G7 DENSITY: sampled paragraphs each carry a number, mechanism
  step, failure mode, or decision rule.
- G8 STYLE: STE100; no contractions, em dashes, semicolons in prose,
  banned filler (grep + read).
- G9 HONESTY: no invented numbers/quotes/claims; [uncertain] where
  unverifiable.

## FIGURE GATES (enforcer → figure auditor)
- F1 page audit table complete; no blank figure cell.
- F2 medium ladder honored: first passing medium used.
- F3 lesson plates: one claim each; before → rule → after; spec
  styling (warm paper #F7F4EE, Anthropic Sans, 8px grid).
- F4 chapter plate per concept: dense, tradeoff footer.
- F5 captions name the source and the shell.
- F6 reject list: no banned content; every number code-computed.
- F7 cross-course symbols reused, never redrawn.

## EXAM GATE
- E1: 25+ questions per course, 100% answerable from the text alone.

## RENDER GATE
- R1: 0 overflow px, 0 broken images, 0 page errors, 3 viewports.

## REPORT SCHEMAS
- BUILDER: coverage map + per-lesson stats (lines, ###, QAs,
  figures) + [uncertain] notes.
- AUDITOR: PASS, or FAIL = [{location, gate, what's missing, what
  passing looks like}]. No fixes. No praise.
- EXAMINER: questions + answers + score + unanswerable list.
- COORDINATOR (to me): per-course ledger — which gates passed, how
  many fix rounds, what escalated. Honest about gaps.
