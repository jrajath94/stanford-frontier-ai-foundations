# Lesson Spec v2 — FDE-Dossier Style (2026-10-05)

## The reader
A strong software engineer with ZERO ML background. Define every ML term at first use.
He skips what he knows. Never assume. Never forward-reference.

## The unit
One lesson = one lecture's topics, in the professor's order. The lecture supplies
CONTENT and SEQUENCE only. Each concept inside gets rebuilt from zero.

## Textbook narrative (binding)

Every lesson is a textbook chapter, not a slide summary. The reader
must be able to learn the topic from the lesson alone, with zero
background. Follow the narrative chain:

1. PROBLEM, concrete. Start with a real example the reader can touch.
   Name the job to be done in plain words before any term appears.
2. FIRST ATTEMPT, built from zero. Rebuild the historical first idea
   step by step with a hand-worked toy (at most 8 tokens or 4 numbers).
   Define each term the moment it appears, with a concrete instance.
3. WHERE IT BREAKS, shown not stated. Each limitation gets a worked
   demonstration with numbers. Never list pains; prove them.
4. THE KEY QUESTION. One sentence that pivots from the pain to the
   new idea. This sentence is the hinge of the chapter.
5. THE NEW IDEA, built from zero. Same treatment as step 2: toy first,
   mechanism second, formalism last.
6. MAP BACK. Each property of the new idea answers one specific pain
   from step 3, named explicitly. The reader sees why it won.
7. THE HONEST PRICE. Every idea costs something. Name the tradeoff
   with numbers. Bridge to where the course goes next.
8. CONSOLIDATE. Summary tables and lists go at the END of a section
   to lock in what was built, never at the start to introduce it.

Banned: "bridge lessons" that defer the real explanation to another
page. Each lesson is self-contained; cross-links are "for more",
never "go there to actually learn it". Banned: introducing a concept
with a comparison table. Banned: using a term before its concrete
definition. Banned: stating a limitation without demonstrating it.

## Depth bar (binding, from the user's Oct 5 orders)

The gold standard is content/v2/cs229s/l02-sequence-models.md (19
sections, 21 subchapters, 8 Q&As, 11 plates). Every lesson must
clear this bar:

1. SUBCHAPTERS. Every substantial ## section gets ### subchapters
   that go deep: variants, mechanisms, edge cases. No ## section
   that covers a mechanism family may have zero subchapters.
2. FULL VARIANT COVERAGE. Every mechanism's family is covered in
   depth: all major types (RNN: vanilla/LSTM/GRU; attention:
   self/causal/cross/sliding-window/multi-head/linear/Flash),
   each with its own worked toy or number. The reader must be able
   to think from multiple dimensions.
3. WHAT IS USED WHERE. Every lesson maps its mechanisms to real
   production models (GPT, Gemini, DeepSeek, Llama, Mistral,
   Mamba, etc.): which variant each uses and why. Facts verified
   against public model cards and papers; mark what is not public.
4. IMAGES AS MANY AS NEEDED. Every subchapter that builds a
   mechanism gets its own plate per build/VISUAL_SYSTEM.md. No
   blank figure cells. Captions name the project.
5. INTERVIEW Q&A: 6 to 8 per lesson, each with full follow-up
   answers, including at least one "walk me through the mechanism"
   and one applied design question.

## Expansion first, then densify (binding, user's Oct 5 order)

The user rejected shortened lessons ("how can you reduce the content
in such a short manner"). The cut pass was over-applied. The order
is now:

1. EXPAND: cover EVERYTHING — every concept, angle, number, worked
   example, variant, edge case, and implication in the lecture. A
   lesson is done expanding when nothing interview-relevant from the
   lecture is missing. When in doubt, include.
2. DENSIFY: cut only zero-information sentences — throat-clearing,
   restatement, generic transitions, adjectives doing the work of
   facts. NEVER cut a concept, number, worked example, variant,
   failure mode, or edge case to save space.

Short is not dense. Dense means information per sentence at full
comprehensiveness. A 500-line lesson that misses concepts fails. A
1500-line lesson where every paragraph earns its place passes.

## Depth over breadth; density, no fluff (binding, user's Oct 5 order)

Interviews test depth of knowledge, not breadth. When a lesson must
choose, it goes deeper on the core mechanism instead of adding
another variant. Density rule: every paragraph carries a number, a
mechanism step, a failure mode, or a decision rule. Cut everything
else: no throat-clearing intros, no generic transitions, no
restated conclusions, no adjectives doing the work of facts. A
subchapter that can be removed without losing a number or a
mechanism is fluff and gets cut.

## Videos and go-deeper links (binding, kb-repo parity)

Every lesson page carries curated video and external links, as in
the llm-knowledge-base repo:
1. YOUTUBE EMBEDS via youtube-nocookie.com, 1 to 3 per lesson,
   each matched to the lesson's key subchapters (the lecture's own
   Stanford video where one exists, plus one strong external
   explainer). Verify each video ID exists before embedding.
2. VERIFIED "GO DEEPER" LINKS: papers, docs, and articles cited in
   the lesson, each link checked live. No dead links.

## Section rules
- ATOMIC sections: one idea per section, smallest complete unit.
- ORDER FOR LEARNING: simplest version first, then add complexity step by step.
- NO rigid template. Never force "what/why/how" into every section.
- NO redundancy: each fact stated ONCE at its natural place. Precedence: no redundancy.
- NO bloat. Short direct sentences. ASD-STE100. No em dashes.

## Per concept (only where the concept warrants it, never forced)
1. The problem it solves (one or two sentences).
2. The simplest toy version, with CONCRETE NUMBERS worked by hand.
3. Step-by-step buildup. A visual at EVERY step.
4. How it works under the hood (only the mechanism that matters).
5. The common misunderstanding (the trap interviewers set).
6. Interview Q&A inline as `> [!QA]` blocks (see below). No separate
   "interview relevance" section unless needed to avoid leaving gaps.

## Interview Q&A format
> [!QA]
> Q: <the question, as an interviewer would ask it>
> A: <the answer, 3-8 sentences, with a concrete number or example>
> Follow-up: <the harder follow-up and its one-line answer>

Add follow-ups. Cover general concepts too, not just lecture specifics.
Depth follows INTERVIEW IMPORTANCE, not lecture airtime.

## Visuals — binding spec: build/VISUAL_SYSTEM.md

Every figure on every page follows build/VISUAL_SYSTEM.md strictly.
No page ships without its figures. Summary of the binding rules:

- Draw only on state change (count, merge, score, mask, move, new symbol).
- Source order: official slide figure, official note figure, board/demo frame
  with timestamp, assigned paper figure, original only as last resort.
  Label the source on the figure (Stanford, paper, original).
- One atomic unit gets one lesson plate. One claim per plate.
- Medium ladder: table, equation, ASCII, Mermaid, SVG plate, Canvas 2D,
  three.js, Manim, Hyperframes. Use the first medium that passes the tests.
- Plate style: warm paper #F7F4EE, ink #1B2838, flat fills, no gradient,
  no glow, no shadow, no watermark, no clip art. Caption names source + shell.
- Cross-course symbols are reused, never redrawn (table in VISUAL_SYSTEM.md).
- Page audit before ship: every heading, equation, and architecture noun
  gets a unit id mapped to a figure id. Blank figure cell fails the page.

Figures that teach a process the reader can control (tokenizer lab,
attention explorer) are Canvas 2D with one control, computed live.

## Frontmatter
---
page_id, course_slug, course_name, course_order, order, nav, title, summary,
date, instructor, offering, duration,
video_id, video_title, video_caption,
concepts: [], sources: [{tag, label, url}]
---
Tags: video, slides, notes, paper, code, assignment, supplement, synthesis, inference.

## Writing rules (ASD-STE100)
Short direct sentences. Active voice. No em dashes. No contractions.
No semicolons. No banned filler. No generic intros. Start with substance.

## Hard rules
- Never hallucinate lecture content. [uncertain] where unsure.
- Timestamps: [mm:ss](ts:mm:ss), video_id must be real.
- Each fact once. If taught in an earlier lesson, link it, do not re-explain.
- ASD-STE100 + humanizer on every word, including READMEs and captions.
  Short sentences. Plain words. Zero AI tells. No contractions.
- No page ships until its audit table has no blank figure cell.
