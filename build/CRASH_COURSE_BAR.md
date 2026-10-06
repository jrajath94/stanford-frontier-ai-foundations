# Crash-course potency bar (binding, user's Oct 5 order)

The user: current crash courses are "shitty" — too short, not potent.
The bar: **reading the crash course alone must make him interview-ready
for that course's domain** — equivalent understanding to reading every
lesson in extreme detail. Precedent: the AIP crash course (17,114 →
~39,061 words, 25/25 exam-clearing test answerable from the crash
course alone).

## 1. Completeness: lossless compression

Every interview-relevant concept from the course's lessons must appear:
every mechanism, every number, every failure mode, every "used where"
mapping. Build a **coverage map**: list each lesson's key concepts and
map each into a crash chapter. Anything unmapped is a gap. No gap ships.

## 2. Structure: multi-chapter

One page per course is not required. Split into chapters as needed:
`crash-course.md`, `crash-course-02.md`, `crash-course-03.md`, …
(link chapters to each other explicitly; build.py auto-builds every
.md file). Each chapter follows the textbook arc: concrete problem →
mechanism built with numbers → failure modes → used-where (Oct 2026) →
memory aids → self-test.

## 3. Memory aids (mandatory, not decorative)

- Mnemonics for lists and sequences.
- Never-confuse pairs (X vs Y, and the one-line decider).
- If-this-then-that rules (if the interviewer asks X, reach for Y).
- Trap cards: "interviewers love to ask…" with the worked answer.
- One-glance tables: numbers, decisions, comparisons.

## 4. Figures

A teaching figure for every major mechanism. Reuse the lesson plates
where they fit; create new summary/comparison figures where the crash
course needs its own. Follow build/VISUAL_SYSTEM.md. No decorative
figures; every figure teaches a claim.

## 5. Exam-clearing test (the potency proof)

For each course, write 25+ interview questions spanning the course.
Then answer each using ONLY the crash course text, as a reader would.
Score = answerable / total. Fix gaps until the score is 100%.
Report the score and the questions. This test is what makes "potent"
a measured claim instead of an adjective.

## 6. Density

build/FORMAT.md density rule applies: every paragraph carries a
number, a mechanism step, a failure mode, or a decision rule. No fluff.
Length has no cap — drastic expansion is expected, but every word
must earn its place.

## 7. QA

Same gates as lessons: build green, live overflow sweep at
375/768/1200px on every crash page, zero broken images, STE100 clean,
youtube-nocookie embeds where they help, go-deeper links live.
