# Stanford Frontier AI — Foundations

Private study site. Built from official Stanford lectures, autumn 2026.

## What this is

Twelve courses. One site. Each course teaches from zero: no prior
machine learning needed. Every lesson has two levels. Level 1 gives you
what you need to follow everything. Level 2 gives you what you need to
get the details right.

## What each course contains

- **Lessons.** Full coverage of every lecture video. Interactive
  widgets, diagrams, and interview Q&A with complete answers.
- **Recap.** Each lesson ends with a one-screen recap: the key ideas
  with images, built for recall.
- **Cheatsheet.** One dense page per course. Definitions, formulas,
  numbers, decisions, common mistakes.
- **Crash course.** One fast page per course. The whole story in
  30 minutes, with images and links into the deep lessons.

## Use

Open `site/v2/index.html` in any browser. Chrome, Edge, Firefox,
and mobile browsers all work. The site is static: no server, no
build step, no tracking. Videos need internet; everything else
works offline.

## Courses in this repo

- **CS336:** Language Modeling from Scratch (Spring 2026)
- **CS229:** Machine Learning
- **CS229S:** Systems for Machine Learning

## Build

Content lives in `content/v2/` as Markdown. To rebuild:

```
python3 build/build.py content/v2 site/v2 "Stanford Frontier AI"
```

## Sources

Every lesson cites the official lecture video with timestamps, the
slides or lecture code, and the transcript. Nothing is invented.
Uncertain points carry an `[uncertain]` label.
