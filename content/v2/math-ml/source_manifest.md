# source_manifest.md, math-ml source register

Baseline: 2026-10-06. Status fields follow the v2 prompt register schema.

## SRC-01, supplied short link (seed)

- Artifact ID: SRC-01
- Canonical link: https://lnkd.in/gQe4kMkf
- Publisher: unknown (short link. Destination not resolved)
- Title: not verified
- Edition/date: not verified
- Access status: destination unresolved. No inspection possible
- Inspection extent: none
- Anchors: none
- Extracted concepts: none
- Lesson mapping: none
- Remaining gaps: resolve destination or mark permanently unreachable

## SRC-02, official course preview page

- Artifact ID: SRC-02
- Canonical link: https://onlinecourses.nptel.ac.in/noc26_cs02/preview
- Publisher: NPTEL / Swayam
- Title: Mathematical Foundations of Machine Learning (noc26_cs02)
- Edition/date: Jan-Apr 2026 offering listed
- Access status: public page. Renders in JavaScript only
- Inspection extent: fetch on 2026-10-06 returned a loading shell only. No overview text extracted
- Anchors: none extracted
- Extracted concepts: none extracted by me. The inventory notes prior
  inspection of this overview
- Lesson mapping: none yet
- Remaining gaps: overview text, syllabus, schedule, assignment set all
  unextracted

## SRC-03, instructor teaching page

- Artifact ID: SRC-03
- Canonical link: https://prathosh.in/teaching.html
- Publisher: Dr. Prathosh A P (personal site)
- Title: Teaching & Service
- Edition/date: fetched 2026-10-06. Page states the math-foundations MOOC
  is a 12-week course on linear algebra, probability, statistical inference,
  optimisation, and connections to modern deep learning. Second offering
  (August 2026) currently ongoing
- Access status: public, full text extracted
- Inspection extent: full page text, 101 lines. No lecture detail beyond
  the course paragraph and the YouTube playlist embed
- Anchors: course paragraph (linear algebra through deep learning),
  playlist embed list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu
- Extracted concepts: course identity, instructor identity, scope topics,
  playlist canonical link
- Lesson mapping: none yet (identity only)
- Remaining gaps: none at identity level

## SRC-04, official YouTube lecture playlist

- Artifact ID: SRC-04
- Canonical link: https://www.youtube.com/playlist?list=PLgMDNELGJ1Cay-Q9Cn8KcpUcC58NDWuiu
- Publisher: NPTEL, Indian Institute of Science, Bengaluru (channel)
- Title: Mathematical Foundations of Machine Learning
- Edition/date: playlist as listed on 2026-10-06
- Access status: public. Titles enumerated directly on 2026-10-06
- Inspection extent: 89 titles extracted (1 intro, 69 lectures Lec 01-Lec 69,
  19 tutorial videos). Transcripts not inspected. Durations not extracted.
  Thumbnails not reviewed.
- Anchors: lecture titles listed in source_gaps.md attachment A
- Extracted concepts: lecture-level topic map only. No spoken content,
  no slides, no worked examples extracted
- Lesson mapping: title-level map to U01-U10 in course_map.md
- Remaining gaps: transcripts, slides, demos, per-video durations,
  tutorial content, assignment links

## SRC-05, NPTEL course catalog entry

- Artifact ID: SRC-05
- Canonical link: https://NPTEL.ac.in/fdp (course table)
- Publisher: NPTEL
- Title: Mathematical Foundations of Machine Learning, noc26-cs02
- Edition/date: 12 weeks, Jan 19, 2026 to Apr 10, 2026, exam Apr 25, 2026
- Access status: public table text, seen via search on 2026-10-06
- Inspection extent: single catalog row only
- Anchors: SME name Prof. Prathosh A P, coordinating institute IISc Bangalore
- Extracted concepts: identity corroboration only
- Lesson mapping: none
- Remaining gaps: none at identity level

## SRC-06, Class Central course listing

- Artifact ID: SRC-06
- Canonical link: https://www.classcentral.com/course/swayam-mathematical-foundations-of-machine-learning-503024
- Publisher: Class Central (secondary source)
- Title: Mathematical Foundations of Machine Learning (Swayam)
- Edition/date: listing crawled 2026-10-06
- Access status: public, text seen via search
- Inspection extent: 12-week topic summary only (weeks 1-12 headings)
- Anchors: week headings 1 through 12
- Extracted concepts: week-level syllabus corroboration:
  ERM, Bayes optimality, divergence minimization, MLE/MAP, nonparametric
  estimates, linear models, regularization, kernel machines/SVMs,
  perceptron/NNs/backprop, CNNs, RNNs/LSTMs, attention/transformers,
  trees/ensembles, k-means/GMM/EM/PCA, GAN/VAE/diffusion preview
- Lesson mapping: corroborates course_map.md week structure only. Treated as secondary evidence, not as instructor text
- Remaining gaps: this is a listing summary, not the syllabus itself

## SRC-07, third-party study-notes repository

- Artifact ID: SRC-07
- Canonical link: https://github.com/sivanagaraju/mathematical-foundations-of-ml
- Publisher: sivanagaraju (third party. Not the instructor)
- Title: Mathematical Foundations, Study Notes
- Edition/date: README updated ~12 days before 2026-10-06
- Access status: public
- Inspection extent: NOTES.md and README.md text seen via search only. No per-lecture folders inspected
- Anchors: claims "89 videos" and "~46.6 hours" for the same playlist
- Extracted concepts: none (my playlist enumeration independently gives
  89 titles. The 46.6-hour figure remains unverified)
- Lesson mapping: none
- Remaining gaps: third-party interpretations are not source evidence

## SRC-08, firsthand learner report (LinkedIn)

- Artifact ID: SRC-08
- Canonical link: https://www.linkedin.com/pulse/machine-learning-function-approximation-mathematical-intuition-ishu-pymif
- Publisher: LinkedIn (individual learner, firsthand report, date not recorded)
- Title: Machine Learning as Function Approximation: A Mathematical Intuition
- Edition/date: not recorded
- Access status: public text, seen via search on 2026-10-06
- Inspection extent: article text about the first lecture framing
- Anchors: reports that Lec 01 introduces ML as function approximation
  (unknown f, data as input/output examples, models as approximations)
- Extracted concepts: corroborates Lec 01 title "Overview of Function
  Approximation". Interview provenance class: firsthand public report
- Lesson mapping: none yet
- Remaining gaps: publication date unknown. One learner only

## SRC-09, third-party learner notes and assignment repo

- Artifact ID: SRC-09
- Canonical link: https://github.com/priyo13o4/nptel-sem6 (MathematicalFoundations subfolder)
- Publisher: priyo13o4 (third party. Learner, not the instructor)
- Title: NPTEL-Sem6 / MathematicalFoundations, weekly notes and assignments
- Edition/date: repo updated ~165 days before 2026-10-06 (per search crawl). Weeks 1-12 folders plus numerical-only summaries
- Access status: public
- Inspection extent: repo listing fetched 2026-10-06 (12 week folders confirmed). Week assignment files seen via search snippets only. No file contents copied into this build
- Anchors: Week 1 objectives (function approximation, data representation as (x, y) pairs). Week 2 objectives (vector spaces and subspaces, linear independence, basis and dimension, matrix operations, systems of linear equations, eigenvalues and eigenvectors). Week 3 (entropy). Week 4 (KL divergence). Week 5 (Bayesian estimation, MAP, Beta-Bernoulli)
- Extracted concepts: assignment objectives (topic lists) only. Week 2 corroborates the U02 leaf scope. Not instructor evidence
- Lesson mapping: none. Corroboration only
- Remaining gaps: learner-authored interpretations are not source evidence for leaf attribution. The second offering's assignment set may differ

## Classification summary (2026-10-06, updated RUN 2)

- Found: SRC-01 through SRC-09.
- Inspected full (text): SRC-03, SRC-04 (titles only), SRC-05 (row only),
  SRC-06 (summary only).
- Inspected partial: SRC-02 (loading shell only), SRC-07, SRC-08, SRC-09
  (listing + snippets only).
- Extracted: identity, instructor, scope topics, 89 titles, week headings,
  third-party assignment objectives.
- Mapped: title-level only.
- Taught: none from source (RUN 1 and RUN 2 teach authored prerequisite
  bridge and source-block lessons with original toys. Source attribution
  PENDING).
- Assessed: none from source.
- Mastered: none claimed.
