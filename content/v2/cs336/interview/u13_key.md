# U13 interview key

## B1-B6

B1. URL, timestamp, headers, body. CDX maps URLs to offsets.
B2. Kept/raw bytes, toy 28.6%.
B3. PDF reading order, code file boundaries, book chapters.
B4. URL, crawl date, license tag.
B5. It guards the majority class against contamination.
B6. Sort, seed RNGs, pin dependencies, hash to prove.

## L1

L1.1 Kept bytes / raw bytes.
L1.2 1.9/6.6 GB = 28.6%.
L1.3 Text-dense blocks are article, tag-dense are chrome.
L1.4 `extract`, no script/style text survives.
L1.5 Readability for articles, classifiers when labeled data
exists, debug: JS pages, critique: toy distribution, experiment:
two extractors.

## L2

L2.1 Code, config, input, output hashes.
L2.2 Id 613575a7e17d78eb, reversal gives the same.
L2.3 Order-independence of the digest.
L2.4 `version`, reversed input matches.
L2.5 Timestamps are not reproducible, debug: nondeterministic
library, critique: toy scale, experiment: two processes.

## E1

(a) 10e12/4.2 = 2.38e12 tokens. (b) 10/0.286 = 35 TB raw. (c)
Code tokens: 0.15*35e12/2.5 = 2.1e12, recompute the total with
per-domain rates. Rubric: (a) 1 pt, (b) 1 pt, (c) 2 pts. Red
flag: using 4.2 for code.

## E2

(a) Query provenance index, delete all copies incl. dedup
survivors, log. (b) No: dedup means fewer copies, same steps.
(c) Re-query returns zero, the log shows the deletion.

## D1

Bug: iteration order over `docs` is the input order (and set
order is nondeterministic). Fix: sort by document id before
hashing. Rubric: find 2 pts, fix 2 pts.

## S1

Diagnose: charset misdeclaration or a new source. Policy:
quarantine the domain, fix the decoder, re-run, do not silently
keep mojibake.

## S2

Pin: OS image, library versions, tokenizer version, RNG
algorithm, thread count (order effects), locale/encoding. Then
verify the id matches.

## R1

Gaps: (1) no tokenizer: token counts meaningless, name it. (2)
no dedup: repetition unknown, describe the method. (3) no
licenses: usability unknown, break down the mix.
