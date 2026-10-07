# U13 answer key

All numeric claims from `../visuals/compute_u13.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

1M pages * 4.9 KB median ~ 4.9 GB raw, at 28.6% yield ~1.4 GB text.
(Toy medians, the lesson states the method, not the constant.)

## A1

(a) URL, timestamp, headers, raw bytes. CDX index. (b) 200k toy
records, 6.6 GB raw. (c) The CDX file: filter by URL before reading
WARC bodies.

## A2

(a) Kept bytes / raw bytes. (b) 28.6%, median 4.9 KB/page. (c)
Forum extractors that keep thread structure, readability-style
would drop the quotes that carry the discussion.

## A3

(a) PDF reading order, code file boundaries, book chapters. (b)
Column-aware sort recovers the order. (c) Fetch, extract with
layout analysis, split on section markers, keep provenance per
paper.

## A4

(a) (url, date, license) per document. (b) 700 trainable of 1000.
(c) Query the provenance index by URL, delete all copies, log the
takedown.

## A5

(a) Accuracy and the non-en->en rate, the latter guards the
majority class. (b) 0.952, 0.015. (c) Raise the English threshold
until the FP rate < 0.01 on a labeled sample, accept the recall
cost.

## A6

(a) Article pages whole, index pages split, books by chapter.
(b) 50 clean documents vs 1500 tokens of soup. (c) Conversation
boundaries, tool calls stay with their turn.

## A7

(a) Emails, phones, addresses, ids, tune for recall. (b) Precision
0.94, recall unknown: report honestly. (c) Stricter patterns plus
human review of the medical slice, accept over-redaction there.

## A8

(a) Sample with probability tgt/src per domain. (b) Multipliers
0.67/1.50/1.67/2.00/1.00. (c) Upweight math sources 2-3x, cap
repeats, watch the memorization tradeoff (U14-C12).

## A9

(a) Bytes/token measured per domain. (b) 4.20 and 3.60. 1.9 GB ->
~450M/~530M tokens. (c) Sample per domain, tokenize, sum
bytes/tokens: never assume 4.0.

## A10

(a) Truncated->drop/flag, bad encoding->re-decode/drop,
empty->drop. (b) 3.07/2.02/0.97%. (c) Check the fetcher for that
domain, likely a charset misdeclaration: fix or quarantine the
domain.

## A11

(a) Sort, seed, pin, hash proves it. (b) Stable: True, id
613575a7e17d78eb. (c) Not deterministic: replace with a sorted
structure or sort before hashing.

## A12

(a) Code, config, input, output hashes. (b) The toy manifest with
id 613575a7e17d78eb. (c) The manifest plus the pipeline code: the
regulator rebuilds the exact dataset.
