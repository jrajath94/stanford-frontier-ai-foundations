# U13 interview bank , questions

Closed-book. Keys in `u13_key.md`. All numbers are synthetic toys
from `visuals/compute_u13.py`.
Quotas: 6 breadth, 2 deep ladders of 5, 2 analytical, 1
implementation/debug, 2 changed-constraint, 1 research-critique.

## Breadth (6)

B1. Name the WARC record fields and the CDX role.
B2. What is extraction yield, and what was the toy value?
B3. Name the three non-HTML format hazards.
B4. What three fields make a provenance record?
B5. Why is the non-English-called-English rate the key metric?
B6. State the determinism recipe for a data pipeline.

## Deep ladders (2 x 5)

L1. Extraction.
- L1.1 Define yield.
- L1.2 Toy: 6.6 GB raw -> 1.9 GB kept, 28.6%.
- L1.3 Justify block scoring by text-to-tag ratio.
- L1.4 Sketch `extract`, state the no-script-text check.
- L1.5 Compare readability-style with classifier extractors,
  debug the JS-rendered case, critique the toy distribution,
  propose the two-extractor experiment.

L2. Versioning.
- L2.1 Define the manifest's four hashes.
- L2.2 Toy: id 613575a7e17d78eb, stable across reversal.
- L2.3 Justify sorting before hashing.
- L2.4 Implement `version`, state the reversal check.
- L2.5 Compare with timestamped snapshots, debug the
  nondeterministic-library case, critique the toy scale, propose
  the two-process test.

## Analytical exercises (2)

E1. 10 TB of mixed web data at 4.2 bytes/token. (a) How many
tokens? (b) At the toy yield of 28.6%, how much raw crawl is
needed? (c) The code domain is 15% of raw but tokenizes at 2.5
bytes/token. Recompute the token count.
E2. A takedown request lists 50 URLs. (a) List the removal steps.
(b) Dedup kept one copy of each, does that change the steps?
(c) What proves completion?

## Implementation/debug task (1)

D1. This version function is nondeterministic. Find the bug.

```
def version(docs):
    h = hashlib.sha256()
    for d in docs:  # input order not sorted
        h.update(d.encode())
    return h.hexdigest()
```

## Changed-constraint scenarios (2)

S1. Bad-encoding rate jumps to 15% on one domain. Diagnose and
set the policy.
S2. The pipeline must reproduce bit-identical output on a second
machine. List the additional pins beyond sorting and seeding.

## Research critique (1)

R1. A corpus paper reports size in tokens with no tokenizer
named, no dedup described, and no license breakdown. List three
gaps and the disclosure for each.
