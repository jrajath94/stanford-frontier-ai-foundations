# U01 interview key

Minimum sufficient explanation, strong answer, red flags, rubric,
remediation per question.

## B1-B6 (breadth)

B1. Character: abstract unit ("é"). Code point: its number (U+00E9).
Byte: storage unit, UTF-8 maps the code point to [195, 169]. Red flag:
"characters are bytes".
B2. Bytes give a fixed 256-id base with no unknown hole and cover every
script, characters would need a huge id set for full Unicode. Red flag:
"bytes are shorter".
B3. Merge order decides which merges apply first at encode time, earlier
merges rewrite the text that later merges see. Same merges in different
order can give different segmentations.
B4. Pretokenization cuts text with a regex before merges run. Merges apply
per pretoken span, so no merge crosses a cut. Red flag: "the regex is
just for speed".
B5. I1 decode(encode(s)) == s, I2 encode is deterministic, I3 ids decode
to the exact consumed bytes.
B6. Mismatch classes: merges/vocab, regex/normalization, specials,
error handling. Checks: hash the merge list, compare regex source,
compare special maps, compare error policy. Red flag: naming only one
class.

## L1 (BPE mechanics)

L1.1 Pair count: sum over words of word-frequency times occurrences of
the adjacent pair.
L1.2 "ab" x3: pairs (a,b):3. "bc" x2: (b,c):2. First merge ("a","b")
count 3. After rewrite: "ab" x3, "bc" x2, pairs (ab,c):2, (b,c):2,
tie-break (count, then pair): ("b","c") vs ("ab","c"): "b" > "a", so
("b","c") count 2 wins second.
L1.3 Rank order replays training history: each merge assumes the
rewrites of earlier merges. Any other order applies merges to text that
training never produced.
L1.4 One pass: scan symbols left to right, join on match, O(n) per merge.
L1.5 WordPiece scores likelihood gain, BPE scores raw count. Debug: sort
by rank before applying. Critique: word boundaries are a design choice.
Experiment: train on code versus prose, compare top merges.

## L2 (production artifact)

L2.1 Config: (vocabulary, merge list, regex, special map, normalization,
error policy).
L2.2 "Testing 123": digit-keeping regex gives [" Testing", " 123"],
digit-splitting gives [" Testing", " 1", " 2", " 3"], ids shift.
L2.3 The model sees only ids, a shifted map is permuted input.
L2.4 Checker: hash merges, compare regex source and flags, compare
special maps, compare error policy, name the failing class.
L2.5 Bundled: versions with the model, registry: shared. Debug: compare
`pattern.pattern` plus flags, not object identity. Critique: convention
without enforcement drifts. Experiment: perplexity per mismatch class.

## E1

Pair counts: (t,h): 150, (h,e): 150, (e,</w>): 100, (e,n): 40,
(n,</w>): 40, rest <= 10. Tie-break (count, then pair): ("t","h") wins
with count 150. Fertility before: (100*4 + 40*5 + 10*6)/150 = 4.40.
After: (100*3 + 40*4 + 10*5)/150 = 3.40 tokens per word. Red flag:
forgetting the end-of-word token. Rubric: counts 2 pts, tie-break 1 pt,
both fertilities 2 pts.

## E2

BPE: 4096 / 1.3 = 3150 words (approx). Bytes: 4096 / 4.5 = 910 words
(approx). Implication: compare context in bytes or words, not tokens,
the byte model observes about 3.5x less text per window. Red flag:
"4096 tokens is 4096 tokens".

## D1

Bugs: (1) `feed` encodes empty strings from consecutive spaces
("a  b" splits to ["a", "", "b"]), and `encode_bpe("")` emits a phantom
end-of-word token, (2) `flush` encodes an empty remainder the same way.
Fix: skip empty words in both paths (`if w:`). Restored invariant:
emitted ids equal whole-string ids with no phantom tokens. Rubric: find
both 2 pts, fix 1 pt, invariant 1 pt. Remediation: re-read U01-C07.

## S1

Mitigations: (1) accept piece-level encoding (no code change, cost is
longer sequences and weaker representations for the new script),
(2) extend the vocabulary with new merges and fine-tune embeddings only
(cost is a small training run and a versioned tokenizer artifact).
Red flag: "retrain everything" as the only answer.

## S2

Identical ids: (1) compiled tokenizer (Rust) with the same config,
(2) precompiled regex cache plus batching. Safe id-changing option:
retrain merges with the same regex but a larger budget, then re-encode
all stored data and version the artifact. Tradeoff: engineering work
versus a data migration. Red flag: "faster regex" without the
identical-ids check.

## R1

Gaps: (1) no multilingual fertility numbers, close with tokens-per-word
across scripts at fixed V. (2) No downstream quality eval, close with a
fixed-model perplexity or task eval before/after. (3) No significance or
variance, close with multiple seeds and a stated comparison baseline.
Red flag: accepting the headline number.
