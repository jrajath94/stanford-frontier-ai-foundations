# U01 answer key , lesson assessments

Closed-book first. Each answer gives the minimum sufficient explanation,
the strong answer, common red flags, and a rubric.

## R1 (remediation)

Encode "€" (U+20AC) by hand: U+20AC is in the 3-byte range. Bits:
0010 0000 1010 1100. Byte 1: 1110_0010 = 0xE2 = 226. Byte 2:
10_000010 = 0x82 = 130. Byte 3: 10_101100 = 0xAC = 172. Result
[226, 130, 172], matching `compute_u01.py`.
Red flag: writing 2 bytes (wrong range). Rubric: correct lead byte 1 pt,
correct continuation bytes 1 pt.

## A1

(a) Lead patterns: `0xxxxxxx`, `110xxxxx`, `1110xxxx`, `11110xxx`.
(b) Ladder. Code point: the number for a character. "ñ" U+00F1: 2-byte
range, bits 11110001 -> [0xC3, 0xB1] = [195, 177]. The lead byte encodes
length because its high bits are a prefix code no continuation byte can
match (`10xxxxxx`). 2-byte decode: read lead, length = 2, value =
((b0 & 0x1F) << 6) | (b1 & 0x3F). UTF-32 comparison: fixed 4 bytes, simple
indexing, 4x memory on ASCII. Debug: reject lead bytes 0xC0, 0xC1 and
check the decoded value meets the range minimum. Critique: real input
includes invalid sequences, the decoder needs an error policy. Transfer:
the receiver sees a lead byte without its continuations, a length check
per sequence detects it, and the first broken sequence is where decoding
stops.

## A2

(a) Char: distinct characters (about 100 for ASCII text). Byte: 256. Word:
distinct words plus UNK (unbounded).
(b) Ladder. Byte level needs no UNK because 256 ids cover every possible
byte. "hi!" char: [h, i, !] as 3 ids, byte: [104, 105, 33], word:
["hi!"] or [UNK] if unseen. Sequence lengths: byte >= char >= word
typically. Debug: keep punctuation as its own token instead of dropping
it. Critique: closed vocabulary fails on open text (code, names).
Transfer: emoji are multi-byte, byte level handles them with no new ids,
so byte degrades least.
(c) Red flag: "word tokenizers have no UNK problem".

## A3

(a) Loop: count pairs, pick max, record merge, rewrite words, repeat.
(b) Ladder. Pair count: sum of word frequencies containing the adjacent
pair. First merge of the toy: ("o","w") count 4. Greedy order matters
because a later merge can only match after earlier rewrites. One rewrite:
scan left to right, join on match, skip two. WordPiece: likelihood gain,
not raw count. Debug: sort merges by rank before applying, applying in
discovery order after a re-sort bug changes ids. Critique: word
boundaries are a choice, some designs merge across them. Transfer: the
new word encodes with existing merges only (possibly as pieces), the
fix is a retrain or accepting piece-level encoding.

## A4

(a) Merges run inside pretokens because encoding applies merges per
pretoken span, no span crosses a regex cut.
(b) Ladder. Pretoken: one regex match. "cannot stop": ["can", "'t",
" stop"]. Order matters: contraction alternatives match before letter
runs. Coverage check: `"".join(findall(pat, s)) == s`. No-pretok BPE:
allows cross-word merges, less control. Debug: a pattern that drops
spaces fails the coverage check, fix by adding a whitespace alternative.
Critique: digit splitting harms numbers. Transfer: snake_case is one
letter run, so it stays whole and BPE can merge across the underscores
only if underscores are letters in the pattern, otherwise each part is
its own pretoken.

## A5

(a) Specials bypass BPE because they are control signals, not text, a
merge inside one would destroy its meaning.
(b) Ladder. Special token: reserved id with control meaning. Toy:
[ids("hi"), 50000]. Longest match first so nested fences resolve to the
outermost. Splitter: regex alternation of escaped specials. No-special
encoding: the fence string becomes ordinary pieces. Debug: sort specials
by length descending before building the alternation. Critique: content
can contain fence strings (prompt injection, help text). Transfer: the
bug removes document boundaries, so the loss is computed across
documents and the model learns cross-document continuation. That is a
training/inference mismatch (C09).

## A6

(a) I1: decode(encode(s)) == s. I2: encode is deterministic. I3: ids
decode to the exact consumed bytes.
(b) Ladder. Roundtrip: byte-exact return. "naïve": encode then decode
returns "naïve". Stage injectivity implies I1 because the composition of
injective maps is injective. Checker: loop with three asserts. Full
versus weak: weak allows normalization, fine for search, wrong for
training data. Debug: strip the trailing space in decode or in the test
corpus, then re-run. Critique: both sides must share config. Transfer:
encode a string with an accented character under each tokenizer, the one
that fails I1 differs.

## A7

(a) Hold back the last L_max - 1 characters, emit the rest, flush at end.
(b) Ladder. Boundary problem: merges cannot cross chunk edges, so ids
differ from whole-string encoding. "unhappy" whole versus "unha"+"ppy":
the "a|p" merge is lost in the split. The bound holds because no token
is longer than L_max. feed/flush: as item 7 of the lesson. Re-encode:
O(n^2), correct but slow. Debug: flush must encode the carry, not drop
it. Critique: regex context can need more than characters. Transfer: the
safe boundary is the last position where the regex cannot extend a
match, hold back to there.

## A8

(a) Larger V compresses more but costs V*d parameters and leaves rare
tokens undertrained.
(b) Ladder. Fertility: tokens per word. Toy: 3.89 char-level, about 1.30
small-BPE. Diminishing returns: frequent words become single tokens
early, later merges serve rare words. Curve: train per budget, count.
32k versus 256k for code: 256k wastes capacity on rare tokens, 32k with
code-heavy training is the sane default. Debug: exclude EOT from the
token count. Critique: the corpus must match deployment. Transfer: T
doubles, so attention cost rises, V fixed, so embedding cost does not.

## A9

(a) Four classes: vocab/merges, regex/normalization, specials,
byte-fallback/error handling.
(b) Ladder. Config tuple: (vocab, merges, regex, specials,
normalization). Hand-show: the C04 digit regex changes segmentation.
Ids are the input alphabet because the model only sees ids. Checker:
hash each component. Bundled versus registry: bundled versions with the
model, registry shares across models. Debug: compare `pattern.pattern`
and flags, not just the compiled object. Critique: versioning must be
enforced, not assumed. Transfer: lowercasing is a normalization
mismatch, casing information is destroyed and ids shift.

## A10

(a) Scripts with longer UTF-8 forms start further from good compression
at fixed V.
(b) Ladder. Byte-length effect: 3-byte characters need more merges to
become one token. "中文" at byte level: 6 tokens. Upsampling: give rare
scripts more pair counts during training. script_stats: group by UTF-8
length. Digit-split: one token per digit, place value hidden, digit-keep:
runs stay together. Debug: train on multilingual data, not English-only.
Critique: corpus mix must match deployment. Transfer: the digit-keeping
regex keeps the number short, the agent sees one token per number part,
which matters for exact copying.

## A11

(a) V = 256 removes the vocabulary entirely, sequences get 3-5x longer
and attention work grows with the square.
(b) Ladder. Encoding: list of UTF-8 bytes. Toy multiplier 3.8x on the
sample sentence. m^2: attention is O(T^2), T scales by m. encode_bytes:
one line. Multilingual comparison: bytes have no script bias. Debug: use
errors="strict" so invalid bytes raise instead of silently becoming the
replacement character. Critique: the architecture must absorb the longer
sequences. Transfer: 4096 byte tokens hold about 1000 English words at 4 bytes
per word, versus more under BPE, compare in bytes, not tokens.

## A12

(a) Regex time, merge time, Python overhead.
(b) Ladder. Throughput: MB/s or tokens/s. 1 MB at 50 MB/s: 0.02 s.
Median resists outliers from GC pauses and scheduling. Profiler: warmup
then timed repeats. Python versus compiled: 10-100x gap is normal.
Debug: time only the encode call, not file IO. Critique: single-thread
numbers do not predict a 8-worker loader. Transfer: the regex dominates,
so the first fix is a faster pretokenizer (compiled tokenizer or simpler
pattern), not merge optimization.
