# U01 lab , tokenization from bytes to BPE

Environment: CPython 3, standard library only. No torch on this box, all
tasks run on CPU. Run: `python3 u01_lab_run.py`. Expected outputs are in
`u01_lab_key.md`, they were produced by executing this file on the build
box on 2026-10-06.

## Task 1 , UTF-8 encoder/decoder with invalid-input checks

Implement `encode_utf8(cp)` and `decode_utf8(byte_list)` per U01-C01 item
7. Then run: roundtrip for 20 code points across all four lengths,
invalid cases `[0x80]`, `[0xC1, 0x81]` (overlong "A"), and a truncated
3-byte lead `[0xE2, 0x82]`. Each invalid case must raise ValueError.

## Task 2 , three baseline tokenizers

Implement `build_vocab` for char, byte, and word levels per U01-C02.
Corpus: `["low lower", "newest low"]`. Report vocab sizes and the
encoding of `"lowest"`. Assert the word level maps `"lowest"` to UNK and
that char/byte roundtrip exactly.

## Task 3 , BPE training and determinism

Implement `train_bpe` and `encode_bpe` per U01-C03. Corpus counts:
`{"low": 2, "lower": 1, "lowest": 1, "new": 1, "newer": 1}`. Train 4
merges twice, assert both runs give the same merge list. Print the merge
list with counts and the encoding of `"lowest"`.

## Task 4 , pretokenizer coverage

Implement the C04 regex splitter. Assert `join(pretokens) == text` for
the fixed test string `"Hello, world! 123 Testing... don\u0027t"` and for 50
pseudo-random ASCII strings (seed 0). Print the split of the test string.

## Task 5 , special tokens

Extend the encoder with `encode_with_specials` per U01-C05 using the
fence `"<|endoftext|>"` mapped to id 50000. Assert the fence appears as
exactly one id 50000 and that decode roundtrips. Also assert that without
registration the fence string encodes to more than one id.

## Task 6 , roundtrip invariants

Implement `check_invariants` per U01-C06. Run it on 200 seeded random
strings with the plain encoder (expect zero failures), then with a
lowercasing normalizer (expect nonzero failures). Report both counts.

## Task 7 , streaming encoder

Implement `StreamingEncoder` per U01-C07 with the C03 merges. For 100
seeded random splits of the test paragraph, assert streamed ids equal
whole-string ids. Also show that naive per-chunk encoding mismatches on
at least one split.

## Task 8 , fertility curve

Implement `fertility_curve` per U01-C08 on the repeated-sentence corpus
for merge budgets 0..8. Assert non-increasing fertility. Print the curve.

## Task 9 , mismatch checker

Implement `check_tokenizer_match` per U01-C09. Positive case passes.
Then mutate each of the four classes once and assert the checker names
the mutated class.

## Task 10 , script stats and digit regex

Implement `script_stats` per U01-C10 on a mixed sample string. Assert
byte-level token count equals UTF-8 byte count. Compare token counts for
`"2026"` under a digit-splitting regex and a digit-keeping regex.

## Task 11 , byte-level baseline

Implement `encode_bytes` per U01-C11. Report the length multiplier versus
the 4-merge BPE on the test paragraph and the implied attention FLOP
ratio (multiplier squared). Assert strict decode raises on `[0xFF]`.

## Task 12 , profiler

Implement `profile_encode` per U01-C12. Profile the regex stage and the
merge stage separately on 10k words. Report MB/s and the stage shares.
Assert the parts sum to within 10 percent of the whole.
