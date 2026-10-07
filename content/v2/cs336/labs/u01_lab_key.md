# U01 lab key , execution-verified outputs

Produced by running `python3 u01_lab_run.py` on the build box (CPython,
2026-10-06). Learner code must reproduce these values exactly, except
T12 timings are machine-dependent, valid within 3x on a comparable machine, ratios are the stable claim.

```
T1 roundtrips: 20/20
T1 raise ok for [128]
T1 raise ok for [193, 129]
T1 raise ok for [226, 130]
T2 char vocab=9 enc(lowest)=[2, 4, 8, 1, 6, 7] roundtrip=True
T2 byte vocab=9 enc(lowest)=[1, 3, 7, 0, 5, 6] roundtrip=True
T2 word vocab=4 enc(lowest)=[3] roundtrip=True
T3 deterministic: True
T3 merge (o,w) count 4
T3 merge (l,ow) count 4
T3 merge (r,</w>) count 2
T3 merge (n,e) count 2
T3 encode lowest: ['low', 'e', 's', 't', '</w>']
T4 split: ['Hello', ',', ' world', '!', ' 123', ' Testing', '...', ' don', "'t"]
T4 coverage: 50/50
T5 with fence: [104, 105, 50000, 98, 121, 101] single EOT: True
T5 without fence: 18 ids, no 50000: True
T6 plain failures: 0/200, lowercased failures: 163/200
T7 streamed==whole: 100/100, naive mismatches: 100/100
T8 fertility: ['4.89', '4.67', '4.44', '4.22', '4.11', '4.00', '3.89', '3.78', '3.67']
T8 non-increasing: True
T9 positive: True
T9 catches vocab/merges: True
T9 catches regex/normalization: True
T9 catches specials: True
T9 catches byte-fallback: True
T10 byte-length histogram: {1: 8, 2: 2, 3: 2, 4: 1}
T10 byte-level tokens == utf8 bytes: True
T10 '2026' split-digits: ['2', '0', '2', '6'] keep-digits: ['2026']
T11 byte tokens 43 vs BPE 33, multiplier 1.30, attn FLOP ratio 1.7
T11 strict decode raises: True
T12 full( regex-stage) 0.003 s -> 16.1 MB/s, byte-scan 0.001 s
T12 attribution: regex-stage dominates byte-scan by 4.9x
```

Notes:

- T2 ids depend on sorted-unit ordering, the key property is vocab sizes
  (9, 9, 4), the UNK mapping at word level, and exact roundtrips at char
  and byte levels. Any consistent id assignment passes.
- T3 merge order depends on the documented tie-break (max by count, then
  by pair). A different documented tie-break gives a different but valid
  merge list, determinism across runs is the graded property.
- T6: 163 of 200 lowercased strings fail the roundtrip because the sample
  alphabet contains uppercase and accented characters.
- T12 timings are machine-specific, the graded claims are the attribution
  ratio (regex stage several times the byte scan) and the parts-to-whole
  agreement within 10 percent.
