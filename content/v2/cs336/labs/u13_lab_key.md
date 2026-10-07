# U13 lab key

Executed 2026-10-06, numpy CPU. Run `python3 u13_lab_run.py`, outputs
must match exactly.

```
T1 record fields: url, timestamp, headers, body. CDX -> offsets
T2 yield 28.6%, median kept 4.9 KB/page
T3 column-aware order: [(0, 0), (0, 1), (1, 0), (1, 1)]
T4 trainable: 700/1000 permissive, takedown = index delete + log
T5 acc 0.952, non-en->en FP 0.015
T6 index page 50x30tok: split -> 50 docs, article: keep whole
T7 classes: email/phone/address/id, posture: recall-first
T8 multipliers: 0.67/1.50/1.67/2.00/1.00
T9 4.20 and 3.60 B/tok. 1.9 GB -> ~450M/~530M tokens
T10 truncated 3.07%, bad-encoding 2.02%, empty 0.97%
T11 stable across reversal: True (613575a7e17d78eb)
T12 manifest: code+config+inputs+output hashes, output 613575a7e17d78eb
```
