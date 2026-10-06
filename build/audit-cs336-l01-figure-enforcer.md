# Figure Enforcer Audit — cs336 l01: Tokenization

Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent). Content gates G1–G9 PASS — prose untouched except figure captions and one broken image ref.

## Page audit: unit -> figure (zero blank cells)

| # | Unit (heading / equation / code / arch noun) | Unit id | Figure id | Medium | Status |
|---|---|---|---|---|---|
| 1 | The pipeline: five stages, two cross-cutters | u-pipeline | f-pipeline | SVG (new) | REPLACED (dark webp, icons) |
| 2 | Chain rule factorization p(x1..xn) | u-chainrule | f-pipeline | SVG | kept (captioned in pipeline fig) |
| 3 | Char-level tokenization (worked ASCII) | u-char | f-chap-granularity | SVG chapter plate | ADDED |
| 4 | Word-level tokenization | u-word | f-example-tokens | PNG (web source) | kept, caption fixed |
| 5 | The key question: what is a token | u-question | f-chap-granularity | SVG chapter plate | ADDED |
| 6 | The subword idea | u-subword-idea | f-subword-family | SVG (new) | REPLACED (typo webp) |
| 7 | BPE training by hand (256 bytes, merge loop) | u-bpe-train | f-bpe-corpus | webp | kept, caption fixed |
| 8 | BPE training mermaid flowchart | u-bpe-train | f-mer-bpe-train | mermaid (inline) | kept |
| 9 | BPE merge animation | u-bpe-train | f-anim-bpe | video (mp4) | kept |
| 10 | BPE merges hierarchy | u-bpe-train | f-bpe-merges | webp | kept, caption fixed |
| 11 | BPE from Hugging Face (segmentation) | u-bpe-train | f-hf-bpe | SVG (web source) | kept, caption fixed |
| 12 | BPE inference, greedy longest match | u-bpe-infer | f-text-to-ids | SVG (new) | ADDED |
| 13 | Text to tokens to IDs path | u-encode-path | f-text-to-ids | SVG (new) | ADDED |
| 14 | Pretokenizer splits before BPE | u-pretok | f-chap-bpe | SVG chapter plate | ADDED |
| 15 | GPT-2 regex, piece by piece | u-gpt2-regex | f-chap-bpe | SVG chapter plate | ADDED |
| 16 | Special tokens are atomic | u-special | f-text-to-ids | SVG | kept (noted in caption) |
| 17 | WordPiece grows by likelihood lift | u-wordpiece | f-subword-family | SVG (new) | REPLACED |
| 18 | Unigram prunes from a large start | u-unigram | f-subword-family | SVG (new) | REPLACED |
| 19 | Same corpus, three vocabularies | u-same-corpus | f-subword-family | SVG (new) | REPLACED |
| 20 | SentencePiece is a wrapper | u-sentpiece | f-subword-family | SVG | kept |
| 21 | Byte-level initialization | u-bytes | f-chap-bpe | SVG chapter plate | ADDED |
| 22 | Fertility measured and priced | u-fertility | f-fertility-cost | webp | kept, caption fixed |
| 23 | Fertility across languages | u-fertility-lang | f-fertility-table | markdown table | REPLACED (invented-value webp) |
| 24 | Numbers tokenized badly | u-numbers | f-fertility-table | markdown table | kept |
| 25 | Vocabulary tax curve | u-vocab-tax | f-chap-fertility | SVG chapter plate | ADDED |
| 26 | Tokenizer choice at serving scale | u-serving | f-chap-fertility | SVG chapter plate | ADDED |
| 27 | Six-tokenizer fertility measurements | u-six-tok | t-census (inline table) | markdown table | kept |
| 28 | Production tokenizer census | u-production | t-census (inline table) | markdown table | REPLACED (image-table webp) |
| 29 | Perplexity: the score the pipeline optimizes | u-perplexity | f-chap-perplexity | SVG chapter plate | ADDED |
| 30 | Bits per byte, the fair comparison | u-bitsbyte | f-chap-perplexity | SVG chapter plate | ADDED |
| 31 | Glitch tokens | u-glitch | f-chap-perplexity | SVG chapter plate | kept |
| 32 | Token healing at boundaries | u-healing | f-text-to-ids | SVG | kept |
| 33 | Training-serving skew | u-skew | f-chap-fertility | SVG chapter plate | kept |
| 34 | Tokenizer training at scale | u-scale | f-bpe-corpus | webp | kept |
| 35 | Freeze it, then never touch it | u-freeze | f-embedding-lookup | SVG (new) | REPLACED (white-bg webp) |
| 36 | Embedding interface: IDs index rows | u-embed | f-embedding-lookup | SVG (new) | REPLACED |
| 37 | Embedding matrix cost (1.24 GB) | u-embed-cost | f-embedding-lookup | SVG (new) | REPLACED |
| 38 | Batching: padding, masks | u-batch | none (ladder: prose suffices) | — | noted |
| 39 | tiktoken production implementation | u-tiktoken | f-text-to-ids | SVG | kept |
| 40 | Recap cards 1-8 | u-recap | f-pipeline, f-text-to-ids, f-example-tokens, f-bpe-merges, f-bpe-corpus, f-hf-bpe, f-fertility-cost, f-embedding-lookup | mixed | FIXED (4 card refs) |

## Figures kept / fixed / added, by medium

- SVG lesson plates (kept, caption fixed): none pre-existing in l01.
- SVG lesson plates (new): l01-pipeline.svg, l01-text-to-ids.svg, l01-subword-family.svg, l01-embedding-lookup.svg.
- SVG chapter plates (new): l01-chap-granularity.svg, l01-chap-bpe.svg, l01-chap-fertility.svg, l01-chap-perplexity.svg.
- webp kept (caption fixed): media-generation-bpe-training-corpus-0-17dca458-28c4-4e91-9d91-ce048d890047.webp, media-generation-bpe-merges-0-ddd1dc1d-fd76-46c8-8490-81be363d2dab.webp, media-generation-cs336-l01-fertility-cost-0-33eeb420-d8ec-487b-8b65-750e7ae17764.webp.
- webp replaced/deleted: lm-pipeline-stages (dark bg, icons, glow -> l01-pipeline.svg), tokenizer-pipeline (white bg, invented IDs -> mermaid removed; recap uses l01-text-to-ids.svg), subword-family (typo "wsing" -> l01-subword-family.svg; file deleted), production-tokenizer (image-table -> inline census table; file deleted), fertility-languages (invented per-language values -> prose-computed markdown table; file kept on disk because crash-course.md still links it), embedding-lookup (white bg, invented vectors -> l01-embedding-lookup.svg; file kept on disk because crash-course.md still links it).
- Tables: fertility-by-language markdown table added; census table kept as figure.
- mermaid: BPE training flowchart kept (6 nodes, compliant). Tokenizer-pipeline flow claim now carried by l01-text-to-ids.svg.
- ASCII: char/word worked examples kept.

## Mechanical fixes

- Recap card 5 broken image ref fixed: media-generation-bpe-training-corpus-0-a3e89ce1-79c2-45c3-9a21-5ae0cf780fde.webp -> ...-17dca458-28c4-4e91-9d91-ce048d890047.webp.
- Recap cards 1, 2, 7, 8 repointed to l01-pipeline.svg, l01-text-to-ids.svg, fertility-cost webp, l01-embedding-lookup.svg. Card 2 body text corrected to match the new figure.
- production-tokenizer image line replaced with "The census table below is the current figure."

## Prose problems noticed, NOT touched

- BPE merges webp merges (e,r) in step 2 while prose merges (lo,w) second: same final [low][er], order differs. Noted for figure auditor (F6).
- Card 3 rc-num "26 letters vs 1,000,000+ word forms": prose char vocab is ~100 (bytes), not 26. Noted, not edited (prose-adjacent caption text).
- fertility-languages webp and embedding-lookup webp files retained on disk only because crash-course.md still references them; crash coordinator drops them at rebuild.
