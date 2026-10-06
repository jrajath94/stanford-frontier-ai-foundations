# Figure Audit — CS229 lessons L13–L17
# Figure Enforcer report. Generated 2026-10-06.
# Scope: l13, l14, l15, l16, l17 only. l01–l12, crash-course, cheatsheet, index untouched.

## Method
- Read MASTER_BRIEF.md, REJECTIONS.md, PIPELINE.md, FIGURE_SPEC_STANFORD.md verbatim.
- Grepped each lesson for `assets/plate-*.webp`, `assets/svg/*.svg`, ```ascii blocks, youtube-nocookie embeds.
- Checked every referenced file exists and is non-empty on disk.
- Generated the 8 missing SVGs via `assets/svg/make_missing_l13_17.py`
  (Python SVG rendering, same visual system as the existing 44 SVGs:
  warm paper #F7F4EE, ink #1B2838, flat fills, 8px grid).
  AI image generation was NOT used: F6 requires code-computed numbers
  (softmax weights, distances, byte counts), which AI image models cannot
  guarantee. All numbers in the new figures are asserted in code.
- Rendered every new SVG to PNG (cairosvg) and visually inspected for
  overlap/clipping; fixed 6 layout issues across 2 revision rounds.

## Result: 0 missing figures. All captions present in verified-caption format.

Caption format verified: plates end `Source: <named source>. Project: Stanford Frontier AI.`;
SVGs end `Source: original plate for Stanford Frontier AI.` Every figure reference in
the lessons carries such a caption.

---

## L13 — l13-contrastive-rag.md (contrastive learning + RAG)

| Figure referenced | Status | Caption |
|---|---|---|
| assets/plate-l13-false-negative.webp | EXISTS on disk (135,266 B, valid WebP) | OK — "The false-negative trap… Source: original plate for the mining poison. Project: Stanford Frontier AI." |
| assets/plate-l13-loss-audit.webp | EXISTS on disk (121,910 B, valid WebP) | OK — "The loss, audited… Source: original audit for the contrastive loss. Project: Stanford Frontier AI." |
| assets/plate-l13-refund-priced.webp | EXISTS on disk (135,266 B, valid WebP) | OK — "The refund toy, priced… Source: original plate for the RAG budget. Project: Stanford Frontier AI." |
| assets/plate-l13-tau.webp | EXISTS on disk (121,910 B, valid WebP) | OK — "Tau, the sharpness dial… Source: original plate for the temperature dial. Project: Stanford Frontier AI." |
| assets/svg/l13-contrastive.svg | EXISTS on disk (2,121 B) | OK — "Contrastive learning… Source: original plate for Stanford Frontier AI." |
| assets/svg/l13-rag.svg | EXISTS on disk (2,775 B) | OK — "Retrieval-augmented generation… Source: original plate for Stanford Frontier AI." |
| assets/svg/l13-rerank.svg | GENERATED 2026-10-06 (2,889 B, valid XML, visually QC'd) | OK — pre-written in text: "Retrieve then rerank. ANN: 10M to 100, fast. Cross-encoder: 100 to 5, careful. LLM reads 5. Source: original plate for Stanford Frontier AI." Figure content: vertical funnel 10M→100→5 with the lesson's refund-window toy (bi-encoder 0.82 vs 0.71, cross-encoder 0.94 vs 0.12). |
| assets/svg/l13-triplet.svg | GENERATED 2026-10-06 (4,019 B, valid XML, visually QC'd) | OK — pre-written in text: "Triplet loss vs InfoNCE. Triplet: one negative must lose by a margin. InfoNCE: all negatives compete in one softmax. Source: original plate for Stanford Frontier AI." Figure content: lesson toy, d(A,P)=0.141, d(A,N)=1.204, margin 0.2, L=0; InfoNCE softmax weights 0.9990/0.0003×3 (sum 1.0000), loss 0.0010, all code-computed. |
| ascii block (line 84) | in text | InfoNCE loss formula trace |
| ascii block (line 190) | in text | similarity-matrix audit (c1/c2/d1/d2) |
| youtube-nocookie embed 7Id8SPH31UE | in text (2 embeds total) | — |
| youtube-nocookie embed xkiC3gg1AlM | in text | — |

## L14 — l14-transformers.md

| Figure referenced | Status | Caption |
|---|---|---|
| assets/plate-l14-block-count.webp | EXISTS (119,052 B, valid WebP) | OK — "A block, counted… Source: original plate for the parameter count. Project: Stanford Frontier AI." |
| assets/plate-l14-exposure.webp | EXISTS (145,940 B, valid WebP) | OK — "Exposure bias, priced… Source: original plate for the train-generate gap. Project: Stanford Frontier AI." |
| assets/plate-l14-sqrt-d.webp | EXISTS (148,392 B, valid WebP) | OK — "Why divide by sqrt(d)… Source: original plate for the scaling audit. Project: Stanford Frontier AI." |
| assets/plate-l14-toy-audit.webp | EXISTS (148,548 B, valid WebP) | OK — "The toy, audited… Source: original audit for the attention toy. Project: Stanford Frontier AI." |
| assets/svg/l14-attn.svg | EXISTS (2,816 B) | OK — "Attention… Source: original plate for Stanford Frontier AI." |
| assets/svg/l14-mask.svg | EXISTS (5,230 B) | OK — "The causal mask… Source: original plate for Stanford Frontier AI." |
| assets/svg/l14-t2.svg | EXISTS (1,571 B) | OK — "The transformer block… Source: original plate for Stanford Frontier AI." |
| assets/svg/l14-tokenize.svg | GENERATED 2026-10-06 (4,685 B, valid XML, visually QC'd) | OK — pre-written: "Tokenization. Text to pieces to ids to vectors. BPE merges frequent pairs. Byte fallback: no unknown tokens. Source: original plate for Stanford Frontier AI." Content: 4-stage pipeline with BPE "lowest"→"low est" toy, 256-byte fallback, vocab sizes 32k/50k/128k/256k, embedding matrix 128,256×4096 = 525M params (asserted in code). |
| assets/svg/l14-mha.svg | GENERATED 2026-10-06 (5,365 B, valid XML, visually QC'd) | OK — pre-written: "Multi-head attention. Width d splits into h heads of d_h. Same total compute as one head. Capacity is in the diversity of mixes. Source: original plate for Stanford Frontier AI." Content: d=4096 → 32 heads × 128 → concat → W_O; cost audit 32×N²×128 = N²×4096 (asserted). |
| assets/svg/l14-rope.svg | GENERATED 2026-10-06 (3,394 B, valid XML, visually QC'd) | OK — pre-written: "RoPE. Queries and keys rotate by position. The dot product keeps only the relative distance. Source: original plate for Stanford Frontier AI." Content: worked 2-D toy, θ=0.5, m=3, n=1; q=[0.0707,0.9975], k=[0.8776,0.4794], dot=0.5403=cos(1.0) (asserted, <1e-9). |
| assets/svg/l14-flash.svg | GENERATED 2026-10-06 (7,351 B, valid XML, visually QC'd) | OK — pre-written: "FlashAttention. Tile QKV into SRAM, online softmax across tiles. Exact math, O(N) memory traffic. Source: original plate for Stanford Frontier AI." Content: HBM N×N (67,108,864 floats/head at N=8192, asserted) vs SRAM tiles + online-softmax accumulator (m, l, O). |
| ascii block (line 71) | in text | autoregressive factorization trace |
| ascii block (line 212) | in text | attention toy walkthrough (weights 0.21/0.21/0.58, frame [0.79, 0.79]) |
| youtube-nocookie embed eMlx5fFNoYc | in text (2 embeds total) | — |
| youtube-nocookie embed wjZofJX0v4M | in text | — |

## L15 — l15-efficient-icl-sft.md

| Figure referenced | Status | Caption |
|---|---|---|
| assets/plate-l15-cache-bytes.webp | EXISTS (140,998 B, valid WebP) | OK — "The cache in bytes… Source: original plate for the KV memory. Project: Stanford Frontier AI." |
| assets/plate-l15-cubic-audit.webp | EXISTS (141,438 B, valid WebP) | OK — "The cubic, audited… Source: original audit for the generation cost. Project: Stanford Frontier AI." |
| assets/plate-l15-gqa-counted.webp | EXISTS (120,436 B, valid WebP) | OK — "GQA, counted… Source: original plate for the GQA arithmetic. Project: Stanford Frontier AI." |
| assets/plate-l15-icl-brittle.webp | EXISTS (146,462 B, valid WebP) | OK — "ICL is brittle… Source: original plate for the brittleness. Project: Stanford Frontier AI." |
| assets/svg/l15-icl.svg | EXISTS (1,644 B) | OK — "In-context learning… Source: original plate for Stanford Frontier AI." |
| assets/svg/l15-kvcache.svg | EXISTS (1,871 B) | OK — "The KV cache… Source: original plate for Stanford Frontier AI." |
| assets/svg/l15-moe.svg | EXISTS (3,598 B) | OK — "Mixture of experts… Source: original plate for Stanford Frontier AI." |
| assets/svg/l15-sft.svg | EXISTS (2,244 B) | OK — "Supervised fine-tuning… Source: original plate for Stanford Frontier AI." |
| assets/svg/l15-mla.svg | GENERATED 2026-10-06 (4,322 B, valid XML, visually QC'd) | OK — pre-written: "MLA. Store one latent vector per token, reconstruct K and V on the fly. 16x smaller than MHA cache. Source: original plate for Stanford Frontier AI." Content: MHA 16 KB (32×128×2×2, asserted) → latent c 512 dims = 1 KB (asserted), 16x (asserted); up-projection absorption note; decoupled RoPE note. |
| assets/svg/l15-paged.svg | GENERATED 2026-10-06 (7,676 B, valid XML, visually QC'd) | OK — pre-written: "PagedAttention. Fixed-size KV blocks, block tables, allocate on demand. Fragmentation dies. Source: original plate for Stanford Frontier AI." Content: lesson fragmentation numbers — 10 reqs × 4096 × 512 KB = 20 GB reserved, 2.4 GB used, 12% (asserted); paged 2.4/2.5 GB, 96% (asserted, matches lesson rounding); logical→physical block table. |
| ascii block (line 495) | in text | few-shot translation trace |
| youtube-nocookie embed emZxqRScfc0 | in text (2 embeds total) | — |
| youtube-nocookie embed tGp6Ns9GtSU | in text | — |

## L16 — l16-reinforcement-learning.md

| Figure referenced | Status | Caption |
|---|---|---|
| assets/plate-l16-baseline.webp | EXISTS (154,450 B, valid WebP) | OK — "Subtract the luck… Source: original plate for the baseline. Project: Stanford Frontier AI." |
| assets/plate-l16-credit.webp | EXISTS (156,190 B, valid WebP) | OK — "Who earned the +10?… Source: original plate for the credit assignment. Project: Stanford Frontier AI." |
| assets/plate-l16-value-audit.webp | EXISTS (201,516 B, valid WebP) | OK — "The value walk, audited… Source: original plate for the baseline. Project: Stanford Frontier AI." |
| assets/svg/l16-mdp.svg | EXISTS (1,991 B) | OK — "The Markov decision process… Source: original plate for Stanford Frontier AI." |
| assets/svg/l16-pg.svg | EXISTS (1,523 B) | OK — "REINFORCE… Source: original plate for Stanford Frontier AI." |
| ascii block (line 275) | in text | Bellman equation |
| ascii block (line 500) | in text | policy-gradient expectation |
| ascii block (line 509) | in text | REINFORCE procedure trace |
| youtube-nocookie embed cvGh1NMTq8A | in text (2 embeds total) | — |
| youtube-nocookie embed mfjWWOsCNIo | in text | — |

No figures missing in L16.

## L17 — l17-rl-for-llms.md

| Figure referenced | Status | Caption |
|---|---|---|
| assets/plate-l17-blind-spot.webp | EXISTS (145,754 B, valid WebP) | OK — "The verifier's blind spot… Source: original plate for the reward hacking. Project: Stanford Frontier AI." |
| assets/plate-l17-clip-cases.webp | EXISTS (162,372 B, valid WebP) | OK — "The four cases, corrected… Source: original audit of the PPO objective. Project: Stanford Frontier AI." |
| assets/plate-l17-ratio.webp | EXISTS (123,294 B, valid WebP) | OK — "The ratio, priced… Source: original plate for the importance weights. Project: Stanford Frontier AI." |
| assets/svg/l17-ppo.svg | EXISTS (1,371 B) | OK — "PPO. The importance ratio r is clipped to [0.8, 1.2]… Source: original plate for Stanford Frontier AI." |
| ascii block (line 233) | in text | PPO clipped objective |
| youtube-nocookie embed HWo8LNcBLdc | in text (2 embeds total) | — |
| youtube-nocookie embed iSvC5VmDHL4 | in text | — |

No figures missing in L17.

---

## Totals

- Lesson plates (webp): 16 referenced, 16 exist, 16 captioned.
- Section figures (SVG): 20 referenced — 12 existed, 8 generated, 20 captioned.
- ASCII walkthrough blocks: 9 across the five lessons (all in text, ≤12 lines each).
- YouTube-nocookie embeds: 10 (2 per lesson).

## Notes for the parent / figure auditor

1. Generator script: `content/v2/cs229/assets/svg/make_missing_l13_17.py`
   (imports helpers from the existing `make_plates.py`; follows the same visual
   system). Every figure number is asserted in code (F6); a post-pass greps the
   SVG text for the computed values.
2. Reject-list compliance: flat fills only; no gradients, glow, shadows, 3D,
   logos, watermarks, robots, brains, clip art; one claim per plate; arrows name
   real operations; type stack matches the existing 44 SVGs.
3. Orphan plates on disk, NOT referenced by l13–l17 lessons (left untouched;
   possibly referenced by crash-course/cheatsheet, which are out of scope):
   `assets/plate-l16-greedy.webp`, `assets/plate-l17-advantage.webp`.
4. `assets/figs-notes/*.png` are not referenced by any of the five lessons.
5. YouTube video-ID liveness (oEmbed verification) was NOT re-checked here —
   embeds are in place per the medium audit, but ID validity belongs to the
   media-laws check, not the figure enforcer.
6. Did not run build/build.py, did not touch l01–l12, crash-course, cheatsheet,
   or index files.
