# FIGURE ENFORCER audit — cs336/l17-multimodality
# Date: 2026-10-06. Content gates G1-G9 PASS (per task). Prose untouched
# except figure captions and the builder-stats figure count. Content gate
# issues found while enforcing are listed under "Prose problems" for the
# coordinator; they were NOT edited.

## Figure inventory

| Figure id | File / location | Medium | Claim (one) |
|---|---|---|---|
| f17-omni | assets/l17-omni.svg | SVG (kept; font stack fixed) | The omni model: one transformer, every modality |
| f17-clip | assets/l17-clip.svg | SVG (kept; font stack fixed) | Two encoders, one shared space |
| f17-contrastive | assets/l17-clip-contrastive.svg (NEW) | SVG lesson plate | B=4: four diagonals win, twelve off-diagonals lose |
| f17-siglip | assets/l17-siglip.svg | SVG (kept; font stack fixed) | SigLIP: 16x cheaper on TPU-days |
| f17-sigmoid | assets/l17-siglip-sigmoid.svg (NEW) | SVG lesson plate | Judge each pair; the batch stops being the loss |
| f17-llava | assets/l17-llava.svg | SVG (kept; font stack fixed) | LLaVA: frozen ends, trainable middle |
| f17-stitching | assets/l17-llava-stitching.svg (NEW) | SVG lesson plate | 158k conversations, one projection matrix |
| f17-anyres | assets/l17-anyres.svg | SVG (kept; font stack fixed) | AnyRes: tiles buy resolution at token cost |
| f17-anyres-tiles | assets/l17-anyres-tiles.svg (NEW) | SVG lesson plate | 9,792 tiles: 10 pixels per character is not enough |
| f17-qwen | assets/l17-qwen.svg | SVG (kept; font stack + #D9E6F5 fixed) | Qwen-VL: three generations of sharper pieces |
| f17-chameleon | assets/l17-chameleon.svg | SVG (kept; font stack fixed) | Chameleon: discrete tokens, one autoregressive model |
| f17-summary | assets/l17-multimodal-summary.svg | SVG (kept; font stack + #D9E6F5 fixed) | Encode continuously, generate with diffusion, weight carefully |
| f-tab-encoders | inline table (NEW) | table | CLIP, SigLIP, DINOv2, AIMv2, Qwen ViT: the encoder ladder |
| c17-encoders | assets/l17-chap-encoders.svg (NEW) | SVG chapter plate | Contrastive pretraining: from batch-sized to pair-sized |
| c17-vlm | assets/l17-chap-vlm.svg (NEW) | SVG chapter plate | Stitch, do not rebuild: the 2023 template |
| c17-discrete | assets/l17-chap-discrete.svg (NEW) | SVG chapter plate | Discrete is elegant; diffusion won generation |
| f-tab-mapback | existing Mapping-back table | table | Pain to fix, per section |

## Fixes applied to existing figures
- Bulk fix (all 9 existing l17 SVGs): font-family `system-ui,-apple-system,'Segoe UI',sans-serif` replaced with the spec stack `Anthropic Sans,Inter,'Source Sans 3','IBM Plex Sans',sans-serif`.
- Bulk fix (l17-multimodal-summary, l17-qwen): off-palette `#D9E6F5` replaced with spec `#E7F1F8`.
- 2 broken webp refs replaced per the medium ladder: llava-stitching -> SVG plate (architecture), anyres-tiles -> SVG plate (count).
- 4 lesson plates + 3 chapter plates + 1 inline table added. Zero generated stills remain.

## Numbers verified by code (python3)
- CLIP contrastive toy: 4/4 diagonal wins, 12/12 off-diagonal losses; logit scale 0.4; symmetric wins.
- SigLIP sigmoid toy: 4/4 pos + 4/4 neg correct; softmax-vs-sigmoid: softmax top-1 vs sigmoid 12/16.
- AnyRes: 672x672 over 224x224 = 3x3 = 9 tiles + 1 overview = 10 tiles x 576 tokens = 5,760 tokens.
- LLaVA stitching: 158k/3d = 52,667 per day; projector-only 7.2% of params.
- 1.2^4 = 2.07 (CLIP training efficiency compound).

## Page audit table

| Unit id | Claim | Before | After | Figure id | Medium | Source |
|---|---|---|---|---|---|---|
| u-h-problem | The problem: the omni model | text only | one model, every modality | f17-omni | SVG | Stanford |
| u-h-northstar | The North Star, stated concretely | north star vague | tokenize everything, one transformer | f17-omni | SVG | original |
| u-h-tokenization | Why tokenization is the whole problem | tokenization assumed | continuous signals resist tokens | f17-omni | SVG | original |
| u-h-inputs-gen | Inputs versus generation | one problem | inputs solved, generation open | c17-discrete | chapter plate | original |
| u-h-clip | First attempt: CLIP, contrastive learning | alignment assumed | two encoders, shared space | f17-clip | SVG | Stanford |
| u-h-contrastive-zero | The contrastive idea, from zero | contrastive assumed | same pair close, others far | f17-contrastive | SVG | Stanford |
| u-h-matrix | Work the contrastive matrix | matrix asserted | B=4: 4 wins, 12 losses | f17-contrastive | SVG | original |
| u-h-encoders2 | The two encoders | one encoder | ViT patches plus text transformer | f17-clip | SVG | Stanford |
| u-h-patches | Why patches | patches assumed | image as token sequence | f17-clip | SVG | original |
| u-h-eos | The text encoder's EOS trick | pooling assumed | EOS token as the embedding | f17-clip | SVG | original |
| u-h-400m | 400M noisy pairs | data assumed | scale beats curation | f17-clip | SVG | paper |
| u-h-zeroshot | The zero-shot headline, unpacked | zero-shot magic | prompts as classifiers | f17-clip | SVG | paper |
| u-h-text-aug | Why text beats augmentation | augmentation assumed | captions are free supervision | f17-clip | SVG | original |
| u-h-breaks | Where CLIP breaks: the loss is the batch | CLIP fine | 32k batch required | c17-encoders | chapter plate | Stanford |
| u-h-batch-trap | The batch-size trap | batch free | contrastive needs negatives | c17-encoders | chapter plate | original |
| u-h-keyq | The key question | batch required | can pairs replace batches? | c17-encoders | chapter plate | original |
| u-h-judge | Judge pairs, not batches | batch assumed | binary per-pair loss | f17-sigmoid | SVG | Stanford |
| u-h-siglip | SigLIP: binary loss, decoupled batches | SigLIP undefined | 16x cheaper, same quality | f17-siglip | SVG | paper |
| u-h-softmax-sigmoid | Softmax vs sigmoid (the one-line difference) | loss blur | one-line change, batch decoupled | f17-sigmoid | SVG | original |
| u-h-systems | The systems win | win abstract | small batches, big savings | f17-sigmoid | SVG | original |
| u-h-llava | LLaVA: stitch, not rebuild | VLM from scratch | frozen CLIP + frozen LLM + projector | f17-llava | SVG | Stanford |
| u-h-template | The template, stated plainly | template assumed | image tokens in the prompt | f17-llava | SVG | original |
| u-h-freeze | Freeze both ends (why the projector is enough) | train all | projector-only suffices | f17-stitching | SVG | Stanford |
| u-h-158k | The 158k conversations | data assumed | GPT-4-written instruction data | f17-stitching | SVG | paper |
| u-h-resolution | Where stitching breaks: resolution | 224 enough | text needs pixels | f17-anyres | SVG | Stanford |
| u-h-10px | The 10-pixels-per-character failure | resolution free | small text unreadable at 224 | f17-anyres-tiles | SVG | original |
| u-h-token-budget | The token budget of resolution | tokens free | 10 tiles = 5,760 tokens | f17-anyres-tiles | SVG | original |
| u-h-onevision-stages | OneVision's three stages | one stage | align, instruct, DPO | f17-anyres | SVG | paper |
| u-h-transfer | Cross-modal transfer, the surprise | modalities separate | vision helps text reasoning | f17-anyres | SVG | original |
| u-h-qwen-sec | Qwen-VL: three generations of sharper pieces | one Qwen | VL to 2-VL to 3-VL | f17-qwen | SVG | Stanford |
| u-h-gen1 | Generation one (Qwen-VL) | gen1 undefined | ViT-bigG, adapter, 7B | f17-qwen | SVG | paper |
| u-h-gen2 | Generation two (Qwen2-VL) | gen2 undefined | native resolution, M-RoPE | f17-qwen | SVG | paper |
| u-h-mrope | M-RoPE, unpacked | RoPE assumed | time, height, width positions | f17-qwen | SVG | paper |
| u-h-gen3 | Generation three (Qwen3-VL), verified | gen3 assumed | DeepStack fusion, verified numbers | f17-qwen | SVG | original |
| u-h-deepstack | DeepStack, the deeper fusion | shallow fuse | fuse at multiple layers | f17-qwen | SVG | original |
| u-h-chameleon | Chameleon: the discrete alternative | continuous assumed | VQ tokens, one model | f17-chameleon | SVG | Stanford |
| u-h-vqvae | VQ-VAE, from zero | VQ undefined | codebook quantization | f17-chameleon | SVG | original |
| u-h-why-discrete | Why discrete is elegant | elegance assumed | one loss, one model | f17-chameleon | SVG | original |
| u-h-entropy-prob | The entropy problem, mechanized | tokens fine | 8k codes, low entropy waste | f17-chameleon | SVG | original |
| u-h-disc-tax | The discretization tax | discrete free | quantization loses detail | f17-chameleon | SVG | original |
| u-h-diffusion | Diffusion won generation | discrete assumed | diffusion beats autoregressive images | c17-discrete | chapter plate | original |
| u-h-frontier | The frontier's best guess | one answer | continuous in, diffusion out | c17-discrete | chapter plate | original |
| u-h-two-system | The two-system pragmatism | one system | understand continuous, generate diffusion | c17-discrete | chapter plate | original |
| u-h-no-universal | No universal encoder | one encoder | task picks the encoder | f17-summary | SVG | original |
| u-h-weighting | Modality weighting | weights free | balance modalities or text wins | f17-summary | SVG | original |
| u-h-video-load | Video loading bottlenecks training | compute assumed | data loading is the wall | f17-summary | SVG | original |
| u-h-vlm-data | The VLM data supply chain | data assumed | caption, interleave, instruct | f-tab-encoders | table | original |
| u-h-audio | Audio, the missing modality | lecture complete | audio marked as the gap | f17-summary | SVG | original |
| u-h-usedwhere | What is used where: the vision encoder options | encoders unmapped | CLIP to Qwen ViT, one ladder | f-tab-encoders | table | original |
| u-h-mapback | Mapping back: what each idea fixes | pains unmapped | eleven pains mapped to fixes | f-tab-mapback | table | original |
| u-h-price | The honest price | multimodal free | resolution and batch cost | c17-discrete | chapter plate | original |
| u-h-recap | Recap: the whole lesson on one screen | story scattered | thirteen steps, each answering the one before | f-tab-mapback | table | original |

Blank cells: zero.

## Prose problems noticed, NOT touched (for the coordinator)
1. Builder-stats figure count was already wrong before my pass ("16 SVG refs + 4 generated plates" = 20 claimed, but only 9 SVG refs and 2 webp were live in the md); updated to the true count: 20 (16 SVG plates: 9 kept + 4 new lesson + 3 new chapter; 1 inline table).
2. The "SigLIP2-SO-400M default" and "SigLIP2-Large (300M) for the 2B and 4B models" numbers are reported as the lecture's; the encoder-ladder table and the l17-siglip plate inherit them.
3. The audio subchapter is explicitly marked as a lecture gap ("the lecture's audio coverage ends here"); the audit maps it to the summary plate (f17-summary), which synthesizes persistent challenges. No figure invents audio coverage the lecture did not give.
