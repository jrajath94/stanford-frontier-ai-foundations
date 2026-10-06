---
page_id: cs336-l17
course_slug: cs336
course_name: "CS336: Language Modeling from Scratch"
course_order: 1
order: 17
nav: "L17 · Multimodality"
title: "Lecture 17: Multimodality"
summary: "Beyond text: the omni-model North Star, CLIP and SigLIP, LLaVA stitching, AnyRes, Qwen-VL generations, and Chameleon's discrete-token alternative."
date: "2026-05-25"
instructor: "Percy Liang"
offering: "Spring 2026"
duration: "1:17:29"
video_id: 26FtD08ZpOU
video_title: "Stanford CS336 Spring 2026 Lecture 17: Multimodality"
video_caption: "Original lecture. Percy Liang on CLIP, VLMs, and the path to omni models."
concepts: [multimodality, clip, siglip, llava, qwen-vl, anyres, m-rope, chameleon, vq-vae]
sources:
  - tag: video
    label: "Lecture 17 video, Stanford Online YouTube"
    url: https://www.youtube.com/watch?v=26FtD08ZpOU
  - tag: notes
    label: "Official subtitle transcript (en-US)"
---

## The problem: the omni model

The North Star: any combination of modalities in, any combination
out. Image plus video in, answer plus generated image out
[01:05](ts:01:05). Transformers speak tokens, so the whole problem is
tokenization: what is the image equivalent of BPE? A pixel is not a
semantic unit, so the answer takes work [01:57](ts:01:57).

![Omni](assets/l17-omni.svg "Anything in, anything out. Everything must become tokens.")

This lecture covers inputs. Generation gets its sketch at the end.

## First attempt: CLIP, contrastive learning

**CLIP** (2021) is the foundation of modern vision-language models.
Take N image-text pairs (N=32,000). Encode each side. The objective:
image I1 must be closer to its text T1 than to every other text, and
vice versa. That is 2N softmax classification problems
[05:53](ts:05:53).

![CLIP](assets/l17-clip.svg "Contrastive image-text learning. Text supplies the semantics.")

Work the toy. N=4 pairs: (I1,T1) a dog photo with "a dog", (I2,T2) a
cat with "a cat", (I3,T3) a car with "a car", (I4,T4) a bird with "a
bird". The loss asks: is I1 closer to T1 than to T2, T3, T4? And is
T1 closer to I1 than to I2, I3, I4? Eight softmax problems. The
encoders learn: pull matched pairs together, push the rest apart.

The vision encoder that won: ViT-L/14, 14x14 patches with positional
embeddings through a standard transformer, attention pooling at the
end [12:41](ts:12:41). A **patch** is a 14x14 square of pixels treated
as one input token: the image's version of a subword. The text
encoder: GPT-2 style, EOS activation as the sequence vector
[16:24](ts:16:24). Data: 400M web image-text pairs, noisy by nature
(captions rarely describe the image literally) [08:24](ts:08:24).

Headline: zero-shot ImageNet beat a ResNet trained on 1.2M labeled
images [17:21](ts:17:21). No labels, just captions, and it classified
better than the supervised champion. Why text and not augmentation
(SimCLR)? Augmentation cannot turn one dog into another dog. Text
supplies high-level semantics [11:24](ts:11:24).

## Where CLIP breaks: the loss is the batch

The softmax runs over all N pairs in the batch. Small batch: few
negatives, weak signal. The loss is the batch, so CLIP needs huge
batches: 32,768. Work the cost: 10 days on 256 TPUv3 [25:33](ts:25:33).
The batch size is not a hyperparameter here. It is the objective.

## The key question

What if the loss did not depend on the batch? What if each pair could
be judged on its own, positive or negative, without the softmax over
everyone else? Then small batches work, and the systems win compounds.

## SigLIP: binary loss, decoupled batches

**SigLIP** replaces the 2N-way softmax with binary classification:
diagonal pairs positive, off-diagonals negative, sigmoid loss
[23:37](ts:23:37). Work the toy again. N=4: 16 pair decisions, each
independent. (I1,T1): positive. (I1,T2): negative. No softmax, no
competition between negatives.

![SigLIP](assets/l17-siglip.svg "Binary loss, decoupled batches, chunked parallelism.")

The loss is now the same in expectation at any batch size, so small
batches work (CLIP degrades) and 32K is the effective ceiling
[27:38](ts:27:38). Systems win: devices compute local losses and
rotate text embeddings to cover off-diagonal blocks, like DDP with
interactions [28:37](ts:28:37). Result: 5 days on 32 TPUv4 versus
CLIP's 10 days on 256 TPUv3 [25:33](ts:25:33).

## LLaVA: stitch, not rebuild

The VLM template: take a vision encoder (CLIP), take an LLM (Vicuna),
stitch them with a projector, train in stages [28:58](ts:28:58).

![LLaVA](assets/l17-llava.svg "CLIP plus a projector plus Vicuna. Align, then fine-tune.")

Image through CLIP, through matrix W into text-embedding space,
concatenated with text tokens through the transformer. Training:
**stage 1** freezes everything and trains only W (alignment).
**Stage 2** freezes the vision encoder and trains W plus the LM
[33:24](ts:33:24). Data: 158k GPT-4-synthesized conversations from
COCO captions [31:07](ts:31:07).

> [!QA]
> Q: Why not train the vision encoder and language model jointly from scratch?
> A: Because you would throw away two excellent pre-trained components. CLIP already knows image semantics. The LLM already knows language and reasoning. The stitching approach (projector plus staged training) reuses both and needs only 158k examples to get visual reasoning working. Joint training from scratch would need billions of image-text pairs and enormous compute to relearn what each side already knows. The tradeoff: the vision encoder stays frozen and classification-flavored (CLIP was built for ImageNet), so fine-grained abilities like OCR need extra machinery (AnyRes) or encoder fine-tuning (Qwen's later choice).
> Follow-up: What does the projector actually learn?
> A: A mapping from CLIP's embedding space into the LLM's token embedding space. After stage 1, an image encoding should "look like" a sequence of token embeddings to the transformer: same dimensionality, same rough distribution. It is alignment, not understanding: the understanding was already in CLIP and in the LLM. The projector just introduces them.

## Where stitching breaks: resolution

CLIP's 336x336 resize-and-crop cannot read documents. Work the
failure: a page of text at 336x336 is about 10 pixels per character.
Nothing is legible. The encoder was built for ImageNet, not OCR.

**AnyRes** splits the image into encoder-sized crops, encodes each,
and concatenates, plus one downsampled overview [37:36](ts:37:36).
Crop, do not downsample. A 1344x1344 document becomes 16 crops of
336x336 plus one overview: each crop is legible, the overview keeps
layout. Adaptive: big images get more crops, videos get fewer tokens
per frame (up to 32 frames) [40:09](ts:40:09).

![AnyRes](assets/l17-anyres.svg "Crop, do not downsample. Modalities transfer.")

LLaVA OneVision (2024): SigLIP encoder, Qwen-2 decoder, 2-layer MLP
projector, three training stages [35:25](ts:35:25). The surprise:
cross-modal transfer. Single-image OCR data plus multi-image
relational data generalize to two-image table-plus-chart questions
and video visual prompting, neither seen in training
[43:30](ts:43:30).

## Qwen-VL: three generations of sharper pieces

![Qwen](assets/l17-qwen.svg "Qwen-VL to Qwen3-VL: dynamic resolution, better positions, deeper fusion.")

**Qwen-VL**: OpenCLIP, cross-attention adapter to 256 tokens, three
stages with 1.4B examples in stage 1, bounding-box outputs
[46:06](ts:46:06). **Qwen2-VL**: dynamic resolution (11k tokens for a
big image, 8 for a tiny equation), 2x2 token compression, **M-RoPE**:
RoPE over height, width, and time concatenated [49:16](ts:49:16).
Position now has three axes: where in the image, when in the video.
**Qwen3-VL**: SigLIP-2, interleaved M-RoPE frequencies (every axis
sees high and low frequencies), explicit timestamp tokens,
sqrt-length normalized loss so video does not dominate, and
**DeepStack**: vision layers fused directly into the LM residual
stream [52:50](ts:52:50). Context to 256K for long video. Training
runs 8K to 32K to 256K.

## Chameleon: the discrete alternative

The elegant alternative: make everything discrete tokens. **VQ-VAE**
maps a 512x512 image to 1,024 tokens from an 8,000-code codebook. Then
it is just language-model training, no adapter [67:18](ts:67:18).
Interleaved text and images, true omni-style generation.

![Chameleon](assets/l17-chameleon.svg "One token space for everything. Entropy fights back.")

The problems, demonstrated. Image tokens are high-entropy (which exact
blue?): parameter norms grow and training destabilizes. QK-norm and
z-loss mitigate [72:11](ts:72:11). Discretization loses fine detail,
so OCR suffers: the codebook has 8,000 entries for all possible
image patches, and small text falls between codes. Diffusion later
won generation. VQ-VAE faded [74:28](ts:74:28).

## Mapping back: what each idea fixes

| Pain | Fix | How |
|---|---|---|
| Pixels are not semantic units | CLIP patches | 14x14 patches as tokens. Text supplies semantics via contrastive loss. |
| Loss needs huge batches | SigLIP | Binary sigmoid loss. Decoupled batches. 5 days vs 10. |
| Two good components, no bridge | LLaVA stitching | Projector W, staged training. 158k examples. |
| 336x336 cannot read documents | AnyRes | Crop, not downsample. Overview plus detail crops. |
| One resolution, one position | Qwen generations | Dynamic resolution, M-RoPE over height/width/time, DeepStack fusion. |
| Adapters are inelegant | Chameleon | VQ-VAE discrete tokens. Elegant, unstable, detail-losing. |

## The honest price

Frontier models are natively multimodal, details undisclosed. The best
guess: continuous encoders for understanding (no information loss),
diffusion for generation (micro-optimized detail) [74:43](ts:74:43).
That picture is the lecturer's speculation, flagged as such. Compute
comparisons (TPUv3 vs v4) are not flops-normalized. Chameleon results
were not shown in lecture. Persistent challenges: no universal encoder
(classification wants semantics, OCR wants pixels), modality weighting
(video is low-density, do not let it drown text), and systems (video
loading alone can bottleneck training) [61:16](ts:61:16).

![Summary](assets/l17-multimodal-summary.svg "Encode continuously, generate with diffusion, weight carefully.")

## Recap: the whole lesson on one screen

The story in eight steps. Each step answers the one before it.

1. **The omni goal.** Any modality in, any out. Transformers speak
   tokens. A pixel is not a semantic unit: tokenization takes work.
2. **CLIP contrasts.** 2N softmax problems on 400M pairs. ViT-L/14
   patches. Zero-shot ImageNet beats supervised ResNet. Text gives
   semantics.
3. **The loss is the batch.** CLIP needs 32K batches: 10 days on 256
   TPUv3. The batch size is the objective.
4. **SigLIP decouples.** Binary sigmoid loss, independent pair
   decisions. Small batches work. 5 days on 32 TPUv4.
5. **LLaVA stitches.** CLIP plus projector plus Vicuna. Align W,
   then fine-tune. 158k examples. The projector aligns spaces.
6. **AnyRes crops.** 336x336 cannot read documents. Crop, not
   downsample. Overview plus details. Modalities transfer.
7. **Qwen sharpens.** Dynamic resolution, M-RoPE over 3 axes,
   DeepStack fusion into the residual stream. 256K context.
8. **Chameleon discretizes.** VQ-VAE: elegant, unstable (entropy),
   detail-losing. Diffusion won generation. Continuous encode,
   diffusion generate.

## Official sources and further reading

**Official:**
- Lecture 17 video.
- CLIP, SigLIP, LLaVA, LLaVA-OneVision, Qwen-VL series, Chameleon
  papers.

**Further reading:**
- LAION-5B / OpenCLIP (open replication).
- VQ-VAE (Oord 2017). DeepStack adapter paper.
- AI2's VLM report (data details the lecture skipped).

**Caveats from these sources.** Frontier omni models are undisclosed.
The "continuous plus diffusion" picture is the lecturer's speculation,
flagged as such. Compute comparisons (TPUv3 vs v4) are not
flops-normalized. Chameleon results were not shown in lecture.

## Connections to the other courses

- **CS336 L01:** tokenization generalized beyond text.
- **CS336 L03:** RoPE generalized to M-RoPE.
- **CS224N:** vision-language models from the NLP side.
