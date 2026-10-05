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

## How to read this lesson

One lecture, one new world: images into and out of transformers.
**Level 1 (Core):** the omni goal, CLIP, SigLIP, LLaVA. **Level 2
(Deep):** AnyRes, Qwen generations, Chameleon, the state of play.

## Level 1: The omni model

The North Star: any combination of modalities in, any combination
out. Image plus video in, answer plus generated image out
[01:05](ts:01:05). Transformers speak tokens, so the whole problem
is tokenization: what is the image equivalent of BPE? A pixel is not
a semantic unit, so the answer takes work [01:57](ts:01:57).

![Omni](assets/l17-omni.svg "Anything in, anything out. Everything must become tokens.")

This lecture covers inputs. Generation gets its sketch at the end.

## Level 1: CLIP

CLIP (2021) is the foundation of modern vision-language models.
Take N image-text pairs (N=32,000). Encode each side. The objective:
image I1 must be closer to its text T1 than to every other text,
and vice versa. That is 2N softmax classification problems
[05:53](ts:05:53).

![CLIP](assets/l17-clip.svg "Contrastive image-text learning. Text supplies the semantics.")

The vision encoder that won: ViT-L/14, 14x14 patches with positional
embeddings through a standard transformer, attention pooling at the
end [12:41](ts:12:41). The text encoder: GPT-2 style, EOS activation
as the sequence vector [16:24](ts:16:24). Data: 400M web image-text
pairs, noisy by nature (captions rarely describe the image
literally) [08:24](ts:08:24). Headline: zero-shot ImageNet beat a
ResNet trained on 1.2M labeled images [17:21](ts:17:21). Why text and
not augmentation (SimCLR)? Augmentation cannot turn one dog into
another dog. Text supplies high-level semantics [11:24](ts:11:24).

## Level 1: SigLIP

CLIP needs huge batches because the loss is the batch: the softmax
runs over all N. SigLIP replaces it with binary classification:
diagonal pairs positive, off-diagonals negative, sigmoid loss
[23:37](ts:23:37).

![SigLIP](assets/l17-siglip.svg "Binary loss, decoupled batches, chunked parallelism.")

The loss is now the same in expectation at any batch size, so small
batches work (CLIP degrades) and 32K is the effective ceiling
[27:38](ts:27:38). Systems win: devices compute local losses and
rotate text embeddings to cover off-diagonal blocks, like DDP with
interactions [28:37](ts:28:37). Result: 5 days on 32 TPUv4 versus
CLIP's 10 days on 256 TPUv3 [25:33](ts:25:33).

## Level 1: LLaVA

The VLM template: take a vision encoder (CLIP), take an LLM
(Vicuna), stitch them with a projector, train in stages
[28:58](ts:28:58).

![LLaVA](assets/l17-llava.svg "CLIP plus a projector plus Vicuna. Align, then fine-tune.")

Image through CLIP, through matrix W into text-embedding space,
concatenated with text tokens through the transformer. Training:
stage 1 freezes everything and trains only W (alignment). Stage 2
freezes the vision encoder and trains W plus the LM
[33:24](ts:33:24). Data: 158k GPT-4-synthesized conversations from
COCO captions [31:07](ts:31:07).

> [!QA]
> Q: Why not train the vision encoder and language model jointly from scratch?
> A: Because you would throw away two excellent pre-trained components. CLIP already knows image semantics. The LLM already knows language and reasoning. The stitching approach (projector plus staged training) reuses both and needs only 158k examples to get visual reasoning working. Joint training from scratch would need billions of image-text pairs and enormous compute to relearn what each side already knows. The tradeoff: the vision encoder stays frozen and classification-flavored (CLIP was built for ImageNet), so fine-grained abilities like OCR need extra machinery (AnyRes) or encoder fine-tuning (Qwen's later choice).
> Follow-up: What does the projector actually learn?
> A: A mapping from CLIP's embedding space into the LLM's token embedding space. After stage 1, an image encoding should "look like" a sequence of token embeddings to the transformer: same dimensionality, same rough distribution. It is alignment, not understanding: the understanding was already in CLIP and in the LLM. The projector just introduces them.

## Level 2: AnyRes and OneVision

CLIP's 336x336 resize-and-crop cannot read documents. AnyRes splits
the image into encoder-sized crops, encodes each, and concatenates,
plus one downsampled overview [37:36](ts:37:36). Adaptive: big images
get more crops, videos get fewer tokens per frame (up to 32 frames)
[40:09](ts:40:09).

![AnyRes](assets/l17-anyres.svg "Crop, do not downsample. Modalities transfer.")

LLaVA OneVision (2024): SigLIP encoder, Qwen-2 decoder, 2-layer MLP
projector, three training stages [35:25](ts:35:25). The surprise:
cross-modal transfer. Single-image OCR data plus multi-image
relational data generalize to two-image table-plus-chart questions
and video visual prompting, neither seen in training
[43:30](ts:43:30).

## Level 2: Qwen-VL generations

![Qwen](assets/l17-qwen.svg "Qwen-VL to Qwen3-VL: dynamic resolution, better positions, deeper fusion.")

Qwen-VL: OpenCLIP, cross-attention adapter to 256 tokens, three
stages with 1.4B examples in stage 1, bounding-box outputs
[46:06](ts:46:06). Qwen2-VL: dynamic resolution (11k tokens for a
big image, 8 for a tiny equation), 2x2 token compression, M-RoPE:
RoPE over height, width, and time concatenated [49:16](ts:49:16).
Qwen3-VL: SigLIP-2, interleaved M-RoPE frequencies (every axis sees
high and low frequencies), explicit timestamp tokens, sqrt-length
normalized loss so video does not dominate, and DeepStack: vision
layers fused directly into the LM residual stream [52:50](ts:52:50).
Context to 256K for long video. Training runs 8K to 32K to 256K.

## Level 2: Chameleon

The elegant alternative: make everything discrete tokens. VQ-VAE
maps a 512x512 image to 1,024 tokens from an 8,000-code codebook.
Then it is just language-model training, no adapter [67:18](ts:67:18).
Interleaved text and images, true omni-style generation.

![Chameleon](assets/l17-chameleon.svg "One token space for everything. Entropy fights back.")

The problems: image tokens are high-entropy (which exact blue?),
so parameter norms grow and training destabilizes. QK-norm and
z-loss mitigate [72:11](ts:72:11). Discretization loses fine detail,
so OCR suffers. Diffusion later won generation. VQ-VAE faded
[74:28](ts:74:28).

## Level 2: The state of play

Frontier models are natively multimodal, details undisclosed. The
best guess: continuous encoders for understanding (no information
loss), diffusion for generation (micro-optimized detail)
[74:43](ts:74:43). Persistent challenges: no universal encoder
(classification wants semantics, OCR wants pixels), modality
weighting (video is low-density, do not let it drown text), and
systems (video loading alone can bottleneck training)
[61:16](ts:61:16).

![Summary](assets/l17-multimodal-summary.svg "Encode continuously, generate with diffusion, weight carefully.")

## Recap: the whole lesson on one screen

<div class="recap-grid">
<div class="recap-card">
<img src="assets/l17-omni.svg" alt="Omni">
<div class="rc-body">
<strong>1. The omni goal</strong>
<p>Any modality in, any out. Transformers speak tokens: images must
become tokens first.</p>
<p class="rc-num">Key: tokenize everything</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-clip.svg" alt="CLIP">
<div class="rc-body">
<strong>2. CLIP</strong>
<p>2N-way contrastive on 400M pairs. ViT-L/14. Zero-shot ImageNet
beat supervised ResNet.</p>
<p class="rc-num">Key: text gives semantics</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-siglip.svg" alt="SigLIP">
<div class="rc-body">
<strong>3. SigLIP</strong>
<p>Binary sigmoid loss. Batch decoupled. Chunked rotation. 5 days
vs 10 days.</p>
<p class="rc-num">Key: simpler and parallel</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-llava.svg" alt="LLaVA">
<div class="rc-body">
<strong>4. LLaVA</strong>
<p>CLIP plus projector plus Vicuna. Align W, then fine-tune.
158k synthesized conversations.</p>
<p class="rc-num">Key: stitch, do not rebuild</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-anyres.svg" alt="AnyRes">
<div class="rc-body">
<strong>5. AnyRes</strong>
<p>Crop instead of downsample. Overview plus detail crops.
Modalities transfer.</p>
<p class="rc-num">Key: resolution matters</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-qwen.svg" alt="Qwen">
<div class="rc-body">
<strong>6. Qwen-VL lineage</strong>
<p>Dynamic resolution, M-RoPE, interleaved frequencies,
timestamps, DeepStack fusion.</p>
<p class="rc-num">Key: sharper pieces</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-chameleon.svg" alt="Chameleon">
<div class="rc-body">
<strong>7. Chameleon</strong>
<p>Everything discrete via VQ-VAE. Elegant, unstable,
detail-losing. Diffusion won generation.</p>
<p class="rc-num">Key: entropy fights back</p>
</div>
</div>
<div class="recap-card">
<img src="assets/l17-multimodal-summary.svg" alt="Summary">
<div class="rc-body">
<strong>8. State of play</strong>
<p>Continuous encode, diffusion generate. Weight modalities.
No universal encoder.</p>
<p class="rc-num">Key: match encoder to task</p>
</div>
</div>
</div>

## Official sources and further reading

**Official:**
- Lecture 17 video.
- CLIP, SigLIP, LLaVA, LLaVA-OneVision, Qwen-VL series, Chameleon papers.

**Further reading:**
- LAION-5B / OpenCLIP (open replication).
- VQ-VAE (Oord 2017). DeepStack adapter paper.
- AI2's VLM report (data details the lecture skipped).

**Caveats from these sources.** Frontier omni models are
undisclosed. The "continuous plus diffusion" picture is the
lecturer's speculation, flagged as such. Compute comparisons
(TPUv3 vs v4) are not flops-normalized. Chameleon results were not
shown in lecture.

## Connections to the other courses

- **CS336 Lecture 1:** tokenization generalized beyond text.
- **CS336 Lecture 3:** RoPE generalized to M-RoPE.
- **CS224N:** vision-language models from the NLP side.

> [!CHEAT]
> **Multimodality cheatsheet.** Omni: any in, any out. Tokenize everything. CLIP: 2N contrastive, 400M pairs, ViT-L/14, zero-shot beats ResNet. Text gives semantics. SigLIP: binary loss, batch decoupled, chunked rotation, 5 vs 10 days. LLaVA: CLIP + W + Vicuna, align then fine-tune, 158k synthetic. Projector = space alignment. AnyRes: crops not downsampling, overview + details, modality transfer. Qwen-VL: cross-attention, 1.4B stage 1. Qwen2: dynamic res, 2x2 compression, M-RoPE. Qwen3: SigLIP-2, interleaved frequencies, timestamps, sqrt loss, DeepStack, 256K context. Chameleon: VQ-VAE discrete, 1024 tokens per image, unstable (entropy), loses detail. State: continuous encode, diffusion generate, weight modalities.

> [!MEMORY]
> **Stitch, do not rebuild.** Reuse encoders and LLMs. The projector aligns spaces. Resolution and weighting decide quality.
