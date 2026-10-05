import sys
sys.path.insert(0, "/home/hatch/workspace/stanford-frontier-ai/content/v2/cs336/scratch")
from plates import *

# ---- omni ----
p = Plate(960, 540); p.defs_arrow()
y = p.title("The omni model", "Any modality in, any modality out. The North Star.")
p.panel(60, 140, 840, 130, "the goal")
p.text(84, 188, "text, image, audio, video: any combination in, any combination out", 15, INK, 600)
p.text(84, 214, "transformers speak tokens, so everything must become tokens", 14, MUT)
p.panel(60, 296, 840, 130, "the question")
p.text(84, 344, "a pixel is not a semantic unit. What is the image BPE?", 15, ORANGE, 700)
p.text(84, 370, "discrete or continuous tokens: the two answers this lecture", 14, MUT)
p.footer("Source: lecture intro slides, original plate. Shell 1: the North Star.")
p.save("l17-omni.svg")

# ---- CLIP ----
p = Plate(960, 600); p.defs_arrow()
y = p.title("CLIP: semantics from captions", "Contrastive learning on 400M image-text pairs.")
p.panel(60, 140, 840, 140, "the objective")
p.text(84, 188, "2N softmax: I1 closer to T1 than to all other texts, and vice versa", 15, INK, 600)
p.text(84, 214, "text gives high-level semantics augmentation never could", 14, MUT)
p.panel(60, 306, 840, 150, "the machinery")
p.text(84, 354, "ViT-L/14: 14x14 patches, positional embeddings, attention pooling", 15, INK, 600)
p.text(84, 380, "text: GPT-2 style, EOS activation as the sequence vector", 14, MUT)
p.text(60, 480, "Zero-shot ImageNet beat a ResNet trained on 1.2M labeled images.", 15, TEAL, 700)
p.footer("Source: lecture CLIP slides, original plate. Shell 2: the foundation.")
p.save("l17-clip.svg")

# ---- SigLIP ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("SigLIP: simpler and parallel", "Binary sigmoid loss. Batch size decoupled from the loss.")
p.panel(60, 140, 840, 130, "the loss")
p.text(84, 188, "diagonal = positive, off-diagonal = negative. log_sigmoid.", 15, INK, 600)
p.text(84, 214, "same loss at any batch size: small batches work, 32K is the ceiling", 14, MUT)
p.panel(60, 296, 840, 130, "the systems win")
p.text(84, 344, "chunked rotation: each device computes local losses, rotates text", 15, INK, 600)
p.text(84, 370, "5 days on 32 TPUv4 vs CLIP's 10 days on 256 TPUv3", 14, TEAL, 700)
p.footer("Source: lecture SigLIP slides, original plate. Shell 2: the improvement.")
p.save("l17-siglip.svg")

# ---- LLaVA ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("LLaVA: stitch a VLM together", "CLIP encoder + projector + Vicuna. Two stages.")
p.panel(60, 140, 840, 150, "the architecture")
p.text(84, 188, "image -> CLIP -> matrix W -> text embedding space -> transformer", 15, INK, 600)
p.text(84, 214, "158k GPT-4-synthesized COCO conversations as training data", 14, MUT)
p.panel(60, 316, 840, 140, "the training")
p.text(84, 364, "stage 1: freeze everything, train only W (alignment)", 15, INK, 600)
p.text(84, 390, "stage 2: freeze vision, train W + language model", 14, MUT)
p.footer("Source: lecture LLaVA slides, original plate. Shell 3: the template.")
p.save("l17-llava.svg")

# ---- AnyRes ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("AnyRes: read at any resolution", "336x336 cannot read a document. Crop instead of downsampling.")
p.panel(60, 140, 840, 150, "the idea")
p.text(84, 188, "split the image into encoder-sized crops, encode each, concatenate", 15, INK, 600)
p.text(84, 214, "one downsampled overview + up to 9 detail crops. Downsample if too many.", 14, MUT)
p.panel(60, 316, 840, 140, "the payoff")
p.text(84, 364, "transfer across modalities: single-image OCR + multi-image reasoning combine", 15, TEAL, 700)
p.text(84, 390, "video = frames with fewer tokens each (up to 32 frames)", 14, MUT)
p.footer("Source: lecture OneVision slides, original plate. Shell 3: resolution.")
p.save("l17-anyres.svg")

# ---- Qwen ----
p = Plate(960, 620); p.defs_arrow()
y = p.title("Qwen-VL: three generations", "Same template, sharper pieces each time.")
rows = [
    ("Qwen-VL", "OpenCLIP + cross-attention adapter. 1.4B examples stage 1.", MUT),
    ("Qwen2-VL", "dynamic resolution, 2x2 token compression, M-RoPE (h, w, t)", BLUE),
    ("Qwen3-VL", "SigLIP-2, interleaved M-RoPE, explicit timestamps, DeepStack fusion", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 110
    p.panel(60, yy, 840, 94, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="94" rx="5" fill="{col}"/>')
    p.text(84, yy + 64, b, 14, MUT)
p.text(60, 490, "Context to 256K for long video. sqrt-normalized loss so video does not dominate.", 14, INK, 700)
p.footer("Source: lecture Qwen slides, original plate. Shell 4: the lineage.")
p.save("l17-qwen.svg")

# ---- Chameleon ----
p = Plate(960, 580); p.defs_arrow()
y = p.title("Chameleon: everything is a token", "VQ-VAE discretizes images. One model, text and images interleaved.")
p.panel(60, 140, 840, 140, "the machinery")
p.text(84, 188, "512x512 image -> 1,024 tokens from an 8,000-code codebook", 15, INK, 600)
p.text(84, 214, "then: plain language-model training. No adapter, no encoder.", 14, MUT)
p.panel(60, 306, 840, 150, "the problems")
p.text(84, 354, "image tokens are high-entropy: norms grow, training destabilizes", 15, ORANGE, 700)
p.text(84, 380, "discretization loses fine detail (OCR). QK-norm + z-loss mitigate.", 14, MUT)
p.footer("Source: lecture Chameleon slides, original plate. Shell 4: the elegant alternative.")
p.save("l17-chameleon.svg")

# ---- summary ----
p = Plate(960, 560); p.defs_arrow()
y = p.title("Multimodality: the state of play", "Continuous encoders understand. Diffusion generates.")
rows = [
    ("understanding", "continuous encoders (CLIP/SigLIP): no information lost", TEAL),
    ("generation", "diffusion heads: micro-optimize fine detail", BLUE),
    ("weighting", "video is low-density: do not let it drown the text", ACTIVE),
]
y0 = 140
for i, (a, b, col) in enumerate(rows):
    yy = y0 + i * 100
    p.panel(60, yy, 840, 84, a)
    p.parts.append(f'<rect x="60" y="{yy}" width="10" height="84" rx="5" fill="{col}"/>')
    p.text(84, yy + 56, b, 14, MUT)
p.text(60, 470, "No universal encoder: classification wants semantics, OCR wants pixels.", 14, MUT)
p.footer("Source: lecture closing slides, original plate. Shell 5: the state of play.")
p.save("l17-multimodal-summary.svg")
