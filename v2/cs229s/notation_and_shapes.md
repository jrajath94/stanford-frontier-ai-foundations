# notation_and_shapes.md: cs229s shared notation

Date: 2026-10-06. Every lesson defines symbols before use. This
file fixes the shared conventions so units reuse identical
symbols.

## Model shape symbols

| Symbol | Meaning |
|---|---|
| `n` or `d_model` | model width (residual stream size) |
| `L` | number of transformer layers |
| `h` | number of attention heads |
| `d` or `d_head` | per-head dimension, `d = n / h` |
| `T` or `S` | sequence length (context length) |
| `V` | vocabulary size |
| `d_ff` | feedforward hidden size (often `4n`) |
| `B` | batch size |
| `P` | total parameter count |

## Tensors

| Symbol | Shape | Meaning |
|---|---|---|
| `X` | `(B, T, n)` | layer input activations |
| `Q, K, V` | `(B, h, T, d)` | query, key, value per head |
| `W_q, W_k, W_v` | `(n, n)` | attention projections (all heads) |
| `W_o` | `(n, n)` | attention output projection |
| `W_1` | `(n, d_ff)` | MLP up projection |
| `W_2` | `(d_ff, n)` | MLP down projection |
| `A` | `(B, h, T, T)` | attention score matrix (materialized in naive attention) |
| `logits` | `(B, T, V)` | output logits |

## Hardware symbols

| Symbol | Meaning |
|---|---|
| `F` | FLOPs (operation count) |
| `Q_b` | bytes moved |
| `I = F / Q_b` | arithmetic intensity (FLOP/byte) |
| `pi` | peak compute (FLOP/s) |
| `beta` | peak memory bandwidth (byte/s) |
| `I_star = pi / beta` | ridge point (FLOP/byte) |

## Quantization symbols

| Symbol | Meaning |
|---|---|
| `s` | scale (step size) |
| `z` | zero point |
| `x_q` | quantized integer |
| `x_hat` | dequantized approximation |

## Conventions

- Counts use powers of two for memory (`2^30` = 1 GiB) and
  powers of ten for rates unless stated.
- FLOPs count multiply-add as 2 operations.
- fp16/bf16 = 2 bytes, fp32 = 4 bytes, int8 = 1 byte, fp8 = 1
  byte.
- A claim with "Not in source" means the value is a toy
  computed in the lesson, not a vendor spec.
