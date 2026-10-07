# Notation and shapes , cs336 U01-U09

Shared symbols used across all nine units. A lesson may add local symbols,
local symbols never override these.

## Index conventions

- B: batch size (sequences per batch). Unit: count.
- T (or S): sequence length in tokens. Unit: count.
- d (or d_model): model width. Unit: count.
- h: number of attention heads. Unit: count.
- d_h: head dimension, d_h = d / h. Unit: count.
- V: vocabulary size. Unit: count.
- L: number of transformer layers. Unit: count.
- d_ff: feedforward hidden width. Unit: count.
- E: number of experts (MoE). Unit: count.
- k: top-k experts per token (MoE). Unit: count.
- N: parameter count. Unit: count.
- D: training token count. Unit: count.
- C: training compute in FLOP. Unit: FLOP.

## Tensor shapes

- Token ids: x with shape (B, T), dtype int64, values in [0, V).
- Embeddings: E_tok with shape (V, d), lookup gives (B, T, d).
- Q, K, V per head: (B, h, T, d_h).
- Attention scores: (B, h, T, T) before masking.
- Attention weights: same shape, rows sum to 1 over keys.
- Logits: (B, T, V).
- KV cache per layer per token: 2 * d values (K and V), bytes = 2 * d *
  bytes_per_element.

## Probability and loss

- p: true distribution. q (or p_theta): model distribution.
- H(p, q): cross-entropy. Unit: nats (natural log) or bits (log base 2),
  the lesson states the base.
- Perplexity: exp(H) in nats. Unit: none (ratio scale).
- KL(p || q): Kullback-Leibler divergence. Unit: nats or bits.

## Optimization

- theta: parameters. g: gradient of loss wrt theta.
- eta (or lr): learning rate. Unit: 1 / (loss curvature scale).
- t: optimizer step index. m_t, v_t: Adam first and second moments.
- beta_1, beta_2: Adam decay rates. eps: numerical guard.
- lambda: weight decay coefficient.

## Hardware and systems

- FLOP: one floating-point operation. FLOP/s: rate.
- Byte counts use decimal GB (1e9) unless a lesson states GiB.
- Arithmetic intensity: FLOP per byte moved.
- Ranks: r = 0..R-1 for R data-parallel workers.
- Microbatch count: m, gradient accumulation steps: a.

## Text and tokenization

- U+XXXX: Unicode code point in hex.
- UTF-8 bytes: sequences of 1 to 4 bytes per code point.
- Token: integer id in [0, V). A token may decode to bytes, not to a full
  character.
- Compression rate: bytes per token, or tokens per word.

## Conventions

- Shapes are written (B, T, d) in that order unless stated.
- Log means natural log unless the base is stated.
- "Not in source" marks any value not computed by a lesson script or
  quoted from a cited paper.
