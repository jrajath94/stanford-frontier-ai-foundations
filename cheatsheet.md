# CS229S cheatsheet

Date: 2026-10-06. All 10 units. Toy values from the
lessons. "toy" marks computed examples, not specs.

## Shapes

| Symbol | Shape |
|---|---|
| X | (B, T, n) |
| Q, K, V | (B, h, T, d) |
| W_q,k,v,o | (n, n) |
| W_1 / W_2 | (n, d_ff) / (d_ff, n) |
| scores | (B, h, T, T) |

## Cost formulas

| What | Formula | Toy |
|---|---|---|
| Train FLOPs | 6 N D | 1.2e20 (1B, 20B tok) |
| Decode/token | 2 N + KV traffic | - |
| Attention scores | 2 T^2 d / head | 17.2 TFLOPs (8k, 128) |
| Score memory | T^2 floats | 256 MiB fp32 (8k) |
| KV/token | 2 L n bytes/float | 0.5 MiB (32, 4096, fp16) |
| KV total | 2 L T n bytes/float | - |
| Linear attn | 2 T d^2 | 0.268 GFLOPs (64x less) |
| SSM state | d^2 floats | 64 KiB |
| FFT conv | 3 n log2 n | 33x at n=1024 |
| Ring all-reduce | 2(n-1)/n S/BW | 0.35 s (20 GB, 8, 100 GB/s) |
| Bubble | (p-1)/(m+p-1) | 27.3% (4, 8) |
| ZeRO-3/GPU | 16 P / n bytes | 20.0 GB (10B, 8) |
| Checkpoint | (N/k) c | 300 s (1000, 100, 30 s) |
| MTBF cluster | per-GPU / n | 4.3 days (256) |
| MFU | achieved / peak | 48.1% (150/312) |

## Quantization

s = (max-min)/(2^b-1). Z = round(-min/s).
x_q = clamp(round(x/s)+z, 0, 2^b-1).
x_hat = s(x_q-z). In-range error <= s/2.

## Adaptation

Scaling: log10 L = a + b log10 N (toy b =
-0.0748/decade). SFT: loss on response tokens
only. RLHF: E[r] - beta KL. LoRA: W0 + BA,
r(d+k) params (128x fewer at 4096/16).

## Serving

Step = max(compute, fetch+assemble). P99 over
means. Capacity = tokens*k/experts*factor.
Dispatch = tokens*k*d bytes one way. R(B) =
B/(c0+c1 B). $/MTok = 1e6/R * $/hr / 3600.

## Retrieval

Amortized = B/N + q. RRF = sum 1/(k+rank).
Cascade = sum count*cost. Gate: delta >=
-epsilon.

## Scheduling

DRF share = max(demand/total). Straggler tax =
(max-median)/median. Makespan via FIFO sim.

## Evidence grades

measured > derived > stated > opinion. Never
promote without the artifact.

## Defense five

claim, evidence, baseline, failure, cost.
