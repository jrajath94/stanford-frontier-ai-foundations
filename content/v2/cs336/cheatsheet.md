# cs336 cheatsheet , formulas and decision rules

One page per unit, formulas first. Toy numbers from executed
course scripts. HYP = hypothetical.

## U01 tokenization

- UTF-8: code points < U+0080 one byte, longer need 2-4.
- bytes/token: English ~4.2, code ~2.5 (toy).
- Rule: the training and inference tokenizers must match.

## U02 tensor programming

- matmul: 2*m*n*k FLOPs, backward ~2x forward.
- Adam memory: params + grads + 2 moments (+ master weights).
- Rule: intensity = FLOPs/byte decides bound or bandwidth.

## U03 transformer

- Attention: softmax(QK^T/sqrt(d))V, causal mask = -inf upper.
- SwiGLU: x * silu(W1 x) * (W2 x) gating.
- Rule: pre-norm for deep stacks.

## U04 attention/MoE

- KV per token: 2*L*h_kv*d_h*bytes.
- MoE active params: top-k experts, not all.
- Rule: balance the router or experts collapse.

## U05 optimization

- AdamW: decay the weights, not the gradients.
- Clip: global norm <= c.
- Rule: checkpoint weights + optimizer + RNG + data position.

## U06 hardware

- Roofline: min(peak FLOPs, bandwidth * intensity).
- MFU: achieved / peak.
- Rule: measure first, optimize second.

## U07 kernels

- SRAM >> HBM speed, tile to fit.
- Online softmax: running max + running sum.
- Rule: correctness vs reference before benchmarks.

## U08 sharding

- ZeRO-1: optimizer. -2: +gradients. -3: +parameters.
- Rule: overlap comms with compute (bucketing).

## U09 model parallel

- TP: split layers. PP: split depth, bubble ~ (p-1)/(m+p-1).
- 1F1B shrinks the bubble.
- Rule: hybrid DP/TP/PP for large runs.

## U10 scaling

- L = E + A*(C/C0)^{-a}. C = 6ND.
- Toy: a=0.34 -> 0.340. N=7.0e10, D=1.4e12, D/N=20.
- Rule: preregister extrapolations with bands.

## U11 inference

- KV: 2*L*h_kv*d_h*bytes/token (toy 128 KB. 8.59 GB).
- Spec decode speedup = E/(k*c+1) (toy 1.95x).
- Rule: serve to SLOs (TTFT, TPOT, p99), not to throughput.

## U12 evaluation

- ppl = e^loss (toy e^2.3 = 9.97).
- SE = sqrt(p(1-p)/n) (toy 0.72 +- 0.039).
- kappa = (agree - expected)/(1 - expected).
- Rule: never tune on the external eval.

## U13 data sourcing

- yield = kept/raw (toy 28.6%).
- Provenance: URL + date + license.
- Rule: sort, seed, pin, hash, prove determinism.

## U14 dedup/mixing

- Bloom FPR: (1-e^{-kn/m})^k (toy 0.0216, k=6).
- LSH: P = 1-(1-s^r)^b.
- Rule: dedup before filtering, dedup train vs eval.

## U15 midtrain/SFT

- Mask: loss on assistant tokens only.
- LoRA: 2*d*r per pair (toy 8.39M).
- Packing: 47.6% -> 22.0% waste (toy).
- Rule: templates byte-identical train/serve.

## U16 alignment/RL

- BT: sigmoid(r_w - r_l).
- PPO: min(rA, clip(r,1-eps,1+eps)A).
- GRPO: (r - mean)/std per group.
- DPO: r = beta*log(pi/pi_ref) (toy margin 0.139).
- Rule: watch the truth metric, not the proxy.

## U17 multimodal/systems

- Image: 256 tokens (16x16 patches).
- Amdahl: new = total - share + share/speedup (toy 143 ms).
- Rule: profile first, the bottleneck moves.

## U18 defense

- Replication: inside all seed bands.
- Ablations: 2^n / 2^{n-k} / 1+n.
- Rule: never infer content from a speaker's name.
- Ladder: define, toy, derive, implement, compare, debug,
  critique, design.
