# Glossary , cs336 U01-U18

One meaning per term. The lesson that introduces a term owns the full
definition, this glossary points there.

## U01 , Tokenization

- Code point: a number assigned to a character by Unicode (U+0041 = "A").
- UTF-8: a variable-length encoding of code points as 1 to 4 bytes.
- Token: an integer id from a fixed vocabulary used as model input.
- BPE (byte pair encoding): a tokenizer that starts from bytes and merges
  the most frequent adjacent pair, repeating to a target vocabulary size.
- Merge: one BPE step that joins a chosen pair into a new token.
- Pretokenization: splitting raw text (often by regex) before BPE merges.
- Special token: a token with control meaning (for example end of text),
  never produced by normal merges.
- Roundtrip: decode(encode(text)) equals text byte-for-byte.
- Fertility: tokens produced per word, lower means better compression.

## U02 , Tensor programming

- Shape contract: the promised input and output shapes of an operation.
- View: a tensor that shares storage with another tensor.
- Stride: the step in storage between consecutive elements along an axis.
- Contiguous: storage order matches logical order (stride pattern standard).
- Activation: a tensor saved during the forward pass for the backward pass.
- Arithmetic intensity: FLOPs per byte moved between memory levels.
- Roofline: a plot of attainable FLOP/s against arithmetic intensity.
- Checkpointing (activation): recompute activations in backward to save memory.

## U03 , Transformer block

- Embedding: a learned vector per token id, lookup, not computation.
- Query/Key/Value (Q/K/V): linear projections of the input, attention
  compares queries against keys and mixes values.
- Causal mask: a mask that blocks attention to future positions.
- Head: one independent attention unit, heads concatenate their outputs.
- Residual connection: x + block(x), the identity path that carries
  gradient.
- Pre-norm: normalization before the sub-block, post-norm: after.
- RMSNorm: normalization by root mean square, without mean centering.
- SwiGLU: a gated feedforward activation: Swish(xW) elementwise-times (xV).
- RoPE: rotary position embedding, rotates Q/K pairs by position angle.
- Weight tying: the output head reuses the input embedding matrix.

## U04 , Attention alternatives and MoE

- MHA: multi-head attention, separate K/V per head.
- MQA: multi-query attention, one K/V head shared by all query heads.
- GQA: grouped-query attention, groups of query heads share K/V heads.
- MLA: multi-head latent attention, K/V compressed to a latent vector.
- Linear attention: attention rewritten with a kernel so cost grows
  linearly in sequence length.
- MoE: mixture of experts, a router sends each token to k of E experts.
- Router: the gating network that scores experts per token.
- Dropped token: a token an expert skips when its capacity is full.
- Load balancing: auxiliary losses that spread tokens across experts.

## U05 , Optimization

- Cross-entropy loss: negative log probability of the true next token.
- Label shift: targets are inputs shifted by one position.
- Adam: adaptive optimizer with first and second moment estimates.
- AdamW: Adam with decoupled weight decay.
- Bias correction: Adam's division by (1 - beta^t) that fixes early steps.
- Warmup: learning rate rises from 0 over the first steps.
- Cosine decay: learning rate follows a cosine curve down to a floor.
- Gradient clipping: rescale the gradient when its norm exceeds a limit.
- RNG state: the random generator state saved for exact resume.

## U06 , Hardware

- HBM: high-bandwidth memory on the GPU package.
- SRAM: fast on-chip memory (shared memory, registers).
- Tensor core: a matrix-multiply unit on NVIDIA GPUs.
- Occupancy: fraction of the GPU's warp slots that hold live warps.
- Coalescing: adjacent threads read adjacent memory addresses.
- Bank conflict: shared-memory accesses that serialize on the same bank.
- Launch overhead: fixed CPU cost of starting a GPU kernel.

## U07 , Kernels

- Fusion: one kernel does the work of several, keeping data on chip.
- Tiling: split a large operation into SRAM-sized blocks.
- Online softmax: softmax computed in one pass with a running maximum.
- FlashAttention: IO-aware exact attention using tiling and online softmax.
- Triton: a Python-like language for writing GPU kernels.
- Grid: the set of thread blocks (Triton) or blocks (CUDA) of a kernel.
- Register pressure: too many live values per thread lowers occupancy.

## U08 , Sharding

- DDP: distributed data parallel, each rank holds a full replica.
- All-reduce: every rank ends with the sum of all ranks' tensors.
- Reduce-scatter: every rank ends with one shard of the sum.
- All-gather: every rank collects all shards into the full tensor.
- ZeRO: sharding of optimizer state, gradients, and parameters.
- Microbatch: one slice of a batch processed before the optimizer step.
- Bucket: a gradient buffer chunk synchronized as one collective.

## U09 , Parallelism

- Tensor parallelism: split individual layers across ranks.
- Pipeline parallelism: split layers into stages across ranks.
- Pipeline bubble: idle time while the pipeline fills and drains.
- Sequence parallelism: split the sequence dimension across ranks.
- Expert parallelism: place different experts on different ranks.
- Hybrid layout: a named combination such as DP + TP + PP.
- Straggler: a rank that lags and slows a synchronized step.

## U10 , Scaling

- Scaling law: an empirical relation between compute, data, parameters,
  and loss, usually a power law with an irreducible floor.
- IsoFLOP curve: loss versus allocation at fixed compute budget.
- Compute-optimal: the parameter/data split that minimizes loss at a
  fixed compute.
- Preregistration: a written prediction with a band and a decision rule,
  made before the experiment runs.
- Bootstrap interval: an uncertainty band from resampling the data.

## U11 , Inference

- Prefill: the first pass over the prompt, compute-bound at long lengths.
- Decode: token-by-token generation, bandwidth-bound.
- KV cache: stored keys and values that avoid recomputation.
- Continuous batching: inserting and evicting sequences at iteration
  boundaries instead of waiting for a whole batch.
- Speculative decoding: a small model drafts, the target verifies.
- TTFT: time to first token. TPOT: time per output token.

## U12 , Evaluation

- Perplexity: e raised to the cross-entropy loss.
- Contamination: benchmark text present in training data.
- Canary: a planted unique string that detects contamination.
- Internal evaluation: steers iteration. External evaluation: judges.
- Kappa: agreement between judges corrected for chance.
- Slice: a subset of the eval that isolates one capability.

## U13 , Data sourcing

- WARC: the web archive record format. CDX indexes it.
- Extraction yield: kept bytes divided by raw bytes.
- Provenance: the URL, crawl date, and license of a document.
- PII: personally identifiable information, detected and redacted.
- Version hash: a content digest that identifies a dataset build.

## U14 , Dedup and mixing

- Bloom filter: a bit array with k hashes, no false negatives.
- Shingle: an n-gram used as a set element for near-dup detection.
- MinHash: a signature that estimates Jaccard similarity.
- LSH: locality-sensitive hashing that finds candidate pairs.
- Reweighting: scaling a source by target weight over source weight.
- Mixture search: choosing domain proportions for a target metric.

## U15 , Midtraining and SFT

- Midtraining: continued pretraining with a new data mix.
- Loss mask: the 0/1 rule that puts loss on assistant tokens only.
- Packing: concatenating short sequences into full context windows.
- LoRA: a low-rank adapter added to frozen weights.
- Forgetting: base capability lost during adaptation, measured in nats.

## U16 , Alignment and RL

- Reward model: a learned scorer of response quality.
- Bradley-Terry: the pairwise preference probability model.
- KL regularization: a penalty that keeps the policy near a reference.
- PPO: policy optimization with a clipped surrogate objective.
- GRPO: group-relative baselines without a value network.
- DPO: direct preference optimization without RL.
- Reward hacking: the proxy rising while the truth falls.

## U17 , Systems

- Modality token: non-text input (image patches) as tokens.
- Interleaved stream: one token sequence mixing modalities.
- Rollout fleet: workers that generate trajectories, separate from
  training.
- Lineage: the hash-chained record of how an artifact was produced.
- Bottleneck: the slowest stage of a step, named by profiling.
- Amdahl bound: the speedup limit from the unoptimized fraction.

## U18 , Defense

- Honesty protocol: name the source, state the inspected extent, state
  the unknown.
- Ablation: removing one factor to measure its effect.
- Failure diary: date, symptom, hypothesis, fix, lesson.
- Resource ledger: the compute and storage cost record.
- Replication: an independent rerun inside all seed bands.
- Oral defense ladder: define, toy, derive, implement, compare, debug,
  critique, design.
