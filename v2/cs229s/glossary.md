# glossary.md: cs229s terms

Date: 2026-10-06. Full course (U01-U10).

- **Arithmetic intensity**: FLOPs per byte moved. Decides
  whether a kernel is memory-bound or compute-bound.
- **Attention**: weighted averaging of values by query-key
  similarity. Cost grows with the square of sequence length
  in the naive form.
- **Autoregressive decoding**: generation of one token at a
  time, each step conditioned on all previous tokens.
- **Backpropagation**: reverse-mode differentiation that
  computes gradients by a backward pass through the
  computation graph.
- **Butterfly matrix**: product of sparse structured factors
  with a fixed pattern. Reaches all-to-all mixing in
  `O(n log n)` parameters.
- **Coalescing**: adjacent GPU threads accessing adjacent
  memory addresses, which lets hardware merge the accesses
  into few transactions.
- **Compilation/fusion**: turning a graph of operations into
  fewer kernels so intermediate tensors stay on chip.
- **Continuous batching**: serving scheduler that admits new
  requests into a running batch instead of waiting for the
  whole batch to finish. (Named for U08, defined here for
  reference.)
- **Decode**: the token-by-token phase of generation. Memory
  bandwidth bound at small batch.
- **Draft model**: small fast model that proposes tokens for
  the large model to check in speculative decoding.
- **FlashAttention**: exact attention computed in tiled
  blocks with an online softmax, so the `T x T` matrix never
  materializes in HBM.
- **FLOPs**: count of floating-point operations. FLOP/s is a
  rate. The two are different units.
- **Kernel**: a GPU function launched over many threads.
- **KV cache**: stored keys and values of past tokens so
  decode steps avoid recomputation.
- **Monarch matrix**: block-structured matrix family that
  generalizes butterfly factors with permutations.
- **Occupancy**: fraction of the GPU's warp slots that hold
  live warps.
- **Online softmax**: softmax computed in streaming blocks
  with a running maximum and normalizer, so no full row is
  needed in memory.
- **Prefill**: processing of the prompt in one parallel
  pass before decode starts. Compute bound at long prompts.
- **Pruning**: removing weights (setting to zero) by some
  importance rule.
- **Quantization**: mapping floats to fewer bits via
  scale and zero point.
- **Roofline**: model that plots attainable performance
  against arithmetic intensity, capped by bandwidth and peak
  compute.
- **Speculative decoding**: draft-then-verify generation
  that keeps the target distribution exact while accepting
  several tokens per target pass.
- **Tensor core**: matrix-multiply hardware unit on NVIDIA
  GPUs. Needs specific shapes and dtypes.
- **Tiling**: splitting a large operation into blocks sized
  for on-chip memory.
- **Warp**: 32 threads that execute in lockstep on NVIDIA
  GPUs.
- **Admission control**: the door policy for the running
  set. Excess requests wait or are rejected.
- **All-reduce**: collective that sums one buffer across
  GPUs. Ring cost 2(n-1)/n * S per GPU.
- **Backpressure**: the signal upstream to slow down when
  the server queue fills.
- **Bradley-Terry**: pairwise preference model:
  P(A beats B) = sigmoid(r_A - r_B).
- **Bubble**: idle stage time in pipeline parallelism:
  (p-1)/(m+p-1).
- **Constitutional AI**: AI critique and revision against
  written principles, replacing human labels with
  AI-applied rules.
- **Convolution view**: an SSM unrolled into a kernel
  K_k = C A^k B, computed by convolution/FFT.
- **Dominant Resource Fairness**: fair sharing by each
  user's max demand ratio across resources.
- **Emergence caveat**: a threshold metric on a smooth
  skill ramp manufactures a jump. The artifact, not the
  mind.
- **Expert capacity**: per-expert token budget:
  tokens*k/experts*factor. Overflow drops.
- **k-shot prompting**: k examples in the prompt, no
  weight update. Conditioning, not training.
- **Linear attention**: phi(Q)(phi(K)^T V): O(T d^2)
  attention without the T^2 matrix.
- **Load imbalance**: max/mean expert load. Sets MoE
  step time.
- **LoRA**: low-rank adapter W = W0 + BA. Trains r(d+k)
  params, merges free at inference.
- **MFU**: model FLOPs utilization: achieved over peak
  FLOP/s.
- **MTBF (cluster)**: per-GPU MTBF divided by GPU count.
- **Packing**: concatenating short sequences into full
  bins with a document mask. Beats padding.
- **Quality gate**: ship iff task delta >= -epsilon.
- **Recall@k**: fraction of the true top-k a retriever
  returns.
- **RRF**: reciprocal rank fusion: sum 1/(k+rank) over
  systems.
- **Scaling law**: loss as a power law of scale: linear
  in log-log space.
- **Scan**: parallel prefix with an associative combine. 
  O(n) work, O(log n) depth.
- **SFT**: supervised fine-tuning. Loss on response
  tokens only.
- **SSM**: state-space model: h = Ah+Bx, y = Ch. Linear
  recurrence with a convolution view.
- **Straggler**: a slow worker that sets the synchronous
  step time.
- **ZeRO**: sharded data parallel: stages 1/2/3 shard
  optimizer states, gradients, parameters.
