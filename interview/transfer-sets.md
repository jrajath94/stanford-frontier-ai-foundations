# Transfer sets: 10 changed-scenario sets

Date: 2026-10-06. Cross-unit scenarios with changed
constraints. Keys in `keys-transfer.md`. Provenance:
original practice.

## T1. Million-token context

Scenario: a 1M-token document must be summarized.
Softmax attention needs 2 TB of fp16 scores per head
(4 TB fp32). The team proposes a pure SSM.

Q1. Compute the score memory for T=1M, one head,
fp16. Then compute the SSM state for d=128.
Q2. What capability does the pure SSM risk losing
(U07 C10)?
Q3. Propose a hybrid that keeps the capability.
What does it cost?
Q4. The summary needs exact quotes. Does your
hybrid deliver them?

## T2. Bursty MoE serving

Scenario: an 8-expert MoE serves chat. Arrivals
burst 10x at the top of each hour. P99 triples.

Q1. Name two mechanisms that could explain the
p99 (one from U08 serving, one from MoE).
Q2. Continuous batching is on. Why does the burst
still hurt?
Q3. The gate collapses onto 2 experts during the
burst. What happens to capacity drops and step
time?
Q4. Propose two fixes, one for the queue and one
for the gate.

## T3. Spot training across regions

Scenario: training a 10B model on spot GPUs split
across two regions (50 GB/s between, hypothetical
toy). Checkpoint every 30 min.

Q1. Which parallelism split crosses the thin
link, and what does each sync cost (S=20 GB,
n=16)?
Q2. A preemption wave hits one region. What is
lost, and what resumes?
Q3. The inter-region all-reduce dominates the
step. Restructure the splits.
Q4. Price the decision: when does the spot
discount beat the rework?

## T4. Quantized MoE serving

Scenario: the MoE from T2 must fit on half the
GPUs. The team proposes int8 weights.

Q1. Which quantization granularity survives the
expert skew (U05 C03)?
Q2. The gate logits are int8 too. What breaks?
Q3. Compute the memory saving for 8 experts of
2 GB fp16 each under int8.
Q4. State the quality gate and the first
experiment.

## T5. The emergence audit

Scenario: a vendor claims their 7B model shows
emergent tool use that their 1B lacks.

Q1. Name the threshold artifact and how to test
for it.
Q2. The vendor reports pass/fail only. What do
you ask for?
Q3. The continuous score climbs smoothly. What
is the verdict on "emergence"?
Q4. Design the plot that settles it.

## T6. Slow node in the pipeline

Scenario: 4-stage pipeline, 16 microbatches. One
node runs at half speed.

Q1. Compute the healthy bubble fraction. Then
explain why the measured idle is worse.
Q2. Is this a straggler problem or a bubble
problem?
Q3. Propose two fixes at different layers
(schedule vs placement).
Q4. The slow node holds stage 1. Why is that the
worst stage to be slow?

## T7. Hourly news index

Scenario: 1M news docs, rebuilt hourly. Queries:
10k/hour.

Q1. Compute the amortized build cost per query
(build 2 h). Does the index pay off vs 50 ms
brute force?
Q2. The rebuild dominates. Name two ways to cut
it.
Q3. Rerank 20 docs at 20 ms each. What fraction
of a 4 s SLO does retrieval consume?
Q4. Propose the serving architecture that meets
the SLO.

## T8. LoRA on 8 GPUs

Scenario: fine-tune a 70B model (140 GB fp16) on
8x40 GB GPUs with LoRA r=16.

Q1. Does the base model fit? Compute the ZeRO-3
shard per GPU including Adam states.
Q2. Compute the LoRA trainable count for one
4096x4096 matrix and the ratio vs full.
Q3. The task needs a large behavior change.
What is the risk, and how do you detect it?
Q4. After training, what is the inference cost
of the adapter?

## T9. Speculative decoding meets batching

Scenario: the team adds speculative decoding
(draft + verify) to a continuous-batching
server.

Q1. Why does verification complicate the batch?
Q2. The draft accepts 3 tokens on average. What
speedup does the target see per step?
Q3. A huge prefill arrives during a verify step.
What happens?
Q4. When would you turn speculation off?

## T10. Defending a negative result

Scenario: your capstone-style experiment
falsified its own hypothesis (like H2a). The
poster session is tomorrow.

Q1. Is the project a failure? Argue from the
rubric (U10 C08).
Q2. The skeptic asks: "Did you just implement it
wrong?" What evidence answers?
Q3. What does the negative result rule out, and
for whom is that valuable?
Q4. Write the one-paragraph poster abstract.
