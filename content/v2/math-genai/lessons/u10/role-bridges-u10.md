# Role bridges, U10 LLM inference, quantization, and alignment

Date: 2026-10-06. Each bridge is labeled by role and
anchored to a U10 section. Residual gaps are stated
per role: the course alone does not make the learner
hireable. Interview provenance for the bridge
questions: role-derived practice, not employer
material.

## Research

Bridge: the alignment assumptions are the research
surface. The toy quantifies the DPO pathology
(margin 0.07 with implicit rewards 0.0400/-0.0300:
watch the absolute log-ratios, not just the loss)
and the reward-hacking mechanism (proxy 1.20 vs
truth 0.70). The open questions: the win-rate vs
KL frontier (C09 shell 9), the BT-vs-human
correlation breakdown (C11 extension), and the
state-space alternative (W12L54, untaught).

Practice questions: audit DPO's three
assumptions on a new dataset. Design the beta
sweep with the knee criterion. Name the evidence
that would promote W12L53 from title-level to
source-confirmed.

Residual gap: no real-data experiments, no
literature review, no novel claim. The toys do
not transfer to papers.

## Research engineering

Bridge: the numerics are the job. The byte
formulas (KV 2Lhdn·bpe, quant s/zp with the s/2
bound), the decode FLOP ratio (= n), the loss
implementations (BT, PPO clip with the min,
DPO), and the top-p sort-direction assert are
the habits. The canonical bug classes: the
forgotten factor 2, z/0 instead of argmax,
ascending sort in top-p, max instead of min in
PPO.

Practice questions: implement all six from
memory. Reproduce every number via
compute_run5b.py. Write each bug from memory,
then fix it.

Residual gap: no GPU kernels, no real
quantization library, no distributed serving.
The toys are numpy, not systems.

## FDE (forward deployed engineer)

Bridge: the serving arithmetic is the deployable
core: 12.00 MiB cache at n = 512, 24576
bytes/token, the 142.9 -> 571.4 tok/s roofline
(authored), the (T, top-p) product knobs. The
discovery questions: what latency, what
context, what quality floor? The handoff
artifact: the quantized model + cache config +
(T, top-p) + the eval that passed.

Practice questions: price a deploy (bytes,
tok/s ceiling, knobs). Name the quality
evidence required. Write the version tuple.

Residual gap: no real serving stack, no client
contract, no cost model beyond arithmetic. The
roofline is a ceiling, not a measurement.

## ML

Bridge: the eval discipline is the ML surface.
The win-rate 0.5834 is a model-selection
number, the hacking demo (1.20 vs 0.70) is why
the proxy is never the eval, the audit (4/7/1)
is why claims need denominators. Eval rule:
held-out pairs, held-out judge, humans for
ship.

Practice questions: design the eval suite for
an aligned model. Explain the self-grading
trap. Reproduce the audit denominators.

Residual gap: no real eval suite, no human
study, no benchmark suite. The toy eval is not
an eval.

## LLM

Bridge: strong fit. This unit is LLM
engineering: decoding (C01-C02), shrinking
(C03-C04), aligning (C05-C09), and not
fooling yourself (C10-C12). The through-line:
the training objective is never the product
metric, the knobs split into frozen (weights,
beta) and tunable (T, top-p), the eval is
held-out or it is nothing.

Practice questions: map each section to the LLM
lifecycle. Price a full deploy from the
formulas. Run the audit on a new course.

Residual gap: no real training run, no real
deploy, no production incident. The unit is
foundations with honest numbers, not
operations.

## MLOps

Bridge: strong fit. The versioned artifacts:
the quantized model, the (T, top-p) config,
the eval suite, the audit table. The gates:
the byte budget, the tok/s ceiling, the
held-out win-rate, the audit denominators.
The monitoring: proxy vs truth (C10), KL
trace (C09), absolute log-ratios (C08).

Practice questions: define the artifact set
and its versions. Write the deploy gate
checklist. Design the proxy-vs-truth
monitor.

Residual gap: no real registry, no CI/CD, no
incident drill. The gates are specified, not
built.

## Agents

Bridge: partial fit, stated openly. Aligned
LLMs are the substrate agents run on: the
decoding knobs shape agent outputs, the KL
leash bounds drift, the reward-hacking lesson
applies directly (an agent optimizes its
reward signal too). What does not transfer:
no tools, no state, no stopping, no
permissions, no multi-step credit assignment
beyond the response level.

Practice questions: explain how a mispriced KL
(beta too small) breaks an agent's
reliability. Map the hacking demo to an agent
reward signal. Name the agent pieces missing
here.

Residual gap: no tool use, no memory, no
recovery, no agent eval. The aligned model is
necessary, not sufficient.
