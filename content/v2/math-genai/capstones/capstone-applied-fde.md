# Applied FDE capstone: support-desk LLM pilot for a mid-size bank

Date: 2026-10-06. Course: math-genai, unit U10.
Status: practice artifact. The engagement is
hypothetical, a composite of typical FDE discovery
patterns. No real client, no real deployment, no
measured production numbers. Every planning number is
authored arithmetic and labeled as such. Lesson
numbers reused as metric concepts, not as claims
about this pilot.
Claim labels: [Definition] [Derived] [Computed: lesson
or script, date] [Authored: planning assumption]
[Course-title].

## Discovery

The client is a mid-size bank with a human support
desk. [Authored] Discovery interviews with the desk
lead, two agents, and the compliance officer surface
three facts. First, 70 percent of tickets are
password resets, balance questions, and branch hours,
answerable from a 400-page policy knowledge base.
Second, agents copy answers from that base by hand,
which sets the latency floor. Third, compliance
requires every customer-facing answer to cite the
policy page it came from, and no customer PII may
leave the bank VPC.

The FDE question is not "which model is smartest".
It is "what is the cheapest system that answers the
70 percent correctly, cites sources, and keeps PII
inside the boundary". Model sophistication is not
the business goal.

## Workflow baseline

[Authored] Current state, from discovery estimates:
median handle time 6 minutes per ticket, cost per
ticket $4.20 fully loaded, escalation rate 18
percent (tickets the first agent cannot close),
customer satisfaction 4.1 of 5. The baseline that
matters for acceptance is the escalation rate and
the citation requirement, because a wrong fast
answer is worse than a slow right one.

## Objectives

1. Cut median first-response latency under 30
seconds for the 70 percent routine slice.
2. Cut cost per resolved routine ticket under
$1.50. [Authored] Target, not a measurement.
3. Hold escalation rate at or below the 18 percent
baseline. Never trade correctness for speed.
4. Every answer carries a policy-page citation.
Zero exceptions.

## Constraints

Data: the 400-page policy base is the only
grounding source. No web retrieval. Customer PII
never leaves the VPC, never enters logs, never
enters the eval set in raw form.
Model: weights must run inside the VPC on the
bank's two on-prem GPU nodes. No external API at
serve time.
Latency: p95 under 30 seconds end to end,
including retrieval.
Change: the desk keeps its ticketing UI. The LLM
is a draft-answer panel inside it, not a new tool.

## Trust boundaries

Three boundaries, each with an owner. [Definition]
Data boundary: PII enters the retrieval index only
as salted hashes for lookup keys. raw PII stays in
the ticket store. Model boundary: the served
weights are pinned by hash. the evaluation rig
rejects any weight file whose hash differs from
the accepted one. Tool boundary: the model may
call exactly two tools, policy-search and
ticket-escalate. No shell, no browser, no code
execution. A prompt-injection attempt that asks
for a third tool is denied at the tool router,
not by the model.

## Alternatives

RAG over the policy base with a mid-size
instruction model, served in INT8. Versus full
fine-tuning on ticket history. Versus an external
chat API. [Authored] Selection: RAG wins because
the citation requirement needs retrieval anyway,
fine-tuning on PII-heavy tickets raises data
handling risk, and the external API violates the
VPC constraint. The course concept behind the
choice is U10-C03 (quantization: INT8 cuts weight
memory 32 to 8 bytes per weight with error bounded
by s/2) [Computed: lesson U10, 2026-10-06] and
U10-C02 (the KV cache grows linearly in context
length, 12.00 MiB at n = 512 on the lesson toy)
[Computed: lesson U10, 2026-10-06]. Long policy
contexts make the cache the memory term to watch.

Alignment approach: SFT on 2,000 agent-written
exemplars, then DPO on 5,000 preference pairs from
the desk lead. [Authored] Sizes are estimates for planning. The course concepts are U10-C05 (SFT),
U10-C06 (Bradley-Terry preference modeling),
U10-C08 (DPO), and U10-C09 (the KL term prices
drift from the reference policy). The lesson toy
prices a KL of 0.0995 bits on its toy. here the KL
coefficient is a tuning knob with a gate on it,
not a copied number.

## Acceptance gates

The pilot does not advance unless all gates pass
on a held-out ticket set sampled after the
training cutoff.

Gate 1, correctness: citation precision at least
95 percent (the cited page supports the answer, as
judged by two agents). Gate 2, escalation:
escalation rate at or below 18 percent.
Gate 3, latency: p95 under 30 seconds.
Gate 4, safety: zero PII leaks in 500 adversarial
probes, zero tool-router violations.
Gate 5, reward hacking: a held-out human
preference check. [Course-title] U10-C10 teaches
the failure mode: the proxy can win (1.20) while
the truth loses (0.70) on the lesson toy.
[Computed: lesson U10, 2026-10-06] So the gate
measures human preference, never the reward
model's own score.
Gate 6, drift: the KL from the reference policy
stays within the tuned band. A runaway KL fails
the gate even if preference scores look good.

## Cost and latency assumptions

[Authored] All arithmetic below is for planning,
labeled per the course precedent for authored
numbers. None of it was measured.

Per-ticket compute: assume 1,500 input tokens
(policy chunks plus ticket) and 150 output
tokens. At the lesson's authored roofline ratio
(bandwidth-bound decode, the 142.9 tok/s figure
from U10-C04) [Authored: lesson arithmetic,
2026-10-06], decode of 150 tokens takes about 1
second of GPU time, prefill of 1,500 tokens a few
seconds. End-to-end p95 target 30 seconds leaves
headroom for retrieval and the ticketing UI.

Memory: the lesson toy holds 12.00 MiB of KV
cache at n = 512, or 24 KiB per token.
[Computed: lesson U10, 2026-10-06] At n = 2,048
that scales to 48 MiB per sequence, times the
concurrent batch. [Derived] This linear term, not
the weights, sets the max batch on the two nodes.

Cost: [Authored] at $1.10 per GPU hour fully
loaded and 5 seconds of GPU time per ticket, the
compute cost is under $0.01 per ticket. The $1.50
target is dominated by the human-in-the-loop
review during the pilot, not by inference. The
honest price of the pilot is reviewer time.

## Rollout

Stage 0, shadow: the model drafts answers for 2
weeks, agents see them, customers do not. Log
every draft, citation, and agent edit.
Stage 1, canary: 5 percent of routine tickets get
model drafts shown to agents with one-click
accept. Daily gate review.
Stage 2, expanded: 25 percent, with the agent
still required to confirm the citation.
Stage 3, full routine slice: only after all six
gates pass twice, on two consecutive weekly evals.
Each stage has a written entry criterion and an
automatic halt trigger.

## Monitoring

Latency: p50 and p95 per stage, alert on p95 over
25 seconds. Quality: weekly citation-precision
sample, escalation rate vs the 18 percent
baseline, human preference spot checks.
Safety: PII-leak probes weekly, tool-router
violation count (must stay zero), prompt-injection
attempt log. Drift: KL from the reference
policy, reward-model score distribution, and a
canary set of 200 golden tickets re-run daily.
any golden answer change pages the owner.
Cost: GPU hours per ticket, reviewer minutes per
ticket.

## Rollback

Two independent rollback paths. [Definition]
Weight rollback: the previous accepted weight hash
stays on disk. a config flag flips the server to
it in under 5 minutes. Behavior rollback: a
feature flag disables the draft panel in the
ticketing UI instantly, returning agents to the
manual flow with zero data loss. Rollback is
tested in staging before every stage advance. The
rollback decision owner is the FDE during the
pilot, then the bank ML lead after handoff.

## Ownership

During the pilot: the FDE owns the evaluation rig,
the weight pins, the gate reports, and the
rollback drill. The desk lead owns the golden
ticket set and the preference labels. Compliance
owns the PII probe set. After handoff: the bank
ML lead owns serving and monitoring, the desk
lead owns the eval sets, compliance keeps probe
ownership. No shared ownership of the rollback
flag. one name is on it.

## Handoff

The handoff package: the evaluation rig with the
six gates and the golden set, the runbook (serve,
monitor, rollback, page), the weight-hash
registry, the tool-router allowlist, the PII
probe set, and a 90-day log of every gate
decision with reasons. Handoff is complete when
the bank team runs a rollback drill unassisted
and passes.

## Stakeholder defense

"Just use the biggest model." The constraint is
the two on-prem nodes and the VPC. A bigger
model that does not fit the nodes is not a
candidate. The INT8 mid-size model fits with
cache headroom. that is the selection boundary.

"Quantization always loses accuracy." The lesson
bounds the INT8 error by s/2 per weight and the
gates measure the actual behavior. [Computed:
lesson U10, 2026-10-06] If Gate 1 fails at INT8,
the fallback is FP16 with a smaller batch, not
an argument.

"We do not need evals, the demo looked good."
U10-C10 is the answer: the proxy score can climb
while true quality falls. [Course-title] The
gates measure humans and citations, never the
model's own scores.

"Ship to all tickets at once." The escalation
gate exists because the 30 percent non-routine
slice is out of scope. Staged rollout with halt
triggers is the compromise that keeps the option
to stop.

## Claim index

Hypothetical engagement, composite, no real
client. Computed from the U10 lesson
(2026-10-06): INT8 32 to 8 bytes with error at
most s/2, KV cache 12.00 MiB at n = 512 (24 KiB
per token), reward-hacking toy (proxy 1.20 vs
truth 0.70). Authored planning numbers: ticket
volumes, latencies, costs, SFT/DPO set sizes,
roofline-derived timings. Derived: the cache
batch limit, the under-$0.01 compute cost per
ticket. No production measurement is claimed.
