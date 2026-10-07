# keys.md, U10 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Amortized = B/N + q.

## E02

12.2 ms at 1M queries. Break-even: 160,000
queries.

## E03

Volume is below break-even. The build never
amortizes.

## E04

The build cost recurs hourly. Amortization
resets. The break-even N multiplies by rebuilds.

## E05

Hypothesis: the break-even formula predicts the
crossover within 2x. Report both.

## E06

Dense: docs * dims * bytes. PQ: docs *
code_bytes.

## E07

Dense: 3,072,000,000 bytes = 2.86 GiB. PQ:
64,000,000 bytes = 61 MiB. Ratio: 48x.

## E08

Compression was too aggressive. The code bytes
cannot hold the distinctions.

## E09

Dense halves. PQ is unchanged (code bytes
fixed). The ratio halves to 24x.

## E10

Hypothesis: recall is flat then falls past a
knee in PQ bytes. Report the curve.

## E11

Recall@k: fraction of the true top-k the system
returns.

## E12

Brute (1.00 @ 50 ms) and HNSW (0.98 @ 8 ms)
meet 0.97. IVF (0.95) fails.

## E13

Measure recall by query difficulty. The mean
hides the hard-query tail.

## E14

6 ms budget: IVF (5 ms) wins. HNSW (8 ms) is
over budget.

## E15

Hypothesis: hard-query recall is far worse than
the mean. Report the distribution.

## E16

RRF(d) = sum 1/(k + rank) over systems.

## E17

D: 0.031258. 1st+100th: 0.022643. D wins:
agreement beats one loud vote.

## E18

The dense list dilutes the keyword signal. On
pure-keyword queries the fusion adds noise.

## E19

k = 0: RRF = sum 1/rank. Rank differences
shout. Deep ranks vanish.

## E20

Hypothesis: hybrid wins overall but loses on
pure-keyword queries. Report the split.

## E21

Total = sum over stages of count * cost_each.

## E22

405 ms. Tightened to top 10: 205 ms.

## E23

The ANN's top 100 missed it. The reranker never
saw the gold doc.

## E24

Widen the funnel: the reranker sees everything
the ANN returns. Cost is no longer the bind. 
ANN recall is.

## E25

Hypothesis: recall vs latency has a knee in
funnel width. Report it.

## E26

Measured, derived, stated, opinion (in that
order of weight).

## E27

$72,000/day. Useful-day at 48% MFU: $150,000.

## E28

The price (or the MFU) was stale or assumed.
Arithmetic right, inputs wrong.

## E29

It halves to $75,000. Useful work gets cheaper.

## E30

Hypothesis: half the guest claims shift by over
20% on re-derivation. Report the audit.

## E31

Measured, derived, stated, opinion.

## E32

"Beats by N%" with no table: stated. "O(T)":
derived. "Will replace": opinion.

## E33

They skipped the artifact: no table, no
baselines, no seeds. The claim was stated, not
measured.

## E34

The grade can move to measured (if baselines
and seeds are sound).

## E35

Hypothesis: re-grading against the paper moves
the grade. Report the move.

## E36

Question, hypothesis, baselines, budget, failure
criteria.

## E37

Toy: 7/10. Fixed: 9/10.

## E38

Failure criteria were absent. The team had no
contract for the pivot.

## E39

Time and attention still bind. The budget line
is not the only scarcity.

## E40

Hypothesis: the capstone proposal scores 9+
against this rubric. Report the gaps.

## E41

Component, mechanism, predicted share. Then
measure.

## E42

Error 4 points. Verdict: survives (under 10).

## E43

Kill the hypothesis. The real suspect is
elsewhere. Form a new hypothesis around it.

## E44

Fishing trip: you will find something, and it
will be wrong.

## E45

Hypothesis: of three hypotheses, at least one
dies on measurement. Report the survivor.

## E46

Three seeds minimum, mean plus spread, same
budget and hardware.

## E47

Mean 2.31, std 0.02. Rival at 2.30: gap 0.01 <
1 std. No evidence.

## E48

Nondeterministic kernels, or mixed hardware
across seeds.

## E49

Nothing about the idea. One run is a rumor, not
a result.

## E50

Hypothesis: the 6-seed mean holds within 1 std
of the 3-seed mean. Report all six.

## E51

Ship iff delta >= -epsilon on the task metric.

## E52

A (-0.3%): ships. B (-0.7%): blocked. Boundary
(-0.5%): ships.

## E53

The metric is a proxy the optimization games.
Gate the real task score.

## E54

Noise blocks everything. No optimization ships. 
the gate is unusable.

## E55

Hypothesis: the gate catches a known-bad
optimization. Report the catch.

## E56

Claim, evidence, baseline, failure, cost.

## E57

8/10. Gaps: baseline and cost.

## E58

"I skipped it, so I cannot defend that wall.
Here is what I would run." Honest and
specific.

## E59

Yes. Negative results defend fine with
evidence: what was ruled out is a result.

## E60

Hypothesis: every hostile rehearsal finds one
gap. Report the gaps.
