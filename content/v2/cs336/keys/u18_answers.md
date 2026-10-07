# U18 answer key

All numeric claims from `../visuals/compute_u18.py` (synthetic toys,
executed 2026-10-06). Claim class: REQUESTED-BRANCH.

## R1 (remediation)

The three sentences: name the source and what is reported, state
what was inspected (nothing), state that content is unknown and
will not be inferred.

## A1

(a) G1 Jun 1 Daniel Selsam. G2 Jun 3 Dan Fu (reported, secondary
schedule). (b) 17+2=19. (c) Record the rumor with source and date.
mark content unknown, do not teach it.

## A2

(a) Course site, lecture repo, speaker page. "obtained" = bytes
with inspection extent. (b) Three attempts, zero artifacts. (c)
Slides inspected partial (record pages), recording still missing:
record both extents separately.

## A3

(a) No content claims about guests without artifacts. (b) The
systems researcher teaching eval methods: reputation guessed
wrong. (c) "The guest lectures' content is not in our sources.
the schedule lists the speakers and dates." Then stop.

## A4

(a) Every session mapped once, count = 19. (b) 19. S07 split
documented. (c) Add it as S20/G3 with the same unknown-content
protocol, re-verify the count.

## A5

(a) Model plus manifest, config, eval report, ablations, diary,
ledger. (b) 12 items in ../capstones/. (c) Check the eval report
first: a beat without a clean eval is noise.

## A6

(a) Full 2^4, half 2^{3}, OAT 1+4. (b) 16/8/5. (c) 6 factors, 20
runs: fractional factorial 2^{6-2}=16 plus 4 center/confirmation
runs, protect the top suspected interaction from aliasing.

## A7

(a) Date, symptom, hypothesis, fix, lesson. (b) The NaN entry.
(c) The debugging skill is lost: the next identical failure costs
full diagnosis time again.

## A8

(a) GPU-hours, storage, failures itemized. (b) $1,200. 120
GPU-hours of failures inside the 480. (c) Scale the measured
rates: 10x the runs at the ledger's $/GPU-hour, plus a failure
allowance from the diary rate.

## A9

(a) Independent rerun landing inside seed bands. (b) The three
toy bands. (c) No: all three must pass, report 2/3 with the
failing metric named.

## A10

(a) Every claim carries its scale label, no extrapolation without
staged evidence. (b) 1M params evidenced. 70B not evidenced.
(c) "Not evidenced at that scale in this work" - require the
staged argument or strike the claim.

## A11

(a) Canary, metric comparison, pre-written rollback. (b) 1%/24h.
rollback at +0.5 error or +20% p99. (c) Roll back: complaints are
the truth metric, investigate with the canary data.

## A12

(a) Define, toy, derive, implement, compare, debug, critique,
design. (b) Rubric 30/25/20/15/10, limits step usually weakest.
(c) 10 minutes: one mechanism per unit at most, pick the arc:
budgets -> serving -> measurement -> data -> adaptation ->
alignment -> systems -> defense.
