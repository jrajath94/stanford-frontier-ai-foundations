# keys.md, U09 lesson answer keys

Date: 2026-10-06. Closed-book answers. Keep separate from the
lesson file.

## E01

Data: 8 full copies, split data. Tensor: layers
sliced, split matmuls. Pipeline: stages split,
whole layers. Expert: experts split, tokens move.

## E02

DP: 20 GB/GPU. TP-8: 2.5 GB/GPU. PP-8: 2.5
GB/stage. EP-8: 2.5 GB/GPU plus dispatch.

## E03

The all-reduce tax: TP syncs every matmul, and
the thin link multiplies each sync.

## E04

DP dies (100 GB > 40 GB). TP/PP/EP at 12.5
GB/GPU still run.

## E05

Hypothesis: measured step times rank by their
tax: collective vs bubble vs dispatch. Report
all four.

## E06

t = 2*(n-1)/n * S/BW. Passes: reduce-scatter,
then all-gather.

## E07

n=8: 0.35 s. N=64: 0.39375 s.

## E08

Incast congestion, or a degraded link (one slow
peer throttles the ring).

## E09

t falls 4x (linear in S).

## E10

Hypothesis: all-reduce time is linear in S with
slope 1.75/BW. Report the fit.

## E11

Keep tensor-parallel groups inside the node.
Let data parallel (or pipeline) cross nodes.

## E12

Intra-node: 0.058 s. Split: 0.70 s. Ratio: 12x.

## E13

The TP group is split across nodes. Every sync
crosses the thin link.

## E14

Placement stops mattering. Any split pays the
same collective cost.

## E15

Hypothesis: the measured ratio matches
BW_in/BW_out. Report both.

## E16

Step time = max(worker times).

## E17

Step 14 s, tax 40%, 28 GPU-seconds wasted per
step.

## E18

Check per-worker times: a degrading GPU or disk
(a growing straggler).

## E19

Staleness: workers train on old parameters and
convergence can suffer.

## E20

Hypothesis: a small set of workers explains most
of the tail. Report the split.

## E21

Bubble = (p-1)/(m+p-1).

## E22

m=8: 27.3%. M=16: 15.8%.

## E23

Stage times are uneven (e.g. the embedding stage
runs 2x). The formula assumed equal stages.

## E24

m=1: no pipeline at all, just sequential stages
(100% bubble).

## E25

Hypothesis: idle fraction follows
(p-1)/(m+p-1). Report the fit.

## E26

ZeRO-1: optimizer states. ZeRO-2: + gradients.
ZeRO-3: + parameters.

## E27

ZeRO-3: 20.0 GB/GPU. Plain DP: 160 GB/GPU.

## E28

Gather latency dominates for tiny layers. The
per-layer all-gather costs more than the memory
saves.

## E29

ZeRO-1 saves nothing (no states to shard). It is
a no-op for SGD without momentum.

## E30

Hypothesis: memory falls across stages 1/2/3. 
step time rises at stage 3 for small layers.
Report all.

## E31

Overhead = (N/k) * save_cost.

## E32

k=100: 300 s overhead (3%), rework capped at
1000 s. K=500: 60 s (0.6%), rework capped at
5000 s.

## E33

The data-loader state (or RNG state) was not
saved. Resume replays the wrong order.

## E34

Save every step. Overhead is zero and rework is
one step.

## E35

Hypothesis: sharded saves scale per-GPU time
down with shard count. Report the curve.

## E36

Makespan: total time to finish all jobs. Waste:
GPU-hours idle.

## E37

Makespan 2 h, waste 4 GPU-h.

## E38

FIFO without backfill: big jobs block small
ones (head-of-line blocking).

## E39

Elastic jobs shrink to fill gaps: waste falls,
at the cost of slower individual jobs.

## E40

Hypothesis: backfill cuts mean wait vs FIFO.
Report both.

## E41

Rework fraction = preemptions * (interval +
resume) / run time.

## E42

1 expected preemption. Rework 5.8% of the run.

## E43

The preemption rate is too high (or correlated):
rework exceeds the discount.

## E44

One crash loses the whole run. Maximum risk.

## E45

Hypothesis: preemptions are bursty, not
Poisson. Report the fit.

## E46

Dominant share = max over resources of
demand/total.

## E47

A: 0.5. B: 0.5. C: 1.0.

## E48

Demand inflation: gaming the dominant-share
accounting.

## E49

DRF becomes max-min fair sharing of the one
resource.

## E50

Hypothesis: DRF cuts max wait vs FIFO on mixed
workloads. Report both.

## E51

MFU = achieved FLOP/s divided by peak FLOP/s.

## E52

MFU 48.1%. Bubble-free projection: 65.8%.

## E53

Recomputation (or inflated FLOP counting). The
numerator grew without speeding the run.

## E54

MFU halves (same achieved, double peak).

## E55

Hypothesis: the top two taxes explain over half
the MFU gap. Report the split.

## E56

Cluster MTBF = per-GPU MTBF / GPU count.

## E57

102.7 hours = 4.28 days. About 7 failures per
30-day run.

## E58

Independence broke: failures are correlated
(e.g. rack power events).

## E59

26,280/4096 = 6.4 hours.

## E60

Hypothesis: correlated failures exceed the
independent prediction. Report both.
