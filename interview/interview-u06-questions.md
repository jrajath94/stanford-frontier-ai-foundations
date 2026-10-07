# Interview bank U06: Training, adaptation, data pipelines

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Write the two-point scaling-law fit and the compute
budget formula C = 6ND.

B2. Define k-shot prompting. Why do the weights stay
frozen?

B3. State the threshold artifact behind apparent
emergence jumps.

B4. Write the masked SFT loss. Which tokens carry zero
weight, and why?

B5. Write the Bradley-Terry preference probability and
the KL-regularized RL objective.

B6. Describe the constitutional critique-revise loop.
Who writes the rules, who applies them?

B7. Write the LoRA update and its parameter count vs
full fine-tuning.

B8. Write the overlapped pipeline step-time formula.
Name the two regimes.

## Deep ladder 1: scaling economics

L1a. Define: what is a scaling law?
L1b. Toy: N1=1e9 L1=2.40, N2=1e10 L2=2.02. Fit and
predict N=1e11.
L1c. Derive: show the fit passes through its own
points and compute the decade slope.
L1d. Implement and complexity: write fit_law/predict
and state what the points cost.
L1e. Compare: two-point line vs floor form. When
does each win?
L1f. Debug: the big run beats the prediction by 0.2.
Name two causes.
L1g. Critique: state the fixed-recipe assumption and
construct the data-mix counterexample.
L1h. Design: propose the three-run falsification.
Name the acceptance band.

## Deep ladder 2: preference learning

L2a. Define: what is the reward model, and what is
the KL leash?
L2b. Toy: r_A=1.2, r_B=0.8, KL=2.5, beta=0.1.
Compute P(A wins) and the objective.
L2c. Derive: show why a closer policy beats a
farther one at equal reward.
L2d. Implement and complexity: write the objective
and state the label cost difference RLHF vs RLAIF.
L2e. Compare: explicit reward model vs DPO. When is
each the right choice?
L2f. Debug: reward climbs, human ratings fall. Name
the failure and the first fix.
L2g. Critique: state the reward-model-is-truth
assumption. Construct the hacking counterexample.
L2h. Design: propose the reward-vs-KL sweep. Name
the expected curve and the knee.

## Analytical exercises

A1. Scaling toy: C = 6ND with N=1e9, D=2e10. Compute
C. The budget doubles. Under the toy law with slope
-0.0748 per decade in N at fixed D/N ratio, how many
decades of N does doubling C buy, and what loss drop
does it predict?

A2. Seed sweep: losses [2.31, 2.29, 2.33, 2.30, 2.32].
Compute mean and sample std. Recipe B reports 2.295
over 5 seeds. Is there evidence B wins? A third
recipe reports 2.27 over 5 seeds. Now what?

## Implementation / debug task

D1. The packing code below fills bins but training
diverges on long documents. Find the bug, fix it, and
state the invariant.

```python
def pack(seqs, cap):
    bins, cur = [], []
    for s in seqs:
        if sum(len(x) for x in cur) + len(s) <= cap:
            cur.append(s)
        else:
            bins.append(cur)
            cur = [s]
    bins.append(cur)
    return bins  # bug: no document mask is returned
```

## Changed-constraint scenarios

S1. Constraint change: human labels are free and
instant. Does RLAIF still have a role? Rebuild the
RLHF vs RLAIF vs DPO choice under this constraint.

S2. Constraint change: the cluster price drops to
zero but the power cap stays. Which U06 lever
(scaling, packing, loader, LoRA) still matters, and
which becomes irrelevant?

## Research critique

R1. A paper claims "our 1B model matches a 10B model,
so scaling laws are broken." List five audit
questions, and for each state the answer that would
invalidate the claim.
