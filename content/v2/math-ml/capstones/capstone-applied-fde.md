# Capstone F1, applied/FDE: default-risk scorer for a small lender

Track: applied / field engineering. Date: 2026-10-06. Status:
design complete. No model is trained here. No customer data is
used. Numbers below are illustrative toy values from the lesson,
labeled as such, or explicit assumptions labeled ASSUMPTION.

## Discovery

The lender's analysts score loan applications by hand using five
rules of thumb. A typical decision takes 20 minutes
(ASSUMPTION from stakeholder interviews, to be verified). The
pain: inconsistent decisions across analysts and no audit
trail. The ask: a scorer that ranks applications by default
risk, explains each score, and fits the existing review queue.

## Workflow baseline

Today: application arrives -> analyst reads the file (20 min)
-> committee reviews edge cases weekly -> decision logged in a
spreadsheet. The baseline to beat is the analysts' own
agreement rate and speed, not a Kaggle score.

## Objectives

1. Rank applications by estimated default probability.
2. Every score ships with the top three contributing features
in plain language.
3. Decision latency under 1 second per application
(ASSUMPTION, to be load-tested).
4. Full audit trail: inputs, model version, score, explanation.

## Constraints

- Features must come from the current application form only:
no new data collection in phase 1.
- The model must be explainable to a non-technical auditor.
- Retraining needs analyst sign-off. No silent auto-retrain.
- Regulatory: adverse-action notices need specific reasons.

## Trust boundaries

The model never approves or rejects alone: it ranks and
explains, a human decides. Scores are hidden from applicants.
Training data never leaves the lender's VPC. Analyst overrides
are logged and reviewed monthly.

## Alternatives considered

- Hand rules kept: rejected, inconsistent and unaudited.
- Logistic regression: chosen for phase 1. Calibrated
probabilities, linear explanations, convex training.
- Gradient boosting: deferred to phase 2. Better accuracy,
harder explanations. Needs the audit story first.
- Kernel SVM: rejected for phase 1. O(n^2) scoring cost and
opaque decisions fail the explanation constraint.

## Model choice rationale (ties to the course)

Logistic over hinge: the product is a ranked probability with
reasons, not a boundary. The lesson's C12 numbers (hinge wins
under label noise) are noted and set aside: here the labels
are audited defaults, noise is low, and calibration matters
more than noise resistance. If label audits later show noise
above 10 percent, the choice is revisited: that trigger is
written into the monitoring plan.

## Acceptance gates

1. Held-out AUC >= analysts' agreement-implied AUC (baseline
to be measured in discovery. Gate is relative, not absolute).
2. Calibration: predicted vs realized default rates within 2
points per decile on held-out data.
3. Explanation audit: 50 sampled explanations reviewed by two
analysts. 90 percent rated "actionable."
4. Latency: p99 < 1 second on the production shape.
5. Rollback drill passes (see below).

## Cost and latency assumptions

- Training: nightly batch on one CPU box, minutes (ASSUMPTION).
- Scoring: one logistic evaluation per application,
microseconds of compute. The 1-second budget is I/O and
explanation rendering (ASSUMPTION, to be measured).
- People: 0.5 analyst for override review monthly.

## Rollout

Phase 0: shadow mode, 4 weeks. Scores logged, not shown.
Phase 1: 10 percent of applications show scores to analysts.
Phase 2: full queue, human decides throughout. Each phase
needs the acceptance gates re-verified.

## Monitoring

- Feature distribution drift vs training (weekly PSI).
- Score distribution drift (weekly).
- Override rate and override outcomes (monthly).
- Label-noise trigger: if audit finds > 10 percent label
noise, revisit logistic vs hinge (the C12 lesson).
- Calibration deciles recomputed monthly.

## Rollback

One config flag returns the queue to hand scoring. The flag is
tested monthly in a drill. Rollback criterion: any acceptance
gate fails for two consecutive weeks, or override rate exceeds
30 percent.

## Ownership and handoff

Owner: the lender's risk team (named person, not "the team").
Builder handoff: training script, feature list with units,
scaling parameters (the C11 lesson: standardize, record the
scaler), model card, explanation templates, runbook, rollback
drill log. The handoff is complete when the risk-team owner
runs the retrain and the rollback drill unaided.

## Stakeholder defense (one page, plain language)

The model does not replace the analysts. It ranks their queue
and writes down its reasons. Every reason traces to a feature
the analysts already use. If the model is wrong, the audit
trail shows exactly what it saw. The rollback flag is tested
monthly, so the worst case is four weeks of shadow logs, not
a broken process.
