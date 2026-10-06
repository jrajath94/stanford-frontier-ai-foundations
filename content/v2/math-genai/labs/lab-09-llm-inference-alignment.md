# Lab 09, LLM inference and alignment mechanics

Unit: math-genai-U10. Date: 2026-10-06. numpy 1.26.4,
float64, seed 0 where RNG is used. Keys in
labs/keys-lab-09.md. Test-mode: solve closed-book, then
check. Toys: logits [2.0, 1.0, 0.5, 0.1], KV toy (12
layers, 8 heads, 64 dim, fp16), W 4x2, alignment toys
as in the lesson. C04 numbers are authored arithmetic.
Ground truth: compute_run5b.py.

## Task 1, decoding knobs: predict first, measure second

(a) Predict: pmax and entropy at T = 0.5 vs T = 2.0.
Write both before computing.
(b) Measure: implement softmax/T/top-p. Record the
three distributions and entropies.
(c) Apply top-p 0.9 with the lesson's convention.
Record kept indices and renorm probs.
(d) Write one sentence: which knob for adventure,
which for the quality floor?

## Task 2, the KV cache

(a) By hand: KV bytes for L=12, h=8, d_h=64,
n=512, fp16. Verify 12582912.
(b) In code: bytes vs n for n in [128, 512, 2048,
8192]. Verify linear scaling.
(c) Prefill vs decode FLOPs on the toy. Verify
the ratio equals n.
(d) Break it: forget the factor 2. Record the
wrong bytes and the consequence at deploy time.

## Task 3, quantization

(a) By hand: s and zp for W. Verify 0.013725 and
87.
(b) In code: quantize/dequantize all 8 entries.
Record max and mean abs err. Verify the s/2
bound.
(c) Widen the calibration range to [-10, 10].
Recompute s and the bound. Record the error
ratio vs the true range.
(d) Write one sentence: what does the range buy?

## Task 4, alignment losses

(a) By hand: BT loss for r_c=1.2, r_r=0.3. Verify
0.3412 nats.
(b) In code: PPO objective for (1.3, 0.5) and
(0.5, -0.4). Verify 0.6000 and -0.3200. Name
which term the min picks in each.
(c) DPO: verify margin 0.07, loss 0.6588,
implicit rewards 0.0400/-0.0300.
(d) KL: verify 0.0995 bits. Then set pi =
pi_ref and recompute.

## Task 5, hacking and the audit

(a) Recompute the proxy rewards with and without
the length bonus. Record who wins each time.
(b) Compute the BT win-rate over the 4 pairs.
Verify 0.5834.
(c) Reproduce the C12 audit denominators from
the playlist titles and repo paths. Record the
4/7/1 split and name the 4.
(d) Write one paragraph: your model ships when?
Name the eval, the audit, and the knob that is
still tunable after training.
