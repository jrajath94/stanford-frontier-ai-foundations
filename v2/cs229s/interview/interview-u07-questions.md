# Interview bank U07: Linear attention, SSMs, FFT

Date: 2026-10-06. Questions and keys are separate files.
Provenance: original practice, role-derived. Not actual lab
questions.

## Breadth questions

B1. Write the recurrence and convolution update rules
with their costs.

B2. Write the quadratic attention FLOP and memory
formulas. What happens when T doubles?

B3. Write the kernelized linear attention formula.
What is the state, and what is its shape?

B4. Write the SSM recurrence with shapes. What does
A control?

B5. State the convolution view of an SSM. When does
it collapse?

B6. Write the scan combine rule. Why must it be
associative?

B7. State the FFT convolution theorem. When does
direct convolution win?

B8. Write the stability growth formula. What
parameterization keeps A stable?

## Deep ladder 1: from quadratic to linear

L1a. Define: what makes attention quadratic?
L1b. Toy: T=8192, d=128, one head. Compute score
FLOPs and score memory.
L1c. Derive: prove the 4x law when T doubles.
L1d. Implement and complexity: write the regrouped
linear attention and state its cost.
L1e. Compare: softmax vs linear on the toy. Give
the FLOP ratio and the state memory ratio.
L1f. Debug: copy accuracy collapses under linear
attention. Name the cause and the first fix.
L1g. Critique: state the kernel-approximation
assumption and construct the sharp-selection
counterexample.
L1h. Design: propose the phi-sweep experiment on a
copy task. Name the expected ranking.

## Deep ladder 2: SSM triple identity

L2a. Define: recurrence, convolution view, scan.
What object do all three compute?
L2b. Toy: A=0.9, B=C=1, x=[1,0,1,0]. Compute the
states, the kernel, and the outputs.
L2c. Derive: unroll the recurrence into K_k = C A^k
B and prove the duality.
L2d. Implement and complexity: write the scan
combine and state work vs depth.
L2e. Compare: convolution view vs scan. When does
each win?
L2f. Debug: the tree total disagrees with the loop.
Name the failed property.
L2g. Critique: state the fixed-A assumption behind
the convolution view. Construct the selective-SSM
counterexample.
L2h. Design: propose the crossover measurement
(conv-view vs loop vs scan). Name the expected
band structure.

## Analytical exercises

A1. Stability: A=0.99 vs A=1.01 over 1000 steps.
Compute both. At what step does the 1.01 model
cross fp16 max (65,504)? Show the search logic.

A2. Traffic: T=8192, d=128, fp16 scores vs an SSM
with N=64 fp32 state. Compute attention score
bytes, SSM stream bytes, SSM state bytes, and the
ratio. What single implementation choice destroys
the SSM advantage?

## Implementation / debug task

D1. The FFT convolution below disagrees with the
naive result. Find the bug, fix it, and state the
theorem's padding requirement.

```python
import numpy as np
def fft_conv(x, k):
    return np.fft.ifft(np.fft.fft(x) *
                       np.fft.fft(k)).real
```

## Changed-constraint scenarios

S1. Constraint change: SRAM is infinite. Does the
SSM still beat attention? Rebuild the comparison
on FLOPs and expressivity alone.

S2. Constraint change: the task is offline scoring
of fixed documents (no generation). Which
causality and architecture constraints relax, and
what becomes the cheapest accurate choice?

## Research critique

R1. A paper claims "linear attention matches
softmax on all long-context benchmarks." List five
audit questions, and for each state the answer that
would invalidate the claim.
