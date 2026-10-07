# prerequisites.md, cs229

Date: 2026-10-06.

Shared bridges exist at `../shared/prerequisites/`. They are linked
here, not rebuilt. Each unit lesson carries its own local remediation
for the exact dependencies its mechanisms need.

## Prerequisite atlas, zero-assumption order

- P01 Numeracy, algebra, and notation (`p01_numeracy.md`)
- P02 Python and scientific software (`p02_python.md`)
- P03 Vectors, geometry, and linear maps (`p03_vectors.md`)
- P04 Spectral and numerical linear algebra (`p04_spectral.md`)
- P05 Scalar and multivariable calculus (`p05_calculus.md`)
- P06 Probability from events to distributions (`p06_probability.md`)
- P07 Statistical estimation and uncertainty (`p07_estimation.md`)
- P08 Information theory and density objectives (`p08_information.md`)
- P09 Optimization and constrained problems (`p09_optimization.md`)
- P10 ML foundations and evaluation (`p10_ml_foundations.md`)
- P11 Neural networks and autodiff (`p11_neural_nets.md`)
- P12 PyTorch, tensors, and numerical stability (`p12_pytorch.md`)
- P13 Language and sequence modelling (`p13_language.md`)
- P14 Transformer mechanics (`p14_transformer.md`)
- P15 Hardware and computer architecture (`p15_hardware.md`)
- P16 Distributed systems and networking (`p16_distributed.md`)
- P17 Reinforcement learning (`p17_rl.md`)
- P18 Bayesian inference, latent variables, sampling (`p18_bayesian.md`)
- P19 Retrieval and search (`p19_retrieval.md`)
- P20 Tools, MCP, and agents (`p20_tools.md`)
- P21 Security and trust (`p21_security.md`)
- P22 Experimental method and research literacy (`p22_experiments.md`)
- P23 Economics and incentives (`p23_economics.md`)
- P24 Reserved for future bridge modules.

## Per-unit prerequisites, initial list, refined at atomic level inside lessons

| Unit | Prerequisites |
|---|---|
| U01 | P01, P02, P03, P06, P10 |
| U02 | P04, P05, P07, P09 |
| U03 | P05, P06, P07, P09 |
| U04 | P04, P06, P07 |
| U05 | P03, P04, P09 |
| U06 | P03, P05, P09 |
| U07 | P05, P11, P12 |
| U08 | P06, P07, P10 |
| U09 | P07, P09, P10 |
| U10 | P04, P07, P18 |
| U11 | P04, P08, P18 |
| U12 | P06, P08, P18 |
| U13 | P10, P11, P14 |
| U14 | P12, P13, P14 |
| U15 | P06, P17 |
| U16 | P04, P05, P09, P17 |
| U17 | P07, P08, P22 |

## Local remediation rule

Every unit lesson opens with a dependency list: numbered items the
learner does not yet understand. Each gets a self-contained bridge
inside the lesson. A shared bridge is referenced for depth, never as
the only path.
