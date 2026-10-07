# Course map , cs336 nineteen sessions

Nineteen scheduled sessions mapped to units without omissions.
Baseline: 2026-10-06.

Session skeleton (titles, dates, instructors) comes from SRC-04, a secondary
study wiki. It is NOT official evidence. Every session leaf is therefore
PLANNED / SOURCE ATTRIBUTION PENDING. Claim class per session:
OFFICIAL-SOURCE only where an inspected official artifact backs the leaf,
REQUESTED-BRANCH everywhere else.

## Block 1 , Foundations (weeks 1-4)

| # | Date | Session (reported) | Instructor (reported) | Units | Claim class |
|---|------|--------------------|-----------------------|-------|-------------|
| S01 | Mar 30 | Overview, Tokenization | Percy Liang | U01 | REQUESTED-BRANCH, U01-C03/C04 partial OFFICIAL-SOURCE via SRC-03 |
| S02 | Apr 1 | PyTorch, Resource Accounting | Percy Liang | U02 | REQUESTED-BRANCH |
| S03 | Apr 6 | Architectures, Hyperparameters | Tatsu Hashimoto | U03 | REQUESTED-BRANCH |
| S04 | Apr 8 | Attention Alternatives, MoE | Tatsu Hashimoto | U04 | REQUESTED-BRANCH |

## Block 2 , Systems (weeks 5-8)

| # | Date | Session (reported) | Instructor (reported) | Units | Claim class |
|---|------|--------------------|-----------------------|-------|-------------|
| S05 | Apr 13 | GPUs, TPUs | Tatsu Hashimoto | U06 | REQUESTED-BRANCH |
| S06 | Apr 15 | Kernels, Triton, XLA | Percy Liang | U07 | REQUESTED-BRANCH |
| S07 | Apr 20 | Parallelism I (data, pipeline) | Percy Liang | U08, U09-C02/C03/C04 | REQUESTED-BRANCH |
| S08 | Apr 22 | Parallelism II (tensor, expert) | Tatsu Hashimoto | U09 | REQUESTED-BRANCH |

## Block 3 , Scaling (weeks 9-10)

| # | Date | Session (reported) | Instructor (reported) | Units | Claim class |
|---|------|--------------------|-----------------------|-------|-------------|
| S09 | Apr 27 | Scaling Laws | Tatsu Hashimoto | U10 | REQUESTED-BRANCH (second builder) |
| S10 | Apr 29 | Inference | Percy Liang | U11 | REQUESTED-BRANCH (second builder) |
| S11 | May 4 | Scaling Laws II | Tatsu Hashimoto | U10 | REQUESTED-BRANCH (second builder) |

## Block 4 , Data (weeks 11-12)

| # | Date | Session (reported) | Instructor (reported) | Units | Claim class |
|---|------|--------------------|-----------------------|-------|-------------|
| S12 | May 6 | Evaluation | Percy Liang | U12 | REQUESTED-BRANCH (second builder) |
| S13 | May 11 | Data: Sources, Transformation, Filtering | Percy Liang | U13 | REQUESTED-BRANCH (second builder) |
| S14 | May 13 | Data: Mixing, Rewriting, SFT | Percy Liang | U14, U15 | REQUESTED-BRANCH (second builder) |

## Block 5 , Alignment (weeks 13-15)

| # | Date | Session (reported) | Instructor (reported) | Units | Claim class |
|---|------|--------------------|-----------------------|-------|-------------|
| S15 | May 18 | Alignment: RLHF/DPO | Tatsu Hashimoto | U16 | REQUESTED-BRANCH (second builder) |
| S16 | May 20 | Alignment: RL Algorithms | Tatsu Hashimoto | U16 | REQUESTED-BRANCH (second builder) |
| S17 | May 27 | Alignment: RL Systems | Percy Liang | U17 | REQUESTED-BRANCH (second builder) |

## Guest lectures

| # | Date | Session (reported) | Speaker (reported) | Units | Claim class |
|---|------|--------------------|--------------------|-------|-------------|
| G1 | Jun 1 | Guest lecture | Daniel Selsam | U18 | REQUESTED-BRANCH (second builder). Content unknown, see G-08. Do not infer from speaker name. |
| G2 | Jun 3 | Guest lecture | Dan Fu | U18 | REQUESTED-BRANCH (second builder). Content unknown, see G-08. Do not infer from speaker name. |

## First-half unit/session reconciliation

- S01 maps to U01 (12 concepts, C01-C12). No omissions.
- S02 maps to U02 (12 concepts, C01-C12). No omissions.
- S03 maps to U03 (12 concepts, C01-C12). Hyperparameter optimization
  branches that also touch U05-C07 and U05-C10 are cross-linked, not moved.
- S04 maps to U04 (12 concepts, C01-C12). No omissions.
- S05 maps to U06 (12 concepts, C01-C12). No omissions.
- S06 maps to U07 (12 concepts, C01-C12). No omissions.
- S07 maps to U08 (C01-C12) plus U09 pipeline concepts (C02, C03, C04).
  The split follows the reported session focus: data parallelism in S07,
  tensor and expert parallelism in S08. U09 keeps all 12 concepts as one
  unit for teaching coherence, the map records the session split.
- S08 maps to U09 (C01-C12). No omissions.
- Optimization and training correctness (U05) has no dedicated reported
  session in the first half, the course assigns it as a cross-cutting
  branch anchored in reported Assignment 1 objectives (transformer LM,
  cross-entropy, training loop, optimizer). It is taught as U05 with all
  12 concepts. This placement is a REQUESTED-BRANCH decision, recorded
  here rather than hidden.

## Second-half unit/session reconciliation (2026-10-06, second builder)

- S09 (Scaling Laws) and S11 (Scaling Laws II) map to U10 (C01-C12).
  The two-session split follows the reported schedule: fitting and
  isoFLOP curves in S09, allocation and extrapolation limits in S11.
- S10 (Inference) maps to U11 (C01-C12). No omissions.
- S12 (Evaluation) maps to U12 (C01-C12). No omissions.
- S13 (Data: Sources, Transformation, Filtering) maps to U13 (C01-C12).
- S14 (Data: Mixing, Rewriting, SFT) maps to U14 (C01-C12) plus U15
  (C01-C12). The split follows the reported session focus: filtering,
  dedup, mixing in U14, midtraining and SFT adaptation in U15.
- S15 (Alignment: RLHF/DPO) and S16 (Alignment: RL Algorithms) map to
  U16 (C01-C12). The split follows the reported schedule: preference
  modeling and DPO in S15, PPO/GRPO and reward design in S16.
- S17 (Alignment: RL Systems) maps to U17 (C01-C12): multimodal
  interfaces plus the training-system integration the reported session
  title names.
- G1 and G2 map to U18 (C01-C12). Content unknown per G-08, the unit
  teaches the honesty protocol, replication discipline, ablation design,
  and the oral defense ladder. No content is inferred from speaker names.
- Multimodality has no dedicated reported session, it is taught as the
  first half of U17 as a REQUESTED-BRANCH decision, recorded here rather
  than hidden, with scale-scope limits stated (U17-C12).

Session count check: 17 core + 2 guest = 19. All 19 appear above. Zero
sessions unmapped. Zero sessions mapped twice at leaf level except the
documented S09/S11, S15/S16, and S14 splits.

## Claim class summary (all 19 sessions)

Every session leaf is REQUESTED-BRANCH except U01-C03/C04 (partial
OFFICIAL-SOURCE via SRC-03). All leaves are PLANNED / SOURCE
ATTRIBUTION PENDING. The second builder verified this on 2026-10-06
against `source_manifest.md` and `source_gaps.md` before writing U10-U18.
No inspected official artifact backs any second-half leaf.
