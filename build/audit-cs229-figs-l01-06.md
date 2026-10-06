# FIGURE ENFORCER audit — cs229 l01–l06
Date: 2026-10-06. Enforcer: FIGURE ENFORCER (subagent b400fced).
Scope: l01-introduction, l02-linear-regression, l03-logistic-regression,
l04-glms-softmax, l05-gda-naive-bayes, l06-bias-variance.
Prose untouched. No markdown edits. Only asset files regenerated/fixed.

## Method
1. Read MASTER_BRIEF.md, REJECTIONS.md, PIPELINE.md, FIGURE_SPEC_STANFORD.md verbatim.
2. Grepped all six lessons for figure references (assets/plate-*.webp,
   assets/svg/*.svg, ```ascii blocks, youtube-nocookie embeds).
3. Checked every referenced file exists on disk: 41/41 image files present
   (plus 6 youtube-nocookie embeds referenced).
4. Opened every webp plate and read its pixels; compared each number
   against the lesson markdown AND recomputed it with Python.
5. Regenerated the three plates whose pixels contradicted the lesson.
   Deterministic PIL rendering (Liberation Sans, warm paper, dark navy
   pill) — no AI image model, no API keys, no media-pipeline calls.
   Script: build/regen_cs229_plates_l01_06.py (durable, re-runnable).
6. Checked all 18 SVGs parse as SVG and their text content matches captions.

## Regenerations and fixes

| # | File | Defect found | Fix applied |
|---|---|---|---|
| 1 | assets/plate-l03-sigmoid-vs-line.webp | STALE pixels: rendered old toy (-0.3/1.2 -> 0.43/0.77). Lesson l03 rebuilt: line predicts 0.27/0.55, sigmoid squeezes to 0.57/0.63. | REGENERATED. sigmoid(0.27)=0.5671->0.57, sigmoid(0.55)=0.6341->0.63 verified by code. |
| 2 | assets/plate-l02-alpha-traces.webp | WRONG pixels: "100 steps reach 1.47" and "minus 48". Lesson l02 L297-306: 0.53 and "4, -8, 16, -32, 64, -128". | REGENERATED. theta_100 = 4*0.98^100 = 0.53; explode trace [4,-8,16,-32,64] verified by code. |
| 3 | assets/plate-l05-gda-vs-qda.webp | WRONG footer pixels: "10,200 vs 5,150 at d = 100". Lesson l05 L211-213: GDA about 5,251 knobs, QDA about 10,301. | REGENERATED. Footer now "10,301 vs 5,251 at d = 100". (5,050+200+1=5,251; 2*5,050+200+1=10,301.) |
| 4 | assets/svg/l03-sigmoid.svg | Caption names g(2)=0.88 and g(-2)=0.12 but the SVG showed no such points. | Added two labeled curve points at code-computed coordinates: z=2 -> (390,104), z=-2 -> (240,256). sigmoid(2)=0.8808, sigmoid(-2)=0.1192. |
| 5 | assets/svg/l06-hyperband.svg | Typo: "1 cfgs". | Fixed to "1 cfg". |

## Per-lesson figure audit

### l01-introduction.md
| Figure reference (line) | Medium | File status | Caption status |
|---|---|---|---|
| assets/svg/l01-paradigms.svg (L56) | SVG section plate | exists, text verified | verified: names source |
| assets/plate-l01-mitchell-tep.webp (L154) | webp lesson plate | exists; pixels read: 10,000 emails, 94->99% — matches lesson L60/L174/L523 | verified |
| assets/svg/l01-paradigm-choice.svg (L309) | SVG section plate | exists, text verified | verified |
| assets/plate-l01-rule-economics.webp (L327) | webp lesson plate | exists; pixels read: 10 rules/day, 30 min/rule, 500 h per 1000 rules, 1000-email probes, 14-day waves — all trace to lesson L47/L311-345 | verified |
| youtube-nocookie embed peBnLUBaXzI (L420) | video | referenced (1 per lesson) | cap present |
| assets/svg/l01-roadmap.svg (L439) | SVG section plate | exists, text verified | verified |
| assets/plate-l01-honest-price.webp (L453) | webp lesson plate | exists; pixels read: 10,000 labels / 5 labeler-days — matches lesson L331 | verified |

### l02-linear-regression.md
| Figure reference (line) | Medium | File status | Caption status |
|---|---|---|---|
| assets/svg/l02-loss-chip.svg (L97) | SVG section plate | exists, text verified | verified |
| assets/plate-l02-error-times-feature.webp (L221) | webp lesson plate | exists; pixels match caption (no computed numbers) | verified |
| assets/svg/l02-gd.svg (L265) | SVG section plate | exists, text verified | verified |
| assets/plate-l02-feature-scaling.webp (L289) | webp lesson plate | exists; "living area 1000-3000, bedrooms 1-5" matches lesson L586 | verified |
| assets/plate-l02-alpha-traces.webp (L316) | webp lesson plate | REGENERATED (was 1.47 / minus 48) | verified; caption already carried the right numbers |
| assets/svg/l02-sgd.svg (L389) | SVG section plate | exists, text verified | verified |
| assets/plate-l02-minibatch.webp (L411) | webp lesson plate | exists; "B = 32 to 256", "64 times less noise" matches lesson L400/L644 | verified |
| assets/svg/l02-normaleq.svg (L482) | SVG section plate | exists, text verified | verified |
| youtube-nocookie embed nk2CQITm_eo (L532) | video | referenced | cap present |
| ASCII blocks (L71,105,163,176,226,349,421,448) | inline ascii | 8 blocks present in text | n/a |

### l03-logistic-regression.md
| Figure reference (line) | Medium | File status | Caption status |
|---|---|---|---|
| assets/svg/l03-mle.svg (L123) | SVG section plate | exists, text verified | verified |
| assets/svg/l03-sigmoid.svg (L209) | SVG section plate | FIXED: added g(2)=0.88 / g(-2)=0.12 point markers the caption names | verified |
| assets/plate-l03-logodds.webp (L296) | webp lesson plate | exists; pixels read: 0.12/0.5/0.88 -> -2/0/+2 — matches lesson L287-293 | verified |
| assets/plate-l03-sigmoid-vs-line.webp (L314) | webp lesson plate | REGENERATED (was old -0.3/1.2 -> 0.43/0.77) | verified; caption already carried the new numbers |
| assets/svg/l03-newton.svg (L360) | SVG section plate | exists, text verified | verified |
| assets/plate-l03-irls.webp (L397) | webp lesson plate | exists; pixels read: h=0.5/0.5/0.9 -> weights 0.25/0.25/0.09 — code-verified (h(1-h)) | verified |
| assets/plate-l03-mle-menu.webp (L463) | webp lesson plate | exists; pixels match caption (no computed numbers) | verified |
| youtube-nocookie embed yIYKR4sgzI8 (L492) | video | referenced | cap present |
| ASCII blocks (L53,87,101,154,198,223,324) | inline ascii | 7 blocks present in text | n/a |

### l04-glms-softmax.md
| Figure reference (line) | Medium | File status | Caption status |
|---|---|---|---|
| assets/svg/l04-expfam.svg (L133) | SVG section plate | exists, text verified | verified |
| assets/plate-l04-eta-worked.webp (L147) | webp lesson plate | exists; pixels read: eta=log(0.8/0.2)=1.39, sigmoid(1.39)=0.8 — code-verified | verified |
| assets/svg/l04-glm.svg (L225) | SVG section plate | exists, text verified | verified |
| assets/svg/l04-softmax.svg (L281) | SVG section plate | exists; (2.0,1.0,0.5)->(0.63,0.23,0.14) code-verified, sums to 1.00 | verified |
| assets/plate-l04-temperature.webp (L315) | webp lesson plate | exists; pixels read: tau=2 -> (0.48,0.29,0.23); tau=0.5 -> (0.84,0.11,0.04) — code-verified | verified |
| assets/plate-l04-label-smoothing.webp (L375) | webp lesson plate | exists; pixels read: (0.9,0.05,0.05), loss 0.59 vs 0.46 — matches lesson L328/L369 (0.5876->0.59) | verified |
| assets/plate-l04-glm-recipes.webp (L419) | webp lesson plate | exists; pixels match caption (no computed numbers) | verified |
| youtube-nocookie embed ytbYRIN0N4g (L449) | video | referenced | cap present |
| ASCII blocks (L63,236,322) | inline ascii | 3 blocks present in text | n/a |

### l05-gda-naive-bayes.md
| Figure reference (line) | Medium | File status | Caption status |
|---|---|---|---|
| assets/plate-l05-generative-vs-discriminative.webp (L96) | webp lesson plate | exists; pixels match caption | verified |
| assets/svg/l05-gda.svg (L184) | SVG section plate | exists; x=22 toy grounded in lesson L159-160 | verified |
| assets/plate-l05-gda-vs-qda.webp (L235) | webp lesson plate | REGENERATED (footer was 10,200 vs 5,150) | verified |
| assets/plate-l05-bernoulli-vs-multinomial.webp (L322) | webp lesson plate | exists; pixels match caption | verified |
| assets/plate-l05-laplace.webp (L348) | webp lesson plate | exists; pixels read: 1/12 — matches lesson L341 ((0+1)/(10+2)) | verified |
| assets/svg/l05-naivebayes.svg (L350) | SVG section plate | exists, text verified | verified |
| youtube-nocookie embed O2L2Uv9pdDA (L532) | video | referenced | cap present |
| ASCII blocks (L52,337) | inline ascii | 2 blocks present in text | n/a |

### l06-bias-variance.md
| Figure reference (line) | Medium | File status | Caption status |
|---|---|---|---|
| assets/svg/l06-biasvar.svg (L90) | SVG section plate | exists, text verified (four-archer layout) | verified |
| assets/plate-l06-decomposition.webp (L104) | webp lesson plate | exists; pixels read: bias^2 0.01, variance 0.027 — code-verified from 0.7/0.9/1.1 at truth 1.0 | verified |
| assets/svg/l06-dd.svg (L166) | SVG section plate | exists, text verified | verified |
| assets/plate-l06-dev-test.webp (L223) | webp lesson plate | exists; pixels match caption | verified |
| assets/plate-l06-ridge-vs-lasso.webp (L334) | webp lesson plate | exists; pixels match caption (circle/diamond shapes) | verified |
| assets/svg/l06-hyperband.svg (L390) | SVG section plate | FIXED: "1 cfgs" -> "1 cfg" | verified |
| assets/plate-l06-random-search.webp (L407) | webp lesson plate | exists; pixels read: grid 9 trials/3 values per dial, random 9 trials/9 values — matches caption | verified |
| youtube-nocookie embed EuBBz3bI-aA (L567) | video | referenced | cap present |
| ASCII blocks (L256) | inline ascii | 1 block present in text | n/a |

## Gate checklist (FIGURE GATES, enforcer side)
- F1 page audit: 47 figure references listed above (41 image files +
  6 youtube-nocookie embeds), 0 missing files, 0 blank cells.
- F2 medium ladder: webp plates for lesson-level before->rule->after claims;
  SVG for section figures; ASCII retained inline where the text has them.
  No ladder violations introduced.
- F3 lesson plates: one claim each; before -> rule -> after layout preserved
  in all regenerations; warm paper, dark navy, 8px-grid-aligned layout,
  Liberation Sans (single typeface per plate).
- F4 chapter plates: none referenced by these lessons; none required.
- F5 captions: all 41 image captions name the source and the project
  (verified-caption format). Untouched.
- F6 reject list: every number-bearing plate opened and its numbers
  recomputed against the lesson — 3 plates failed and were regenerated.
  No robots, brains, glowing networks, stock photos, Comic Sans,
  watermarks, logos, or "generated by" marks in any plate. All 18 SVGs
  parse and render.
- F7 cross-course symbols: l02-loss-chip.svg is the first-defined loss
  chip (CS229 L02); not redrawn elsewhere in l01-l06.

## Notes for the figure auditor / coordinator
- The lesson markdown captions were NOT edited: all three regenerated
  plates' captions already carried the correct numbers; only the pixels
  were stale. Same for the l03-sigmoid.svg caption (already named
  g(2)/g(-2); the figure now shows them).
- Old plate files were overwritten in place (same filenames); no
  stale duplicates left on disk. Repo keeps latest version only.
- build.py was NOT run (another agent owns renders). l07-l17, crash
  course, cheatsheet, index files untouched.
- YouTube IDs were listed, not oEmbed-verified (builder's media-law
  responsibility); IDs: peBnLUBaXzI, nk2CQITm_eo, yIYKR4sgzI8,
  ytbYRIN0N4g, O2L2Uv9pdDA, EuBBz3bI-aA.
- Generation used zero media-pipeline calls and zero API keys; the
  deterministic script lives at build/regen_cs229_plates_l01_06.py.
