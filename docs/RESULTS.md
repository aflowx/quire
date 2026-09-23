# Results

All numbers are on public items and were measured by us, unless marked *published*. Comparisons between two configurations are paired over the same items (two-sided sign-flip permutation test). Files, configurations, commits and commands are listed in [`results/README.md`](../results/README.md).

## JevBench v1.3

JevBench scores 534 decisions on four axes (Intelligence, Calibration, Speed, Cost) and takes their geometric mean. 231 decisions are public. The judge tier (146 decisions, 28% of Intelligence) and 109 hard decisions are held out and run only by the maintainers, so no number here is a JevBench score.

### Quire on the public items

| configuration (Qwen3.5-4B, frozen, BF16) | easy /48 | standard /72 | hard /111 | hard ECE | fidelity TVD |
|---|---|---|---|---|---|
| plain prompt, 2 orderings | 48 | 63 | 72 | 0.102 | 0.280 |
| quire prompt, 1 ordering | 48 | 64 | 70 | 0.113 | 0.312 |
| **quire prompt, 2 orderings (released)** | **48** | **65** | **73** | **0.090** | **0.241** |

The released configuration measured the same way through JevBench's own harness gives identical results (`results/jevbench/harness-typesafe-adapter/`). Against one ordering, averaging two lowers hard-tier calibration error from 0.113 to 0.090 and adds 3 hard items (9 better / 6 worse, p = 0.61). Against the plain prompt the differences are within two items per tier. (The plain-prompt row was measured before non-string states were rendered as JSON, so it isn't an exact control on the 35 hard items with object states.)

The quire prompt was chosen from three candidate wordings using held-out synthetic policy and adequacy items (`quire`: 381/482, the others 375 and 369), not JevBench items.

### Per-answer-type temperature

Pre-registered before any prediction. The map was fitted on held-out development items (Quire's synthetic suites and kev's decision-v7; 1,464 to fit, 1,466 to check, choices over 26 options excluded), checked there, then confirmed once on JevBench's public items. It changes no answer.

| | before | with the map |
|---|---|---|
| fitted temperatures | — | choice 1.40 · yes/no 1.70 · score 1.40 |
| held-out check half: pooled ECE (NLL) | 0.092 (0.759) | **0.030** (0.710) |
| check half by type: choice / yes/no / score ECE | 0.103 / 0.090 / 0.193 | 0.047 / 0.041 / 0.124 |
| JevBench public hard tier: ECE | 0.090 | **0.061** |
| JevBench public hard tier: fidelity TVD | 0.241 | 0.248 |
| JevBench Calibration axis | 79.0 | **81.4** |

The map is part of the released configuration from v0.1.1. It doesn't improve fidelity to the gold distributions of the probability items. Those need the model to put the argmax on the right side, which a temperature can't do. Files: `results/calibration/`, `results/jevbench/quire-p2-calibrated/`.

### Against other systems, on the same items

JevBench publishes each system's per-item outcomes on the public items, from the maintainers' own runs. `bench/jevbench/public_standing.py --versus` pairs a run against any of them:

| system (*published* outcomes) | weights | standard /72 | hard /111 | total /231 | Quire vs it, paired |
|---|---|---|---|---|---|
| Jev 1.13 (TypeSafe) | closed | 71 | 81 | 200 | 12 better / 26 worse, p = 0.034 |
| SemIf | Qwen3.5-4B, frozen | 71 | 68 | 187 | 13 better / 14 worse, p = 1.0 |
| **Quire (released)** | Qwen3.5-4B, frozen | 65 | 73 | 186 | — |

SemIf is the closest comparison: the same frozen model with a different prompt and readout. Per tier, Quire is 6 items behind on standard (0 better / 6 worse, p = 0.031) and 5 ahead on hard (13 better / 8 worse, p = 0.38). The standard-tier gap is binary policy and answer-adequacy judgements. That is also what the hidden judge tier tests, so Quire's judge-tier accuracy is probably below SemIf's.

### Projection

`bench/jevbench/project.py` applies JevBench's own `composite_v13` formulas: Calibration as the harness computes it (hard-tier ECE plus fidelity on the 10 public probability items), Speed from the measured endpoint (88.1), and Cost from the tokens the endpoint reports (789 per decision at $0.03 per million). The judge tier can't be measured, so the projection is a range over assumed judge accuracy:

| configuration | at judge = 0.85 | at judge = 0.95 |
|---|---|---|
| v0.1.0 (no map; submitted to JevBench) | 74.3 | 75.3 |
| **v0.1.1 (with the per-type map)** | **74.9** | **75.9** |

For reference, on 22 Sep 2026 the published v1.3 scores were Jev 74.4, SemIf 73.1 and djev 73.0. This is a projection from public items, not a score.

*Correction (23 Sep 2026).* Earlier versions of this page reported fidelity TVD 0.343 / 0.349, Calibration axes 73.8 / 76.4 and projections 73.1–74.0 / 73.7–74.7. Our scripts compared yes/no distributions keyed `true`/`false` against gold keyed `yes`/`no`, so the three yes/no probability items always scored TVD 0.5. JevBench's harness receives `yes`/`no` and was not affected. The fixed numbers are above, and the decision to adopt the per-type map is unchanged: the axis still rose.

### What did not help

**One temperature fitted on public items.** Fitted on one random half of the public hard items and scored on the other, over 10 splits (`bench/jevbench/fit_hard_temperature.py`), held-out calibration got worse: 71.3 → 68.4. 55 items are too few, and one temperature can't serve three answer types that need different ones. The per-type map above is fitted on 1,464 held-out items instead.

**Thinking before answering.** Qwen3.5-4B never closed its thought within 256, 512 or 1,024 tokens on the probability items (0 of 10 at every budget). It works through the evidence row by row. Accuracy went 5 → 3 → 3 → 5 of 10.

**More orderings, debiasing.** With the plain prompt (MLX 8-bit), 1, 2, 3 and 5 orderings, with and without contextual debiasing, all landed within one item of each other on the hard tier (every p = 1.0).

**A yes/no word readout.** For binary questions the letter readout sometimes answers by label position: rotating the options flips the answer. Reading the " yes"/" no" logits directly removes that bias, but recovered only 2 of the 10 standard-tier items the plain prompt missed. The model answered the rest wrong for a real reason, not because of position.

### Fine-tuning (LoRA): not part of the released configuration

An adapter trained on kev's decision-v7 set (14.6k short classification items), tested on the public items against the frozen model with the same prompt and orderings, at two strengths α (the LoRA delta scaled by α):

| α | prompt | standard /72 | hard /111 | hard, paired vs frozen | hard ECE |
|---|---|---|---|---|---|
| 1.0 | plain, 2 orderings | 68 | 59 | −13, p = 0.019 | 0.342 |
| 0.25 | plain, 1 ordering | 69 | 71 | −1 | 0.120 |

At full strength the adapter picks up 5 standard-tier items and loses 13 hard ones, and its hard-tier calibration error more than triples. It learned what short classification data teaches and lost ground on long, multi-condition decisions. At a quarter strength both effects mostly disappear. The released configuration uses no adapter; the torch backend can load one (`--adapter`, `--adapter-scale`) for experiments.

## TypeSafe's public 102-row subset

The 102-row subset (20 cases) of TypeSafe's published evaluations that SemIf's benchmark bundle defines, with every published model's per-row answers. The reference label is the argmax of the released models' mean answer, so this measures agreement with a consensus of frontier models, not correctness.

| system | equal-case agreement | correct rows |
|---|---|---|
| Claude Opus 5 (*published*) | 0.912 | |
| GPT-5.6 Sol (*published*) | 0.906 | |
| Jev 1.13 (*published*; recomputed as a check) | 0.883 | |
| SemIf, frozen 4B (*published*) | 0.845 | 87 |
| **Quire, released (MLX 8-bit)** | **0.799** [0.713, 0.880] | **82** |
| Quire, plain prompt, 1 ordering | 0.814 [0.738, 0.887] | 85 |

Quire against Jev's published answers: 3 rows better, 11 worse, p = 0.057. Against SemIf's published answers: 2 better, 7 worse, p = 0.18.

## Wide choices

kev decision-v7 development split, MLX 8-bit, released configuration:

| suite | options | letter readout | per-option fan-out |
|---|---|---|---|
| dbpedia14 | 14 | 0.905 | 0.897 (3 better / 4 worse, p = 1.0), at 6× the compute |
| banking77 | 77 | cannot pose the question | 0.422 (chance 0.013) |

Where both paths apply they are equally accurate, so the fan-out is used only above 26 options, where the letter readout has no answer.

## Decision Index 0.1

Pending. The Decision Index's frozen suite has 132,422 requests. reflex's published full 27B run declared 14,500 of them unsupported because they have more than 26 options, and unsupported requests score zero. Quire's per-option path answers them.
