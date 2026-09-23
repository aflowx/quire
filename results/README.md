# Results

The released-configuration JevBench files (`quire-p2`, `quire-p1`, `harness-typesafe-adapter`) were produced by this repository's own code at commit `8819722`. The other files were produced earlier, in the private working repository where Quire was developed, with the same code under earlier module names; there the `quire` prompt style is recorded under its development name `r2-test`, and `tests/test_prompt.py` pins the rendering. Files produced before commit `d3a3157` rendered JSON-object states (35 of the public hard items) with Python's default formatting rather than as JSON. Each row gives the commit, the configuration, and the hardware. Re-running the command shown with this repository should reproduce the numbers. If it doesn't, please report it.

## JevBench v1.3, 231 public items (`results/jevbench/`)

1 × NVIDIA L40 (48 GB), torch BF16, `Qwen/Qwen3.5-4B` unmodified, one `decide()` per item, serial.

| directory | configuration | easy /48 | standard /72 | hard /111 | hard ECE | commit |
|---|---|---|---|---|---|---|
| `quire-p2/` | **released**: quire prompt, 2 orderings | 48 | 65 | 73 | 0.090 | `8819722` |
| `quire-p1/` | quire prompt, 1 ordering | 48 | 64 | 70 | 0.113 | `8819722` |
| `control-plain-prompt-p2/` | plain prompt, 2 orderings | 48 | 63 | 72 | 0.102 | `eecbe37` |
| `adapters/kev-a1.0/` | plain prompt, 2 orderings + kev LoRA | 48 | 68 | 59 | 0.342 | `eecbe37` |
| `adapters/kev-a0.25/` | plain prompt, 1 ordering + kev LoRA at 0.25 | 48 | 69 | 71 | 0.120 | `eecbe37` |

`harness-typesafe-adapter/` holds the same released configuration measured through JevBench's own harness (`jevbench.cli run --adapter typesafe`, commit `f79a1ca`) against `quire-serve --backend torch`: 231/231 valid, easy 48, standard 65, hard 73, identical to `quire-p2`.

Reproduce: `python bench/jevbench/run_public.py --jevbench <checkout> --backend torch [--permutations 1] [--style plain] [--adapter DIR --adapter-scale A] --out runs/<name>`.

`input_tokens` counts the state once plus every suffix in the `8819722` files (mean 789 per decision across the 231 items, as reported by the endpoint). The `eecbe37` files predate that fix and count the state only, so don't use them for pricing.

## Per-answer-type temperature (`results/calibration/`, `results/jevbench/quire-p2-calibrated/`)

`heldout-predictions.jsonl`: the released configuration's full distributions on 2,930 held-out development items (synth-v1, synth-v2-easy, synth-v2-medium, kev decision-v7), 1 × L40, BF16, from `bench/calibration/predict.py`. `fit.json`: the output of `bench/calibration/fit.py` (fitted temperatures, check-half ECE, the one-time JevBench confirmation). `quire-p2-calibrated/run.json` is `quire-p2/run.json` with the map applied to every distribution; answers are identical by construction. The procedure was pre-registered before any prediction.

## Endpoint speed

`quire-serve --backend torch`, L40, serial requests over the 72 public standard items, HTTP included, three repeats each (`bench/jevbench/speed_probe.py`):

| configuration | p50 | p95 | Speed axis (×2 + 0.15 s) | mean input tokens |
|---|---|---|---|---|
| 2 orderings, batched over the prefix cache | 122 ms | 124 ms | 88.1 | 236 |
| 1 ordering | 58 ms | 58 ms | 91.5 | 152 |

Measured with commit `8819722`. JevBench measures Speed on its own GPU, so these latencies don't transfer.

## kev transfer-v4 development split (`results/transfer-v4/`)

764 items, same hardware and configuration as `quire-p2`, commit `b84c351`: 0.732.

## TypeSafe public 102-row subset (`results/typesafe102/`)

Rows rebuilt from TypeSafe's public evaluation payloads, which are not redistributed here. The reference label is the argmax of the released models' mean answer. Jev's published answers recompute to its published 0.8831 as a check. MLX, `Qwen3.5-4B-MLX-8bit`.

| file | configuration | equal-case agreement | correct rows | commit |
|---|---|---|---|---|
| `quire-p2-mlx.json` | released | 0.799 [0.713, 0.880] | 82/102 | `b84c351` |
| `plain-direct-mlx.json` | plain prompt, 1 ordering | 0.814 [0.738, 0.887] | 85/102 | `46db303` |

## Wide choices (`results/wide/validate-mlx.log`)

`bench/wide/validate.py`, kev decision-v7 development split, MLX 8-bit, released configuration, commit `b84c351`:

- dbpedia14 (14 options): letter readout 0.905, per-option fan-out forced on 0.897 (3 better / 4 worse, p = 1.0), at 6× the compute.
- banking77 (77 options): per-option fan-out 0.422, against a chance level of 0.013.

## Decision Index 0.1 (`results/decision_index/`)

Pending.
