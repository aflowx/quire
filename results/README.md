# Results

Every file here was produced before this repository existed, in the private working repository where Quire was developed. The code that produced them is the code in `src/quire` and `bench/`, under earlier module names. In those files the `quire` prompt style is recorded under its development name, `r2-test`; `tests/test_prompt.py` pins the rendering. Each row gives the working-repo commit, the configuration, and the hardware. Re-running the command shown with this repository should reproduce the numbers. If it doesn't, please report it.

## JevBench v1.3, 231 public items (`results/jevbench/`)

1 × NVIDIA L40 (48 GB), torch BF16, `Qwen/Qwen3.5-4B` unmodified, one `decide()` per item, serial.

| directory | configuration | easy /48 | standard /72 | hard /111 | hard ECE | commit |
|---|---|---|---|---|---|---|
| `quire-p2/` | **released**: quire prompt, 2 orderings | 48 | 64 | 71 | 0.073 | `b84c351` |
| `quire-p1/` | quire prompt, 1 ordering | 48 | 64 | 71 | 0.118 | `b84c351` |
| `control-plain-prompt-p2/` | plain prompt, 2 orderings | 48 | 63 | 72 | 0.102 | `eecbe37` |
| `adapters/kev-a1.0/` | plain prompt, 2 orderings + kev LoRA | 48 | 68 | 59 | 0.342 | `eecbe37` |
| `adapters/kev-a0.25/` | plain prompt, 1 ordering + kev LoRA at 0.25 | 48 | 69 | 71 | 0.120 | `eecbe37` |

Reproduce: `python bench/jevbench/run_public.py --jevbench <checkout> --backend torch [--permutations 1] [--style plain] [--adapter DIR --adapter-scale A] --out runs/<name>`.

`input_tokens` counts the state once plus every suffix in the `b84c351` files. The `eecbe37` files predate that fix and count the state only, so don't use them for pricing.

## Endpoint speed

`quire-serve --backend torch`, L40, serial requests over the 72 public standard items, HTTP included, three repeats each (`bench/jevbench/speed_probe.py`):

| configuration | p50 | p95 | Speed axis (×2 + 0.15 s) | mean input tokens |
|---|---|---|---|---|
| 2 orderings, batched over the prefix cache | 124 ms | 125 ms | 88.0 | 236 |
| 1 ordering | 58 ms | 58 ms | 91.5 | 152 |

Through the server the standard tier scores 65/72, against 64/72 for the run above. The server batches both orderings over a cached prefix while the runner reads each full prompt, and the resulting numerical difference moves one item. JevBench measures Speed on its own GPU, so these latencies don't transfer.

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
