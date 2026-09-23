# Reproducing the results

Each section reproduces one block of [`RESULTS.md`](RESULTS.md). GPU numbers were measured on 1 × NVIDIA L40 (48 GB) with PyTorch 2.14 and CUDA 13.0, Apple Silicon numbers on an M5 Max (128 GB).

## Setup

```sh
git clone https://github.com/aflowx/quire && cd quire
pip install -e ".[torch,bench,dev]"      # NVIDIA
pip install -e ".[mlx,bench,dev]"        # Apple Silicon
pytest                                   # the model-loading tests need the MLX weights cached
```

Benchmark data is read from local checkouts and never committed:

```sh
git clone https://github.com/fstandhartinger/jevbench      # JevBench public items and scoring formulas
export JEVBENCH=$PWD/jevbench
```

## JevBench public items

```sh
# the released configuration
python bench/jevbench/run_public.py --jevbench $JEVBENCH --backend torch --out runs/quire-p2

# the controls in results/jevbench/
python bench/jevbench/run_public.py --jevbench $JEVBENCH --backend torch --permutations 1 --out runs/quire-p1
python bench/jevbench/run_public.py --jevbench $JEVBENCH --backend torch --style plain --out runs/plain-p2
```

Paired comparison, calibration and projection:

```sh
python bench/jevbench/compare.py runs/plain-p2 runs/quire-p1 runs/quire-p2 --tier hard
JEVBENCH=$JEVBENCH python bench/jevbench/fit_hard_temperature.py runs/quire-p2
python bench/jevbench/project.py runs/quire-p2/run.json --jevbench $JEVBENCH --speed 88.1
python bench/jevbench/public_standing.py runs/quire-p2/run.json     # every leaderboard system on the same 231 items
```

`project.py` prices Cost from the run's measured input tokens at $0.03 per million, the hosted list price JevBench used for Qwen3.5-4B. Pass `--speed` from the endpoint measurement below.

### Endpoint speed

```sh
quire-serve --backend torch --model Qwen/Qwen3.5-4B --port 8778 &
JEVBENCH=$JEVBENCH python bench/jevbench/speed_probe.py
```

### The benchmark's own harness

The maintainers run submissions through JevBench's `typesafe` adapter against a `/v1/systemone` endpoint. To run it that way yourself, on the public items:

```sh
quire-serve --backend torch --model Qwen/Qwen3.5-4B --host 127.0.0.1 --port 8778 &
cd $JEVBENCH
python -m jevbench.cli run --tasks datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8778 --key-env '' --model quire \
  --cost-basis no_billable_account_public_endpoint --reserve-usd 0 \
  --results RUN/results.jsonl --raw-dir RUN/raw --ledger RUN/ledger.jsonl
python -m jevbench.cli summarize --tasks datasets/public/original.jsonl --results RUN/results.jsonl
```

## Forgetting control and wide choices (kev's suites)

```sh
git clone https://github.com/jaredpalmer/kev && export KEV_DATA=$PWD/kev/evals   # the decision-v7 and transfer-v4 suites
python bench/wide/validate.py                                                   # dbpedia14 and banking77 (MLX)
```

## TypeSafe public 102-row subset

This comparison set is defined by SemIf's benchmark bundle ([`benchmarks/build_typesafe.py`](https://github.com/TheoLeeCJ/SemIf/tree/master/benchmarks)), which rebuilds the 102 rows from TypeSafe's public evaluation payloads and checks each against a pinned hash. The payloads are TypeSafe's and aren't redistributed here. The runner that produced `results/typesafe102/` is not yet in this repository.

## Decision Index 0.1

The frozen suite dataset isn't published yet ([apolinario/decision-index#1](https://github.com/apolinario/decision-index/issues/1)). The kit can rebuild it from pinned public sources:

```sh
git clone https://github.com/apolinario/decision-index && cd decision-index
pip install -e ".[rebuild]"
python -m decision_index suite rebuild --work work --only 1 2 6 11 20 21 22 23 24 25 26 27 28 29 30 31 36 37 40 41 42 43 44 50
```

The index panel is 19 of those ids; 26–29 and 42 are display-only benchmarks that the MMLU and ContractNLI builders also read. The display-only benchmarks don't enter the index. This part is still in progress: the kit's rebuild writes its HTTP and Hub downloads under `work/raw/` but reads them from `work/artifacts/benchmark-suite/raw/`, and we are working around that with symlinks. Then verify every rebuilt request against the payload hashes published with reflex-27b's full run, and run Quire through the kit's runner:

```sh
python bench/decision_index/verify_rows.py work/.../selected-rows.jsonl.gz
python -m decision_index pipeline --engine quire.adapters.decision_index:QuireTorchEngine \
  --rows work/.../selected-rows.jsonl.gz --out runs/quire-di
```

## LoRA adapters (not the released configuration)

The adapter in `results/jevbench/adapters/` is loaded with `--adapter` on the torch backend. Its training code and weights are not in this release yet.
