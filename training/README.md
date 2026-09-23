# Training (optional; not part of the released configuration)

The released Quire configuration is a frozen model. This directory holds the tools used to train and evaluate LoRA adapters on top of it, so that the adapter results in [`docs/RESULTS.md`](../docs/RESULTS.md) can be reproduced and extended. None of these tools is needed to run Quire.

```sh
pip install -e ".[torch,lora,train]"          # training (CUDA)
pip install -e ".[teacher]"                    # teacher labelling through llama.cpp
```

## Contents

| file | what it does |
|---|---|
| `train_lora.py` | LoRA fine-tuning of the readout on any mix of suites. Loss: cross-entropy on the gold option, plus a soft-target term from an exact gold distribution or a teacher distribution (the latter only where the teacher agrees with gold). 8k-token cap, gradient checkpointing, checkpoint and `--resume`. The `lora+pointer` arm trains a bilinear pointer head instead of reading the letter logits. |
| `synth/` | Deterministic generators for long-state decision scenarios in six families: policy documents with amendments and exceptions, multi-hop lookups, evidence-counting probabilities (exact gold distributions), dates and running totals, overlapping routing, and constraint-checkable answer adequacy. Each scenario is sampled as structured facts, answered by an evaluator, then rendered to prose; its twin changes one fact and flips the answer. `--level easy/medium/hard` sets difficulty. |
| `teacher_label_llamacpp.py`, `teacher_label_mlx.py` | Label a suite with a larger frozen model through the same prompt, two orders averaged, resumable. |
| `corpus.py` | Loads kev's decision suites (`KEV_DATA`, a checkout of kev's `evals/`) and the synthetic suites (`SYNTH_DATA`, default `data/synth`) into one shape. |

## The kev adapter

The adapter in `results/jevbench/adapters/` was trained on kev's decision-v7 training split (14.6k short classification items) with the `plain` prompt:

```sh
export KEV_DATA=$PWD/kev/evals
python training/train_lora.py --arm lora+letter --style plain --seed 0 --rank 16 --lr 5e-5 --epochs 2 --accum 8 \
    --suite decision-v7:train --dev-suite decision-v7:development --out runs/lora/kev-s0
```

It took 1 h 52 min on one NVIDIA L40 and reached 0.876 on decision-v7's development split. The trained weights are published as [`aflowx/quire-lora-kev-4b`](https://huggingface.co/aflowx/quire-lora-kev-4b). On JevBench's public items it gains standard-tier items and loses hard-tier ones (see `docs/RESULTS.md`), which is why the released configuration uses no adapter.

## Synthetic data

```sh
python training/synth/build.py --seed 1 --out data/synth/synth-v1                                   # 12k items
python training/synth/build.py --seed 2 --level easy --families multi_hop,temporal_numeric,probability \
    --scale 0.4 --out data/synth/synth-v2-easy
python training/synth/build.py --seed 3 --level medium --families multi_hop,temporal_numeric,probability \
    --scale 0.3 --out data/synth/synth-v2-medium
```

The same seed always produces the same items (`tests/test_synth.py`).
