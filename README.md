# Quire

**Typed decisions from a frozen language model: read the state once, answer every question from that one read.**

You send Quire a *state* (a document, a ticket, a log, a JSON object) and a set of typed questions: pick one of these options, yes or no, rate on this scale. It returns a probability for every option of every question. It generates no text. Each answer is read from the model's next-token logits at a single position, so the output always matches its type.

Quire speaks the System One wire format (`POST /v1/systemone`), so JevBench and the Decision Index can run their own harnesses against it unchanged. It runs on NVIDIA GPUs (PyTorch) and on Apple Silicon (MLX).

Quire is independent and not affiliated with TypeSafe AI.

## What makes it different

- **One read, many questions.** The state is prefilled once, and every question, plus every ordering of its options, is a short suffix batched over that prefilled cache. On a 1,431-token document with 40 questions that is 0.64 s, **20.7× faster** than asking the questions one at a time (MLX, M5 Max). The more questions share a state, the bigger the win.
- **Any number of options, none refused.** Choices with more options than the 26-letter label pool are answered by asking one yes/no sub-question per option in the same fan-out and normalising the answers into one distribution, a two-stage "score independently, then choose" design. Nothing is truncated or filtered. Where both paths apply they are equally accurate (dbpedia14, 14 options: 0.905 letter readout vs 0.897, p = 1.0); above 26 options the letter readout has no answer at all.
- **A frozen model, deliberately.** The released configuration trains nothing. An adapter fine-tuned on short classification data gained binary-judgement items but lost 13 of JevBench's hard items and tripled its calibration error. The details are in [`docs/RESULTS.md`](docs/RESULTS.md).
- **Order-averaged probabilities.** Each question is read with its options in two orders and the distributions are averaged. On JevBench's hard tier that lowers calibration error from 0.113 to 0.090 without fitting a temperature. (Fitted temperatures made held-out calibration worse.)

## How it compares

The released configuration is `Qwen/Qwen3.5-4B`, unmodified, with Quire's own prompt: a system instruction that frames each question as a test run against the state. On JevBench's 231 public items, paired against the per-item outcomes JevBench publishes for other systems:

- **Jev 1.13** (TypeSafe, closed): Quire is behind, 12 items better and 26 worse (p = 0.034).
- **SemIf** (the same frozen 4B, a different prompt and readout): not distinguishable in total (13 better / 14 worse, p = 1.0). Quire is 6 items behind on binary policy and adequacy judgements and 5 ahead on the hard tier.

## Quick start

```sh
# NVIDIA
pip install -e ".[torch]"
quire-serve --backend torch --model Qwen/Qwen3.5-4B --port 8778

# Apple Silicon
pip install -e ".[mlx]"
quire-serve --backend mlx --model mlx-community/Qwen3.5-4B-MLX-8bit --port 8778
```

```sh
curl -s localhost:8778/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Policy: trial accounts may export CSV; PDF export needs a paid plan. A trial user asks to export CSV.",
  "questions": {
    "allowed": {"type": "noul", "instructions": "Is the requested export permitted under the policy?"},
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"support": "Product support", "billing": "Billing", "sales": "Sales"}}
  }
}'
```

In Python:

```python
from quire.torch_engine import TorchEngine          # or: from quire.engine import Engine (MLX)
from quire.schema import Question

engine = TorchEngine()                               # Qwen/Qwen3.5-4B, quire prompt, 2 orderings
answers = engine.decide(state, [
    Question("Is the requested export permitted?", {"true": "permitted", "false": "not permitted"}, kind="noul"),
    Question("Which team should handle this?", {"support": "Product support", "billing": "Billing", "sales": "Sales"}),
])
answers[0].probabilities   # e.g. {'true': 0.90, 'false': 0.10}
```

## Results

| benchmark | what | Quire |
|---|---|---|
| JevBench v1.3, 231 public items | easy / standard / hard accuracy | 1.000 / 0.903 / 0.658 |
| JevBench, endpoint on 1 × L40 | serial p50 latency, 2 orderings | 122 ms (Speed axis 88.1) |
| TypeSafe public 102-row subset | agreement with the released models' consensus answer | 0.799 [0.713, 0.880] (Jev: 0.883) |
| Decision Index 0.1 | index | pending |

These are our own measurements on public items. JevBench's official score includes 303 held-out decisions that only its maintainers can run. Every number links to a result file, the configuration that produced it and the command that reproduces it in [`results/README.md`](results/README.md).

## Repository layout

```
src/quire/            the engine
  engine.py           MLX backend (Apple Silicon)
  torch_engine.py     PyTorch backend (CUDA): batched fan-out over the prefix cache
  prompt.py           prompt rendering: "quire" (default) and "plain" styles
  fanout.py           shared-state prefill and suffix batching (MLX)
  wide.py             more options than labels: per-option fan-out
  ensemble.py         averaging across option orderings
  daemon.py           quire-serve: /v1/systemone, /decide
  adapters/           Decision Index engine adapter
bench/
  jevbench/           public-item runner, the harness's scoring formulas, paired comparisons, speed probe
  decision_index/     suite verification against published hashes
  wide/               wide-choice validation on kev's suites
results/              every result file, with its configuration and provenance
docs/                 method, results and reproduction notes
```

## Documentation

- [`docs/METHOD.md`](docs/METHOD.md): how a decision is read, the fan-out, order averaging, wide choices
- [`docs/RESULTS.md`](docs/RESULTS.md): all results, including the ones that didn't work
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md): exact commands for every number

## License

MIT, see [`LICENSE`](LICENSE). Third-party work and attributions are in [`NOTICE`](NOTICE).
