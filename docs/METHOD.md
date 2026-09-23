# Method

## A decision is one position's logits

A request is a *state* plus typed questions. Every type reduces to "score each of N declared options":

| type | options | returned |
|---|---|---|
| `choice` | the declared options | a distribution over them |
| `noul` (yes/no) | `true`, `false` | the probability of `true` |
| `score` | the ordered levels | a distribution over levels and its mean |

Each option is rendered against a label (`A`, `B`, …). The prompt ends where the model's next token is the answer, and the answer is the softmax over the logits of the label tokens only. There is no decoding loop and no parsing, so an answer can't be malformed.

## The prompt (`prompt.py`)

The default `quire` style:

- **System instruction:** *"Each question is a test to run against the state. Work out whether the state passes it, then give the letter of the option that matches the result. Letter only."*
- **User turn:** `<state>…</state>`, then `<test>…</test>` holding the question, then `<options>` with one `A. …` line per option.
- **Answer:** the chat template's thinking block is closed, and the answer is read as the first assistant token, scored over bare letters (`A`, `B`, …).

The `plain` style is the earlier rendering: a single user turn with the state in `<state>` tags, the question and lettered options, ending in "Answer:" and scored over space-prefixed letters. The two styles are within one item of each other on every JevBench tier.

The `quire` wording was chosen among three candidates on held-out synthetic policy and adequacy items. On those items, framing the question as a test applied to the state scored higher than framing it as a question asked about the state (381 vs 354 of 482).

## One read, many questions (`fanout.py`, `torch_engine.py`)

The prompt is split at the one point where everything before it is identical for every question:

- **prefix:** the system instruction, the chat header and the state, through `</state>`. Prefilled once per request.
- **suffix:** the test, the lettered options and the chat tail. One per question per option ordering, typically 20–200 tokens.

Qwen3.5 is a hybrid architecture: three of every four layers are linear attention (Gated DeltaNet) with a fixed-size recurrent state, the fourth is full attention with a KV cache. Both kinds of state are captured after the prefix and replicated across a batch of suffixes, which run in one forward pass (at most 32 per batch on CUDA, grouped by length so no padding is needed). The work per question is therefore proportional to its suffix, not to the state.

Measured on MLX (M5 Max), one 1,431-token document and 40 eight-token questions: 0.64 s fanned out against 13.26 s asked one at a time (20.7×). With a 12-token state and 66-token questions the same comparison gives 3.1×: the win scales with how much of each request is shared.

The token count a request reports (`usage.input_tokens`) is the state once plus every suffix, which is what was computed.

## Order averaging (`ensemble.py`)

A small model's letter probabilities depend on which letter an option gets. Every question is read under `n` cyclic rotations of its options (default 2), each distribution is mapped back to option order, and the mean is returned. Averaging costs one extra suffix per question, not a second read of the state. Two by-products come out of the spread across orderings:

- `epistemic`: disagreement between orderings (the model is unsure *which* answer, not just how sure)
- `aleatoric`: the normalised entropy of the mean distribution

On JevBench's public hard tier, two orderings reduced calibration error from 0.113 to 0.090 and added 3 items (9 better / 6 worse, not significant). We fitted temperatures on held-out splits too. They made calibration worse on the other half, so none is applied.

## More options than labels (`wide.py`)

The label pool has 26 single-token letters. A choice with more options is expanded into one yes/no sub-question per option ("Candidate under consideration: …, is this the correct choice?"). The sub-questions go into the same fan-out, and the yes-probabilities are normalised into one distribution over the original options. With it, the same frozen model answers a 255-option question without truncation, option filtering or refusal.

This costs one suffix per option. On kev's dbpedia14 (14 options), where both paths apply, the two are equally accurate (letter readout 0.905, per-option 0.897, p = 1.0), and the per-option path costs about six times the compute. So the letter readout is used up to 26 options and the per-option path only above that.

## What the released configuration does not do

- **No training.** The weights are `Qwen/Qwen3.5-4B` as published.
- **No per-benchmark prompts.** One rendering for every request.
- **No thinking.** The 4B's thoughts don't close within 1,024 tokens on hard items, and cutting them short measured worse than not thinking.
- **No calibration file.** See above.

LoRA adapters can be loaded on the torch backend (`--adapter`, `--adapter-scale`) for experiments. None is part of the released configuration; see [`RESULTS.md`](RESULTS.md) for why.
