"""Does the CUDA fan-out give the same letter logits as one full forward pass?

    KEV_DATA=... SYNTH_DATA=... python bench/checks/fanout_equivalence.py --n 400 --out runs/checks/fanout.json

For each held-out item, every (question, ordering) suffix is scored two ways:

  fan-out  the state prefix prefilled once, its cache (attention KV plus the
           Gated DeltaNet conv/recurrent states) copied and expanded to a batch
           of suffixes, one forward -- exactly TorchEngine's path
  full     prefix + suffix as one sequence, one forward, no cache

and the label-token logits are compared. Items are drawn across state lengths
so that prefix lengths fall at many offsets of the linear-attention chunk size.
Reports the max and quantiles of |delta logit| and how often the argmax flips.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "training"))

import corpus  # noqa: E402

from quire.labels import label_token_ids, pick_labels  # noqa: E402
from quire.prompt import first_token_style, question_suffix, rotate, state_prefix  # noqa: E402
from quire.schema import Question  # noqa: E402
from quire.torch_engine import TorchEngine, expand_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", default=None, help="suite:split (default: synth-v1:dev, decision-v7:development)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    eng = TorchEngine(n_permutations=2)
    tok, model = eng.tokenizer, eng.model
    bare = first_token_style(eng.prompt_style)

    items = []
    for spec in args.suite or ["synth-v1:dev", "decision-v7:development"]:
        s, sp = spec.split(":")
        items += [i for i in corpus.load(s, sp) if len(i.option_ids) <= 26]
    random.Random(0).shuffle(items)
    items = items[: args.n]

    deltas, flips, rows = [], 0, []
    with torch.no_grad():
        for it in items:
            q = Question(instructions=it.instructions,
                         criteria={o: (d or o) for o, d in zip(it.option_ids, it.descriptions)}, kind="choice")
            prefix_ids = tok(state_prefix(tok, it.state, single_turn=eng.single_turn, style=eng.prompt_style),
                             add_special_tokens=False).input_ids
            labels = pick_labels(q.n_options, eng.label_pool)
            ids = label_token_ids(tok, labels, bare=bare)
            orders = [rotate(q.option_ids, k) for k in range(min(2, q.n_options))]
            suffixes = [tok(question_suffix(tok, q, o, labels, single_turn=eng.single_turn, style=eng.prompt_style),
                            add_special_tokens=False).input_ids for o in orders]
            # fan-out: same grouping rule as TorchEngine (equal-length suffixes share a forward)
            cache = model(torch.tensor([prefix_ids]).cuda(), use_cache=True).past_key_values
            fan = {}
            by_len = {}
            for k, s in enumerate(suffixes):
                by_len.setdefault(len(s), []).append(k)
            for _, ks in by_len.items():
                past = copy.deepcopy(cache)
                expand_cache(past, len(ks))
                out = model(torch.tensor([suffixes[k] for k in ks]).cuda(), past_key_values=past,
                            use_cache=True, logits_to_keep=1).logits[:, -1].float()
                for k, row in zip(ks, out):
                    fan[k] = row[ids].cpu().numpy()
            for k, s in enumerate(suffixes):
                full = model(torch.tensor([prefix_ids + s]).cuda(), logits_to_keep=1).logits[0, -1].float()[ids].cpu().numpy()
                d = float(np.abs(fan[k] - full).max())
                flip = int(fan[k].argmax() != full.argmax())
                deltas.append(d)
                flips += flip
                rows.append({"uid": it.uid, "order": k, "prefix_tokens": len(prefix_ids),
                             "offset64": len(prefix_ids) % 64, "batch": len(by_len[len(s)]),
                             "max_abs_delta": d, "argmax_flip": flip})
            del cache
    d = np.array(deltas)
    summary = {"pairs": len(d), "items": len(items), "argmax_flips": flips,
               "max_abs_delta": float(d.max()), "p50": float(np.quantile(d, 0.5)),
               "p99": float(np.quantile(d, 0.99)), "any_nan": bool(np.isnan(d).any()),
               "prefix_offsets_covered": len({r["offset64"] for r in rows})}
    print(json.dumps(summary, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
