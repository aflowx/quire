"""Raw yes/no readouts for the "no"-lean study (pre-registered 23 Sep 2026).

    KEV_DATA=... SYNTH_DATA=... python bench/yesno/readouts.py --out runs/yesno/readouts.jsonl

For every held-out yes/no item, several suffixes are read over the item's own
state prefix in one fan-out, plus the released suffixes over the content-free
state. Each readout is stored as the two raw logits [true-side, false-side], so
every variant (bench/yesno/variants.py) is computed from the same numbers:

  R1/R2  released rendering, option order (true, false) / (false, true)
  R3/R4  polarity-flipped test ("is the correct answer 'no'?"), same two orders
  R5     word readout: the test, then "Answer yes or no.", yes/no word tokens
  R6/R7  options rendered as plain Yes / No, both orders
  C1/C2  R1/R2 against the content-free state "N/A"

Resumable: items already in --out are skipped.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "training"))

import corpus  # noqa: E402

from quire.debias import CONTENT_FREE_STATE  # noqa: E402
from quire.engine import Engine  # noqa: E402
from quire.fanout import batch_fanout  # noqa: E402
from quire.labels import label_token_ids  # noqa: E402
from quire.prompt import _quire_frame, question_suffix, state_prefix  # noqa: E402
from quire.schema import Question  # noqa: E402

SUITES = ["synth-v1:dev", "synth-v2-easy:dev", "synth-v2-medium:dev", "decision-v7:development", "helpsteer2:dev"]
NEGATE = 'Is the correct answer to this question "no"? Question: {q}'


def word_ids(tok, words):
    ids = []
    for w in words:
        enc = tok.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suite", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many new items (smoke test)")
    args = ap.parse_args()
    eng = Engine(model_repo="mlx-community/Qwen3.5-4B-MLX-8bit")
    tok, model = eng.tokenizer, eng.model
    AB = label_token_ids(tok, ["A", "B"], bare=True)
    YES, NO = word_ids(tok, ["yes", "Yes", "YES"]), word_ids(tok, ["no", "No", "NO"])
    _, tail = _quire_frame(tok)

    def released(q, order):
        return tok.encode(question_suffix(tok, q, order, ["A", "B"], single_turn=eng.single_turn, style="quire"),
                          add_special_tokens=False)

    def raw(text):
        return tok.encode(text + tail, add_special_tokens=False)

    cf_ids = tok.encode(state_prefix(tok, CONTENT_FREE_STATE, single_turn=eng.single_turn, style="quire"))
    done = set()
    if args.out.exists():
        done = {json.loads(l)["uid"] for l in open(args.out) if l.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, n = time.perf_counter(), 0
    with open(args.out, "a") as f:
        for spec in args.suite or SUITES:
            suite, split = spec.split(":")
            for it in corpus.load(suite, split):
                if it.kind != "noul" or it.uid in done or set(it.option_ids) != {"true", "false"}:
                    continue
                crit = {o: (d or o) for o, d in zip(it.option_ids, it.descriptions)}
                q = Question(instructions=it.instructions, criteria=crit, kind="choice")
                qn = Question(instructions=NEGATE.format(q=it.instructions), criteria=crit, kind="choice")
                yn = Question(instructions=it.instructions, criteria={"true": "Yes", "false": "No"}, kind="choice")
                tf, ft = ["true", "false"], ["false", "true"]
                suffixes = {
                    "R1": released(q, tf), "R2": released(q, ft),
                    "R3": released(qn, tf), "R4": released(qn, ft),
                    "R5": raw(f"<test>{it.instructions}</test>\nAnswer yes or no."),
                    "R6": released(yn, tf), "R7": released(yn, ft),
                }
                state_ids = tok.encode(state_prefix(tok, it.state, single_turn=eng.single_turn, style="quire"))
                names = list(suffixes)
                rows = batch_fanout(model, state_ids, [suffixes[k] for k in names], batch_size=eng.batch_size)
                cf = batch_fanout(model, cf_ids, [suffixes["R1"], suffixes["R2"]], batch_size=eng.batch_size)
                rec = {"uid": it.uid, "suite": suite, "source": it.source, "gold": it.option_ids[it.label]}
                for k, row in zip(names + ["C1", "C2"], list(rows) + list(cf)):
                    z = np.array(row.astype(mx.float32), dtype=float)
                    if k == "R5":
                        pair = [float(np.logaddexp.reduce(z[YES])), float(np.logaddexp.reduce(z[NO]))]
                    else:
                        # letter A is whichever option the order put first
                        first_true = k in ("R1", "R3", "R6", "C1")
                        a, b = float(z[AB[0]]), float(z[AB[1]])
                        pair = [a, b] if first_true else [b, a]
                    rec[k] = pair
                f.write(json.dumps(rec) + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    break
                if n % 100 == 0:
                    f.flush()
                    print(f"  {n} items, {(time.perf_counter() - started) / n:.2f} s/item", flush=True)
    print(f"done: {n} new items -> {args.out}")


if __name__ == "__main__":
    main()
