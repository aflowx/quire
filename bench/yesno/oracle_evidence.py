"""Diagnostic: is the policy "no" lean a failure to FIND evidence, or to REASON over it?

    KEV_DATA=... SYNTH_DATA=... python bench/yesno/oracle_evidence.py --out runs/yesno/oracle-mlx.jsonl

Synthetic states are relevant sections plus distractor sections headed
"General notices" (training/synth/common.py pad_to), so the relevant text is
known exactly. For each synthetic policy / temporal yes/no item, three readings,
each averaged over the two option orders (the released configuration):

  V0  released: the full state
  O1  the state with every distractor section removed (no distraction at all)
  O2  the full state, with the relevant sections repeated as <evidence> right
      before the test (re-reading with a perfect selector)

If O1/O2 recover the true items, selection is the bottleneck and a real
selector is worth building. If they don't, the failure is reasoning over the
clauses, and re-reading can't fix it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "training"))

import corpus  # noqa: E402

from quire.engine import Engine  # noqa: E402
from quire.fanout import batch_fanout  # noqa: E402
from quire.labels import label_token_ids  # noqa: E402
from quire.prompt import question_suffix, state_prefix  # noqa: E402
from quire.schema import Question  # noqa: E402

FAMILIES = {"policy", "temporal_numeric"}


def relevant(state: str) -> str:
    return "\n\n".join(s for s in state.split("\n\n") if "General notices" not in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--suite", action="append", default=None)
    args = ap.parse_args()
    eng = Engine(model_repo="mlx-community/Qwen3.5-4B-MLX-8bit")
    tok, model = eng.tokenizer, eng.model
    AB = label_token_ids(tok, ["A", "B"], bare=True)

    def suffix(q, order, evidence=None):
        s = question_suffix(tok, q, order, ["A", "B"], single_turn=eng.single_turn, style="quire")
        if evidence:
            s = f"<evidence>\n{evidence}\n</evidence>\n" + s
        return tok.encode(s, add_special_tokens=False)

    def p_true(state, sfx):
        """Mean p(true) over consecutive (A=true, A=false) suffix pairs, one fan-out; one value per pair."""
        rows = batch_fanout(model, tok.encode(state_prefix(tok, state, single_turn=eng.single_turn, style="quire")),
                            sfx, batch_size=eng.batch_size)
        ps = []
        for k, row in enumerate(rows):
            z = np.array(row.astype(mx.float32), dtype=float)
            a, b = z[AB[0]], z[AB[1]]
            pa = 1 / (1 + np.exp(b - a))          # P(letter A)
            ps.append(pa if k % 2 == 0 else 1 - pa)  # even: A = true; odd: A = false
        return [float((ps[i] + ps[i + 1]) / 2) for i in range(0, len(ps), 2)]

    done = set()
    if args.out.exists():
        done = {json.loads(l)["uid"] for l in open(args.out) if l.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "a") as f:
        for spec in args.suite or ["synth-v1:dev", "synth-fresh:train", "synth-fresh:dev"]:
            s, sp = spec.split(":")
            for it in corpus.load(s, sp):
                if it.kind != "noul" or it.source not in FAMILIES or it.uid in done:
                    continue
                crit = {o: (d or o) for o, d in zip(it.option_ids, it.descriptions)}
                q = Question(instructions=it.instructions, criteria=crit, kind="choice")
                tf, ft = ["true", "false"], ["false", "true"]
                ev = relevant(it.state)
                rec = {"uid": it.uid, "source": it.source, "gold": it.option_ids[it.label],
                       "chars_full": len(it.state), "chars_relevant": len(ev),
                       "O1": p_true(ev, [suffix(q, tf), suffix(q, ft)])[0]}
                # V0 and O2 share the full-state prefix: one prefill, four suffixes
                rec["V0"], rec["O2"] = p_true(it.state, [suffix(q, tf), suffix(q, ft), suffix(q, tf, ev), suffix(q, ft, ev)])
                f.write(json.dumps(rec) + "\n")
                n += 1
                if n % 50 == 0:
                    f.flush()
                    print(f"  {n} items", flush=True)
    print(f"done: {n} items -> {args.out}")


if __name__ == "__main__":
    main()
