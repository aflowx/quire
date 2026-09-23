"""Does sharing the question text across orderings change any answer?

    python bench/checks/share_question_equivalence.py --jevbench $JEVBENCH --out runs/checks/share-mlx.json

Runs JevBench's 231 public items through the engine twice, share_question off
(v0.1.1's path) and on, and compares: max |delta p| per item, argmax flips
(with the old path's top-two margin for any flip) and the reported tokens. The
inputs are token-identical by construction; this measures what the changed
cache split does numerically.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "jevbench"))

import tasks as jb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", type=pathlib.Path, required=True)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if args.backend == "mlx":
        from quire.engine import Engine
        eng = Engine(model_repo="mlx-community/Qwen3.5-4B-MLX-8bit")
    else:
        from quire.torch_engine import TorchEngine
        eng = TorchEngine()
    rows, flips, tok_old, tok_new, dmax = [], 0, 0, 0, 0.0
    for t in jb.load(args.jevbench):
        q = t.to_question()
        eng.share_question = False
        a = eng.decide(t.state, [q])[0]
        eng.share_question = True
        b = eng.decide(t.state, [q])[0]
        d = max(abs(a.probabilities[k] - b.probabilities[k]) for k in a.probabilities)
        top = sorted(a.probabilities.values(), reverse=True)
        flip = int(a.choice != b.choice)
        flips += flip
        dmax = max(dmax, d)
        to, tn = a.state_tokens + a.suffix_tokens, b.state_tokens + b.suffix_tokens
        tok_old += to
        tok_new += tn
        rows.append({"id": t.id, "tier": t.tier, "max_abs_dp": d, "flip": flip,
                     "old_margin": top[0] - top[1] if len(top) > 1 else 1.0, "tokens_old": to, "tokens_new": tn})
    n = len(rows)
    summary = {"items": n, "argmax_flips": flips, "max_abs_dp": dmax,
               "flip_margins": [r["old_margin"] for r in rows if r["flip"]],
               "tokens_per_item_old": tok_old / n, "tokens_per_item_new": tok_new / n}
    print(json.dumps(summary, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
