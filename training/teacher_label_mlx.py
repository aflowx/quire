"""The Mac shard of the teacher (docs/PLAN.md W2): same output as teacher_label.py, via the MLX engine.

    python training/teacher_label_mlx.py --model mlx-community/Qwen3.8-27B-8bit --suite synth-v1 --split train --out runs/teacher/synth-train.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quire.engine import Engine  # noqa: E402
from quire.schema import Question  # noqa: E402
import corpus  # noqa: E402



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-8bit", help="teacher: Hub id or local path")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    items = corpus.load(args.suite, args.split)
    if args.source:
        items = [i for i in items if i.source in args.source]
    items = [i for i in items if len(i.option_ids) <= 26]  # single-token letter pool; kev has 77-option items
    if args.limit:
        items = items[: args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(l)["uid"] for l in open(args.out) if l.strip()}
        print(f"resuming: {len(done)} already labelled", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine = Engine(model_repo=args.model, n_permutations=2, use_debias=False, prompt_style="quire")
    print(f"{len(items)} items", flush=True)
    started = time.perf_counter(); n = agree = 0
    with open(args.out, "a") as f:
        for it in items:
            if it.uid in done:
                continue
            q = Question(instructions=it.instructions, criteria={o: (d or o) for o, d in zip(it.option_ids, it.descriptions)}, kind="choice")
            a = engine.decide(it.state, [q])[0]
            gold = it.option_ids[it.label]
            ok = a.choice == gold; agree += ok; n += 1
            f.write(json.dumps({"uid": it.uid, "source": it.source, "kind": it.kind, "gold": gold,
                                "teacher_probs": {k: float(v) for k, v in a.probabilities.items()},
                                "per_order": [{k: float(v) for k, v in d.items()} for d in a.per_permutation],
                                "agree": bool(ok), "input_tokens": a.state_tokens + a.suffix_tokens}) + "\n")
            if n % 100 == 0:
                el = time.perf_counter() - started
                print(f"  [{el:6.0f}s] {n} done, {el / n:.2f} s/item, agrees {agree / n:.3f}", flush=True)
    el = time.perf_counter() - started
    print(f"done: {n} items in {el:.0f}s ({el / max(n, 1):.2f} s/item); agreement {agree / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
