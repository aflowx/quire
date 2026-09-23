"""Full distributions for held-out decision suites, for fitting a calibration map.

    KEV_DATA=... SYNTH_DATA=... python bench/calibration/predict.py --backend torch \\
        --suite synth-v1:dev --suite decision-v7:development --out runs/calibration/predictions.jsonl

Every question goes to the engine exactly as quire-serve would send it (the
answer type is kept only as a label for fitting), with the released prompt
and orderings. Resumable: items already in --out are skipped.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "training"))

import corpus  # noqa: E402

from quire.schema import Question  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--permutations", type=int, default=2)
    ap.add_argument("--suite", action="append", required=True, help="suite:split")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter (torch backend)")
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--max-options", type=int, default=26,
                    help="skip wider choices; they take the per-option path, not the letter readout")
    args = ap.parse_args()
    if args.backend == "torch":
        from quire.torch_engine import TorchEngine
        engine = TorchEngine(model_repo=args.model or "Qwen/Qwen3.5-4B", revision=args.revision,
                             n_permutations=args.permutations,
                             adapter=args.adapter, adapter_scale=args.adapter_scale)
    else:
        from quire.engine import Engine
        engine = Engine(model_repo=args.model or "mlx-community/Qwen3.5-4B-MLX-8bit",
                        n_permutations=args.permutations, use_debias=False)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["uid"] for l in open(args.out) if l.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started, n = time.perf_counter(), 0
    with open(args.out, "a") as f:
        for spec in args.suite:
            suite, split = spec.split(":")
            for it in corpus.load(suite, split):
                if it.uid in done or len(it.option_ids) > args.max_options:
                    continue
                q = Question(instructions=it.instructions,
                             criteria={o: (d or o) for o, d in zip(it.option_ids, it.descriptions)}, kind="choice")
                a = engine.decide(it.state, [q])[0]
                f.write(json.dumps({"uid": it.uid, "suite": it.suite, "source": it.source, "kind": it.kind,
                                    "gold": it.option_ids[it.label], "probs": a.probabilities,
                                    "soft": list(it.soft) if it.soft else None}) + "\n")
                n += 1
                if n % 200 == 0:
                    f.flush()
                    print(f"  {n} items, {(time.perf_counter() - started) / n:.2f} s/item", flush=True)
    print(f"done: {n} new items -> {args.out}")


if __name__ == "__main__":
    main()
