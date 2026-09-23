"""Run Quire over JevBench's 231 public items and write run.json.

    python bench/jevbench/run_public.py --jevbench /path/to/jevbench --backend torch --out results/jevbench/quire-p2

One decide() call per item, exactly as the /v1/systemone server would make it.
The records keep the per-option probabilities, the tier and family, the
latency, and the input tokens the engine actually read (state once plus every
suffix), which is what JevBench prices. Scores come from bench/jevbench/project.py
and the harness's own formulas, never from this script.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tasks import load, to_harness_probs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", type=pathlib.Path, required=True, help="a checkout of fstandhartinger/jevbench")
    ap.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--permutations", type=int, default=2)
    ap.add_argument("--style", choices=["quire", "plain"], default="quire")
    ap.add_argument("--tier", action="append", default=None, help="restrict to these tiers")
    ap.add_argument("--adapter", default=None, help="optional LoRA (torch); not the released config")
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if (args.out / "run.json").exists():
        sys.exit(f"{args.out} already holds a run; choose a new directory")

    if args.backend == "torch":
        from quire.torch_engine import TorchEngine
        engine = TorchEngine(model_repo=args.model or "Qwen/Qwen3.5-4B", revision=args.revision,
                             n_permutations=args.permutations, prompt_style=args.style,
                             adapter=args.adapter, adapter_scale=args.adapter_scale)
    else:
        from quire.engine import Engine
        engine = Engine(model_repo=args.model or "mlx-community/Qwen3.5-4B-MLX-8bit",
                        n_permutations=args.permutations, use_debias=False, prompt_style=args.style)

    items = load(args.jevbench)
    if args.tier:
        items = [t for t in items if t.tier in args.tier]
    engine.decide(items[0].state, [items[0].to_question()])  # warm-up, not timed

    records = []
    for k, task in enumerate(items):
        t0 = time.perf_counter()
        answer = engine.decide(task.state, [task.to_question()])[0]
        latency = time.perf_counter() - t0
        records.append({
            "id": task.id, "tier": task.tier, "family": task.family, "kind": task.kind,
            "expected": task.expected, "n_options": len(task.option_ids),
            "probs": {o: float(p) for o, p in answer.probabilities.items()},
            "harness_probs": to_harness_probs(task.kind, answer.probabilities),
            "latency_s": latency, "margin": answer.margin, "epistemic": answer.epistemic,
            "input_tokens": answer.state_tokens + answer.suffix_tokens,
        })
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=pathlib.Path(__file__).parent).stdout.strip()
    except OSError:
        commit = None
    provenance = {
        "system": "quire", "backend": args.backend, "model": engine.model_repo, "revision": args.revision,
        "permutations": args.permutations, "prompt_style": args.style, "adapter": args.adapter,
        "adapter_scale": args.adapter_scale if args.adapter else None, "n_tasks": len(records),
        "tiers": sorted({r["tier"] for r in records}), "git_commit": commit,
        "public_subset_only": True, "held_out": "the judge tier (146) and 109 hard items are not public",
        "generated": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({"_provenance": provenance, "records": records}, indent=1))
    by = {}
    for r in records:
        b = by.setdefault(r["tier"], [0, 0])
        b[0] += max(r["probs"], key=r["probs"].get) == r["expected"]; b[1] += 1
    print("  ".join(f"{t} {a}/{n}" for t, (a, n) in sorted(by.items())), f"-> {args.out / 'run.json'}")


if __name__ == "__main__":
    main()
