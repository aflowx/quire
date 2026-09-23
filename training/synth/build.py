"""Generate the synthetic suite (docs/PLAN.md W1).

    python training/synth/build.py --seed 1 --out data/synth/synth-v1

Writes train.jsonl / dev.jsonl in kev's row shape plus gold_probs and _meta.
The dev split is by scenario index (every 10th base scenario and its twin),
so a twin never straddles the split. Also writes manifest.json with counts.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from synth import adequacy, multi_hop, policy, probability, routing, temporal  # noqa: E402
from synth.common import rng_for  # noqa: E402

FAMILIES = {
    # family: (module, n_base, long_every)  long_every=k -> 1 in k are long; 0 -> never
    "policy": (policy, 1500, 1.4),
    "multi_hop": (multi_hop, 1500, 1.6),
    "probability": (probability, 1500, 3),
    "temporal_numeric": (temporal, 1200, 3),
    "routing": (routing, 1000, 3),
    "adequacy": (adequacy, 1200, 0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/synth/synth-v1"))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every family's count")
    ap.add_argument("--level", default="hard", choices=["easy", "medium", "hard"],
                    help="difficulty for the families that have a knob (multi_hop, temporal_numeric, probability)")
    ap.add_argument("--families", default=None, help="comma-separated subset")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    tokens = collections.defaultdict(list)
    files = {"train": open(args.out / "train.jsonl", "w"), "dev": open(args.out / "dev.jsonl", "w")}
    for fam, (mod, n, long_every) in FAMILIES.items():
        if args.families and fam not in args.families.split(","):
            continue
        n = int(n * args.scale)
        for i in range(n):
            r = rng_for(args.seed, fam, i)
            long = bool(long_every) and (i % 10) < (10 / long_every)
            split = "dev" if i % 10 == 0 else "train"
            kw = {"level": args.level} if fam in ("multi_hop", "temporal_numeric", "probability") else {}
            for s in mod.build(r, i, args.seed, long=long, **kw):
                files[split].write(json.dumps(s.row(), ensure_ascii=False) + "\n")
                counts[(fam, split, "twin" if s.twin_of else "base")] += 1
                tokens[fam].append(s.n_tokens_est)
    for f in files.values():
        f.close()
    summary = {fam: {"n": sum(v for k, v in counts.items() if k[0] == fam),
                     "train": sum(v for k, v in counts.items() if k[0] == fam and k[1] == "train"),
                     "dev": sum(v for k, v in counts.items() if k[0] == fam and k[1] == "dev"),
                     "twins": sum(v for k, v in counts.items() if k[0] == fam and k[2] == "twin"),
                     "tokens_p50": sorted(tokens[fam])[len(tokens[fam]) // 2],
                     "tokens_max": max(tokens[fam]),
                     "frac_over_2k": sum(t > 2000 for t in tokens[fam]) / len(tokens[fam])}
               for fam in FAMILIES if tokens[fam]}
    total = sum(v["n"] for v in summary.values())
    (args.out / "manifest.json").write_text(json.dumps({"seed": args.seed, "total": total, "families": summary}, indent=1))
    for fam, v in summary.items():
        print(f"{fam:16s} n {v['n']:5d} (train {v['train']}, dev {v['dev']}, twins {v['twins']})  tokens p50 {v['tokens_p50']:5d} max {v['tokens_max']:5d}  >2k {v['frac_over_2k']:.0%}")
    print(f"total {total}")


if __name__ == "__main__":
    main()
