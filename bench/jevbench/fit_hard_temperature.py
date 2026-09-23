"""Temperature on the hard tier, cross-validated, scored with the harness formula.

Calibration is measured on the hard tier only: ECE plus fidelity to the
probability items' gold distributions. A temperature is monotone, so it
changes no answer; it can only move this axis. Fit and evaluation folds are
disjoint halves of the public hard items, so the held-out number is what
the hidden half would see.

    JEVBENCH=/path/to/jevbench python bench/jevbench/fit_hard_temperature.py results/jevbench/quire-p2 ...
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tasks import harness_labels  # noqa: E402

import os  # noqa: E402
JEVBENCH = os.environ.get("JEVBENCH") or sys.exit("set JEVBENCH to a checkout of github.com/fstandhartinger/jevbench")
sys.path.insert(0, JEVBENCH)
from jevbench import composite_v13 as cv  # noqa: E402

GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]


def temper(p, T):
    z = {k: math.log(max(v, 1e-9)) / T for k, v in p.items()}
    m = max(z.values()); e = {k: math.exp(v - m) for k, v in z.items()}; s = sum(e.values())
    return {k: v / s for k, v in e.items()}


def ece(rs, T, bins=10):
    tot = 0.0
    for b in range(bins):
        sub = [r for r in rs if b / bins < max(temper(r["probs"], T).values()) <= (b + 1) / bins]
        if sub:
            conf = sum(max(temper(r["probs"], T).values()) for r in sub) / len(sub)
            acc = sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in sub) / len(sub)
            tot += len(sub) / len(rs) * abs(conf - acc)
    return tot


def tvd(rs, T, gold):
    xs = [0.5 * sum(abs(harness_labels(temper(r["probs"], T)).get(l, 0) - gold[r["id"]][l]) for l in gold[r["id"]])
          for r in rs if r["id"] in gold]
    return sum(xs) / len(xs)


def axis(rs, T, gold):
    return cv.calibration(ece(rs, T), tvd(rs, T, gold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--folds", type=int, default=10, help="random half-splits to average")
    args = ap.parse_args()
    gold = {}
    for line in open(pathlib.Path(JEVBENCH) / "datasets/public/hard.jsonl"):
        r = json.loads(line)
        if r["family"] == "probability":
            gold[r["id"]] = r["provenance"]["gold_probs"]
    for path in args.runs:
        hard = [r for r in json.loads((path / "run.json").read_text())["records"] if r["tier"] == "hard"]
        conf = sum(max(r["probs"].values()) for r in hard) / len(hard)
        acc = sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in hard) / len(hard)
        best_all = max(GRID, key=lambda T: axis(hard, T, gold))
        held = []
        for seed in range(args.folds):
            ids = [r["id"] for r in hard]; random.Random(seed).shuffle(ids); half = set(ids[: len(ids) // 2])
            fit = [r for r in hard if r["id"] in half]; ev = [r for r in hard if r["id"] not in half]
            T = max(GRID, key=lambda T: axis(fit, T, gold))
            held.append((axis(ev, 1.0, gold), axis(ev, T, gold), T))
        base = sum(h[0] for h in held) / len(held); after = sum(h[1] for h in held) / len(held)
        print(f"{path.name:26s} conf {conf:.3f} acc {acc:.3f} | T=1: {axis(hard, 1.0, gold):5.1f} | "
              f"best T on all {best_all:.2f}: {axis(hard, best_all, gold):5.1f} | "
              f"held-out {base:5.1f} -> {after:5.1f} (T fitted: {sorted(set(h[2] for h in held))})")


if __name__ == "__main__":
    main()
