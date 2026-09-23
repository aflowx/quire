"""Paired comparison of JevBench runs on the items they share.

    .venv/bin/python scripts/compare_runs.py results/jevbench/base-p2 results/jevbench/hard-p3 ...

The first run is the control. For every other run: accuracy, ECE, fidelity
TVD, reported input tokens, and a paired sign-flip test on per-item
correctness against the control, over the shared items only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from quire.stats import paired_permutation_test

import os

JEVBENCH = pathlib.Path(os.environ.get("JEVBENCH") or sys.exit("set JEVBENCH to a checkout of github.com/fstandhartinger/jevbench"))


def load(path: pathlib.Path) -> dict:
    blob = json.loads((path / "run.json").read_text())
    return {r["id"]: r for r in blob["records"]}


def correct(r: dict) -> int:
    return int(max(r["probs"], key=r["probs"].get) == r["expected"])


def ece(rs, bins=10):
    tot = 0.0
    for b in range(bins):
        sub = [r for r in rs if b / bins < max(r["probs"].values()) <= (b + 1) / bins]
        if sub:
            conf = sum(max(r["probs"].values()) for r in sub) / len(sub)
            tot += len(sub) / len(rs) * abs(conf - sum(map(correct, sub)) / len(sub))
    return tot


def gold_probs() -> dict:
    out = {}
    for line in open(JEVBENCH / "datasets/public/hard.jsonl"):
        r = json.loads(line)
        if r["family"] == "probability":
            out[r["id"]] = r["provenance"]["gold_probs"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--tier", default=None)
    args = ap.parse_args()
    gold = gold_probs()
    runs = [(p.name, load(p)) for p in args.runs]
    ctrl_name, ctrl = runs[0]
    print(f"control: {ctrl_name}\n")
    print(f"{'run':16s} {'n':>4s} {'acc':>6s} {'ctrl':>6s} {'diff':>6s} {'p':>7s} {'ECE':>6s} {'TVD':>6s} {'tok':>6s}  wins/losses")
    for name, run in runs:
        ids = sorted(set(run) & set(ctrl))
        if args.tier:
            ids = [i for i in ids if run[i]["tier"] == args.tier]
        a = [correct(run[i]) for i in ids]
        b = [correct(ctrl[i]) for i in ids]
        hard = [run[i] for i in ids if run[i]["tier"] == "hard"]
        tv = [0.5 * sum(abs(run[i]["probs"].get(l, 0) - gold[i][l]) for l in gold[i]) for i in ids if i in gold]
        test = paired_permutation_test(a, b) if name != ctrl_name else {"p_value": 1.0}
        wins = sum(x > y for x, y in zip(a, b)); losses = sum(x < y for x, y in zip(a, b))
        print(f"{name:16s} {len(ids):4d} {sum(a)/len(a):6.3f} {sum(b)/len(b):6.3f} {(sum(a)-sum(b))/len(a):+6.3f} "
              f"{test['p_value']:7.3f} {ece(hard) if hard else float('nan'):6.3f} "
              f"{sum(tv)/len(tv) if tv else float('nan'):6.3f} "
              f"{sum(run[i]['input_tokens'] for i in ids)/len(ids):6.0f}  {wins}/{losses}")


if __name__ == "__main__":
    main()
