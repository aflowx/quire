"""Project a local run onto the full JevBench v1.3 score, honestly.

Locally we have 231 public items and no judge tier. This script scores what
we have with the harness's own formulas (composite_v13), computes the
Calibration axis the way the harness does (ECE on hard + fidelity on the
probability items' gold_probs), and prints the composite for a range of
assumed judge-tier accuracies, since that tier is 28% of Intelligence and
we cannot measure it.

    .venv/bin/python scripts/project_jevbench.py results/jevbench/p2-r2/run.json \
        --jevbench /path/to/jevbench --speed 88.1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def ece(records, bins=10):
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [r for r in records if lo < max(r["probs"].values()) <= hi]
        if sub:
            conf = sum(max(r["probs"].values()) for r in sub) / len(sub)
            acc = sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in sub) / len(sub)
            tot += len(sub) / len(records) * abs(conf - acc)
    return tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--jevbench", required=True, help="a checkout of fstandhartinger/jevbench")
    ap.add_argument("--speed", type=float, required=True, help="Speed axis from bench/jevbench/speed_probe.py")
    ap.add_argument("--cost", type=float, default=None,
                    help="Cost axis; default prices the run's own mean input tokens at --input-per-million")
    ap.add_argument("--input-per-million", type=float, default=0.03,
                    help="hosted list price for the weights; 0.03 = deepinfra Qwen3.5-4B, the basis JevBench uses for Qwen3.5-4B entrants")
    args = ap.parse_args()
    sys.path.insert(0, args.jevbench)
    from jevbench import composite_v13 as cv

    run = json.loads(pathlib.Path(args.run).read_text())
    recs = run["records"]
    if args.cost is None:
        mean_tok = sum(r["input_tokens"] for r in recs) / len(recs)
        usd_per_1000 = mean_tok * args.input_per_million / 1e6 * 1000
        args.cost = cv.cost(usd_per_1000)
        cost_note = f"{mean_tok:.0f} tok x ${args.input_per_million}/M = ${usd_per_1000:.4f}/1k -> {args.cost:.1f}"
    else:
        cost_note = "given"
    gold = {}
    for line in open(pathlib.Path(args.jevbench) / "datasets/public/hard.jsonl"):
        r = json.loads(line)
        if r["family"] == "probability":
            gold[r["id"]] = r["provenance"]["gold_probs"]

    tiers = {}
    for tier in ("easy", "standard", "hard"):
        sub = [r for r in recs if r["tier"] == tier]
        tiers[tier] = sum(max(r["probs"], key=r["probs"].get) == r["expected"] for r in sub) / len(sub)
    hard = [r for r in recs if r["tier"] == "hard"]
    e = ece(hard)
    tvds = [cv.tvd(r["probs"], gold[r["id"]], list(gold[r["id"]])) for r in hard if r["id"] in gold]
    mean_tvd = sum(tvds) / len(tvds)
    calib = cv.calibration(e, mean_tvd)
    print(f"run {run['_provenance']['label']}: easy {tiers['easy']:.3f}  standard {tiers['standard']:.3f}  "
          f"hard {tiers['hard']:.3f}  (public items only)")
    print(f"calibration: ECE {e:.4f} -> {100 * (1 - e / 0.5):.1f};  fidelity on {len(tvds)} probability items: "
          f"mean TVD {mean_tvd:.3f} -> {100 * (1 - mean_tvd):.1f};  axis = {calib:.1f}")
    print(f"assumed speed {args.speed}; cost {cost_note}\n")
    print(f"{'judge acc':>10s} {'Intel':>7s} {'Score':>7s}")
    for j in (0.75, 0.80, 0.85, 0.90, 0.95):
        intel = cv.intelligence({**tiers, "judge": j})
        score = cv.jevbench_score({"intelligence": intel, "calibration": calib, "speed": args.speed, "cost": args.cost})
        print(f"{j:10.2f} {intel:7.1f} {score:7.1f}")


if __name__ == "__main__":
    main()
