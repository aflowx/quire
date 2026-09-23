"""Temperature per (answer type, option-count bucket), fitted on held-out predictions.

    python bench/calibration/fit_buckets.py runs/select-v2/frozen.jsonl \\
        --confirm results/jevbench/quire-p2/run.json --jevbench $JEVBENCH --out runs/calibration/buckets.json

Pre-registered (jev-research docs/superpowers/specs/2026-09-23-calibration-map-v2.md):
M0 = released per-type temperatures; M1 = one temperature per bucket
(choice 2-3 / 4-5 / 6+, noul, score), fitted by gold NLL on the fit half,
falling back to the type's released temperature under 100 fit items.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "jevbench"))

from fit import GRID, ece, split_half  # noqa: E402
from tasks import harness_labels  # noqa: E402

from quire.calibration import TypeTemperature  # noqa: E402
from quire.stats import paired_permutation_test  # noqa: E402


def bucket(kind: str, n: int) -> str:
    if kind != "choice":
        return kind
    return "choice:2-3" if n <= 3 else "choice:4-5" if n <= 5 else "choice:6+"


def temper(p: dict, t: float) -> dict:
    w = {k: max(v, 1e-12) ** (1 / t) for k, v in p.items()}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


class BucketMap:
    def __init__(self, temps: dict, fallback: TypeTemperature):
        self.temps, self.fallback = temps, fallback

    def apply(self, probs: dict, kind: str) -> dict:
        b = bucket(kind, len(probs))
        t = self.temps.get(b, self.fallback.temperatures.get(kind, 1.0))
        return temper(probs, t)


def nll(m, r):
    return -math.log(max(m.apply(r["probs"], r["kind"])[r["gold"]], 1e-12))


def tvd_soft(m, r):
    p = m.apply(r["probs"], r["kind"])
    ids = list(r["probs"])  # predict.py keeps option order; soft is aligned to it
    return 0.5 * sum(abs(p[o] - s) for o, s in zip(ids, r["soft"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=pathlib.Path)
    ap.add_argument("--confirm", type=pathlib.Path)
    ap.add_argument("--jevbench", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.predictions) if l.strip()]
    rows = [r for r in rows if len(r["probs"]) <= 26]
    fit, check = split_half(rows)
    m0 = TypeTemperature.default()
    temps = {}
    for b in sorted({bucket(r["kind"], len(r["probs"])) for r in fit}):
        sub = [r for r in fit if bucket(r["kind"], len(r["probs"])) == b]
        if len(sub) < 100:
            print(f"  {b:12s} {len(sub):5d} fit items -> released type temperature")
            continue
        score = {t: sum(nll(BucketMap({b: t}, m0), r) for r in sub) / len(sub) for t in GRID}
        temps[b] = min(score, key=score.get)
        print(f"  {b:12s} {len(sub):5d} fit items -> T = {temps[b]:.2f}")
    m1 = BucketMap(temps, m0)

    def summary(m):
        s = [(max(p.values()), float(max(p, key=p.get) == r["gold"])) for r in check for p in [m.apply(r["probs"], r["kind"])]]
        soft = [r for r in check if r.get("soft")]
        return {"ece": ece(s), "nll": sum(nll(m, r) for r in check) / len(check),
                "tvd_soft": sum(tvd_soft(m, r) for r in soft) / len(soft), "n_soft": len(soft)}
    a, b = summary(m0), summary(m1)
    t = paired_permutation_test([nll(m1, r) for r in check], [nll(m0, r) for r in check])["p_value"]
    conds = {"nll_lower_significant": b["nll"] < a["nll"] and t < 0.05,
             "ece_not_worse": b["ece"] <= a["ece"] + 0.002,
             "tvd_soft_not_worse": b["tvd_soft"] <= a["tvd_soft"] + 0.002}
    print(f"\ncheck half ({len(check)}): M0 ECE {a['ece']:.4f} NLL {a['nll']:.4f} TVDsoft {a['tvd_soft']:.4f} | "
          f"M1 ECE {b['ece']:.4f} NLL {b['nll']:.4f} (p {t:.4f}) TVDsoft {b['tvd_soft']:.4f}   {conds}")
    res = {"temperatures": temps, "M0": a, "M1": b, "p_nll": t, "conditions": conds, "passed": all(conds.values())}
    if res["passed"] and args.confirm:
        sys.path.insert(0, str(args.jevbench))
        from jevbench import composite_v13 as cv
        run = json.loads(args.confirm.read_text())["records"]
        gold = {}
        for line in open(args.jevbench / "datasets/public/hard.jsonl"):
            g = json.loads(line)
            if g["family"] == "probability":
                gold[g["id"]] = g["provenance"]["gold_probs"]

        def axis(m):
            hard = [r for r in run if r["tier"] == "hard"]
            probs = [m.apply(r["probs"], r["kind"]) for r in hard]
            e = ece([(max(p.values()), float(max(p, key=p.get) == r["expected"])) for p, r in zip(probs, hard)])
            tv = [0.5 * sum(abs(harness_labels(p).get(l, 0) - gold[r["id"]][l]) for l in gold[r["id"]])
                  for p, r in zip(probs, hard) if r["id"] in gold]
            return e, sum(tv) / len(tv), cv.calibration(e, sum(tv) / len(tv))
        (e0, t0, x0), (e1, t1, x1) = axis(m0), axis(m1)
        print(f"confirmation (public hard): ECE {e0:.4f} -> {e1:.4f}  TVD {t0:.3f} -> {t1:.3f}  axis {x0:.1f} -> {x1:.1f}")
        res.update({"confirm": {"M0": [e0, t0, x0], "M1": [e1, t1, x1]}, "ship": x1 > x0})
    elif not res["passed"]:
        print("held-out conditions failed: no confirmation run (pre-registered stop)")
        res["ship"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"ship: {res.get('ship')}")


if __name__ == "__main__":
    main()
