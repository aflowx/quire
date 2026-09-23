"""Fit one temperature per answer type on held-out predictions, check it, confirm it once.

    python bench/calibration/fit.py runs/calibration/predictions.jsonl \\
        --confirm runs/quire-p2/run.json --jevbench $JEVBENCH --out calibration/type-temperature.json

Procedure (pre-registered): split held-out items 50/50 by item, stratified by
source (seed 0); per type, pick T on a 0.5..3.0 grid minimising gold NLL on the
fit half (types with < 100 fit items keep T = 1); report pooled top-label ECE
and NLL on the check half; then, once, the JevBench harness's Calibration axis
on the public items with and without the map.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "jevbench"))
from tasks import harness_labels  # noqa: E402

from quire.calibration import TYPES, TypeTemperature  # noqa: E402

GRID = [round(0.5 + 0.05 * i, 2) for i in range(51)]


def ece(rows, bins=10):
    tot = 0.0
    for b in range(bins):
        sub = [(c, ok) for c, ok in rows if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if sub:
            tot += len(sub) / len(rows) * abs(sum(c for c, _ in sub) / len(sub) - sum(ok for _, ok in sub) / len(sub))
    return tot


def scored(items, tt):
    out = []
    for r in items:
        p = tt.apply(r["probs"], r["kind"])
        top = max(p, key=p.get)
        out.append((p[top], float(top == r["gold"]), -math.log(max(p[r["gold"]], 1e-12))))
    return out


def split_half(rows):
    """50/50 by item, stratified by source, seed 0 (the pre-registered split)."""
    by_source = collections.defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    rng = random.Random(0)
    fit, check = [], []
    for src in sorted(by_source):
        items = sorted(by_source[src], key=lambda r: r["uid"])
        rng.shuffle(items)
        fit += items[: len(items) // 2]
        check += items[len(items) // 2:]
    return fit, check


def fit_temperatures(fit, log=print):
    """Per type, the grid T minimising gold NLL; fewer than 100 fit items keeps T = 1."""
    temps = {}
    for t in TYPES:
        sub = [r for r in fit if r["kind"] == t]
        if len(sub) < 100:
            temps[t] = 1.0
            log(f"  {t:6s} {len(sub):5d} fit items -> T = 1 (fewer than 100)")
            continue
        nll = {T: sum(x[2] for x in scored(sub, TypeTemperature({t: T}))) / len(sub) for T in GRID}
        temps[t] = min(nll, key=nll.get)
        log(f"  {t:6s} {len(sub):5d} fit items -> T = {temps[t]:.2f}   NLL {nll[1.0]:.4f} -> {nll[temps[t]]:.4f}")
    return temps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=pathlib.Path)
    ap.add_argument("--confirm", type=pathlib.Path, help="run.json on JevBench public items (run_public.py)")
    ap.add_argument("--jevbench", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.predictions) if l.strip()]
    fit, check = split_half(rows)
    print(f"{len(rows)} held-out items: fit {len(fit)}, check {len(check)}")
    temps = fit_temperatures(fit)
    tt = TypeTemperature(temps)

    base, cal = scored(check, TypeTemperature()), scored(check, tt)
    e0, e1 = ece([(c, ok) for c, ok, _ in base]), ece([(c, ok) for c, ok, _ in cal])
    n0, n1 = sum(x[2] for x in base) / len(base), sum(x[2] for x in cal) / len(cal)
    print(f"\ncheck half ({len(check)}): pooled ECE {e0:.4f} -> {e1:.4f}   NLL {n0:.4f} -> {n1:.4f}")
    for t in TYPES:
        sub = [r for r in check if r["kind"] == t]
        if sub:
            a = ece([(c, ok) for c, ok, _ in scored(sub, TypeTemperature())])
            b = ece([(c, ok) for c, ok, _ in scored(sub, tt)])
            print(f"  {t:6s} {len(sub):5d} items   ECE {a:.4f} -> {b:.4f}")
    proceed = e1 < e0
    result = {"temperatures": temps, "fit_items": len(fit), "check_items": len(check),
              "check_ece": [e0, e1], "check_nll": [n0, n1], "check_passed": proceed,
              "predictions": str(args.predictions)}

    if proceed and args.confirm:
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

        (a_e, a_t, a), (b_e, b_t, b) = axis(TypeTemperature()), axis(tt)
        print(f"\nconfirmation, JevBench public hard tier: ECE {a_e:.4f} -> {b_e:.4f}   fidelity TVD {a_t:.3f} -> {b_t:.3f}"
              f"   Calibration axis {a:.1f} -> {b:.1f}")
        result.update({"confirm_run": str(args.confirm), "confirm_axis": [a, b], "confirm_ece": [a_e, b_e],
                       "confirm_tvd": [a_t, b_t], "adopt": b >= a})
    elif not proceed:
        print("\ncheck half did not improve: the map is not confirmed on public items (pre-registered stop).")
        result["adopt"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"\nadopt: {result.get('adopt')}  -> {args.out}")


if __name__ == "__main__":
    main()
