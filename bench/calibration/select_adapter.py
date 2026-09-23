"""Pick an adapter scale on held-out items and gate it against the frozen model.

    python bench/calibration/select_adapter.py --frozen runs/calibration/frozen.jsonl \\
        --candidate 0.25=runs/calibration/a0.25.jsonl --candidate 0.5=... --out runs/calibration/select.json

Every file is predict.py output over the same held-out items. Procedure
(pre-registered, docs: retrain v2):

- split: the type-temperature study's 50/50 split (fit.py `split_half`) for the
  items it used, and the same procedure applied separately to any new source
  (HelpSteer2 dev), so the frozen model's released temperatures were never
  fitted on a check item;
- per scale: fit per-type temperatures on the fit half; the scale with the
  lowest pooled gold NLL on the check half is chosen;
- gate 1, at the chosen scale against the frozen model with its released
  temperatures, on the check half: accuracy not significantly lower (paired
  sign-flip, p < 0.05), and pooled ECE and NLL both lower.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fit import ece, fit_temperatures, scored, split_half  # noqa: E402

from quire.calibration import TypeTemperature  # noqa: E402
from quire.stats import paired_permutation_test  # noqa: E402

STUDY_SOURCES_EXCLUDE = {"helpsteer2"}   # sources added after the type-temperature study


def load(path):
    return {r["uid"]: r for r in map(json.loads, open(path)) if r}


def split(rows):
    old = [r for r in rows if r["source"] not in STUDY_SOURCES_EXCLUDE]
    new = [r for r in rows if r["source"] in STUDY_SOURCES_EXCLUDE]
    f0, c0 = split_half(old)
    f1, c1 = split_half(new) if new else ([], [])
    return {r["uid"] for r in f0 + f1}, {r["uid"] for r in c0 + c1}


def summary(rows, tt):
    s = scored(rows, tt)
    return {"accuracy": sum(x[1] for x in s) / len(s), "ece": ece([(c, ok) for c, ok, _ in s]),
            "nll": sum(x[2] for x in s) / len(s), "_ok": [x[1] for x in s]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=pathlib.Path, required=True)
    ap.add_argument("--candidate", action="append", required=True, help="scale=predictions.jsonl")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    frozen = load(args.frozen)
    cands = {float(c.split("=")[0]): load(c.split("=", 1)[1]) for c in args.candidate}
    uids = set(frozen)
    for scale, preds in cands.items():
        if set(preds) != uids:
            raise SystemExit(f"scale {scale}: {len(set(preds) ^ uids)} items differ from the frozen file")
    fit_ids, check_ids = split(list(frozen.values()))
    order = sorted(check_ids)
    print(f"{len(uids)} held-out items: fit {len(fit_ids)}, check {len(check_ids)}")

    base = summary([frozen[u] for u in order], TypeTemperature.default())
    print(f"\nfrozen, released temperatures: acc {base['accuracy']:.4f}  ECE {base['ece']:.4f}  NLL {base['nll']:.4f}")
    result = {"items": len(uids), "fit": len(fit_ids), "check": len(check_ids),
              "frozen": {k: v for k, v in base.items() if k != "_ok"}, "scales": {}}
    for scale in sorted(cands):
        preds = cands[scale]
        print(f"\nscale {scale}:")
        temps = fit_temperatures([preds[u] for u in sorted(fit_ids)], log=print)
        s = summary([preds[u] for u in order], TypeTemperature(temps))
        test = paired_permutation_test(s["_ok"], base["_ok"])
        print(f"  check: acc {s['accuracy']:.4f} ({s['accuracy'] - base['accuracy']:+.4f}, p = {test['p_value']:.3f})"
              f"  ECE {s['ece']:.4f}  NLL {s['nll']:.4f}")
        result["scales"][str(scale)] = {"temperatures": temps, "p_accuracy": test["p_value"],
                                        **{k: v for k, v in s.items() if k != "_ok"}}

    best = min(result["scales"], key=lambda k: result["scales"][k]["nll"])
    b = result["scales"][best]
    fell = b["accuracy"] < result["frozen"]["accuracy"] and b["p_accuracy"] < 0.05
    gate = (not fell) and b["ece"] < result["frozen"]["ece"] and b["nll"] < result["frozen"]["nll"]
    result.update({"chosen_scale": float(best), "gate1_passed": gate})
    print(f"\nchosen scale {best} (lowest check NLL).  gate 1: accuracy fell significantly: {fell}; "
          f"ECE {result['frozen']['ece']:.4f} -> {b['ece']:.4f}; NLL {result['frozen']['nll']:.4f} -> {b['nll']:.4f}"
          f"  => {'PASSED' if gate else 'FAILED'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
