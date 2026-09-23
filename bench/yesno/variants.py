"""Score the pre-registered yes/no variants from raw readouts and apply the decision rule.

    python bench/yesno/variants.py runs/yesno/readouts-mlx.jsonl --out runs/yesno/variants-mlx.json

Pre-registration: jev-research docs/superpowers/specs/2026-09-23-yesno-lean.md.
Every variant is computed from the same readouts (bench/yesno/readouts.py);
only Vc / Vc1 use the fit half. All comparisons are on the check half.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "calibration"))

from fit import ece, fit_temperatures, scored, split_half  # noqa: E402

from quire.calibration import TypeTemperature  # noqa: E402
from quire.stats import paired_permutation_test  # noqa: E402

ELIGIBLE = ["Va", "Vb", "Vc1", "Vd", "Ve"]
REPORTED = ["V0", "Va", "Va4", "Vb", "Vc", "Vc1", "Vd", "Ve"]
MIN_N = 20


def sig(x):
    return 1 / (1 + math.exp(-max(min(x, 50), -50)))


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def p_of(r, k):
    a, b = r[k]
    return sig(a - b)


def base_variants(r):
    p = {k: p_of(r, k) for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7")}
    cf = [sig((r["R1"][0] - r["R1"][1]) - (r["C1"][0] - r["C1"][1])),
          sig((r["R2"][0] - r["R2"][1]) - (r["C2"][0] - r["C2"][1]))]
    return {
        "V0": (p["R1"] + p["R2"]) / 2,
        "Va": (p["R1"] + 1 - p["R4"]) / 2,
        "Va4": (p["R1"] + p["R2"] + 2 - p["R3"] - p["R4"]) / 4,
        "Vb": p["R5"],
        "Vd": sum(cf) / 2,
        "Ve": (p["R6"] + p["R7"]) / 2,
    }


def fit_intercept(rows):
    """The one intercept b maximising the likelihood of gold under sigmoid(logit(V0) + b)."""
    grid = [i / 100 for i in range(-300, 301)]
    def nll(b):
        return -sum(math.log(max(sig(logit(r["V"]["V0"]) + b) if r["gold"] == "true"
                                 else 1 - sig(logit(r["V"]["V0"]) + b), 1e-12)) for r in rows)
    return min(grid, key=nll)


def as_pred(r, v):
    p = r["V"][v]
    return {"uid": r["uid"], "source": r["source"], "kind": "noul", "gold": r["gold"],
            "probs": {"true": p, "false": 1 - p}}


def correct(r, v):
    return float((r["V"][v] >= 0.5) == (r["gold"] == "true"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readouts", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.readouts) if l.strip()]
    for r in rows:
        r["V"] = base_variants(r)
    fit, check = split_half(rows)
    b_all = fit_intercept(fit)
    by_src = collections.defaultdict(list)
    for r in fit:
        by_src[r["source"]].append(r)
    b_src = {s: fit_intercept(v) for s, v in by_src.items()}
    for r in rows:
        r["V"]["Vc1"] = sig(logit(r["V"]["V0"]) + b_all)
        r["V"]["Vc"] = sig(logit(r["V"]["V0"]) + b_src.get(r["source"], 0.0))
    print(f"{len(rows)} yes/no items: fit {len(fit)}, check {len(check)};  global intercept {b_all:+.2f}")

    sources = sorted({r["source"] for r in check})
    big = [s for s in sources if sum(r["source"] == s for r in check) >= MIN_N]

    def bal(sub, v):
        acc = []
        for g in ("true", "false"):
            s = [r for r in sub if r["gold"] == g]
            if s:
                acc.append(sum(correct(r, v) for r in s) / len(s))
        return sum(acc) / len(acc)

    res = {"items": len(rows), "fit": len(fit), "check": len(check), "intercept_global": b_all,
           "intercept_by_source": b_src, "variants": {}}
    base_ok = [correct(r, "V0") for r in check]
    # Each check item's weight in macro balanced accuracy: the weighted sum of
    # correctness equals the metric, so a paired sign-flip test on weighted
    # correctness tests the metric itself (amendment 1).
    cls = collections.Counter((r["source"], r["gold"]) for r in check)
    ncls = collections.Counter(s for s, _ in cls)
    w = {r["uid"]: (1 / (ncls[r["source"]] * cls[(r["source"], r["gold"])] * len(big)) if r["source"] in big else 0.0)
         for r in check}
    base_w = [w[r["uid"]] * correct(r, "V0") for r in check]
    for v in REPORTED:
        macro = sum(bal([r for r in check if r["source"] == s], v) for s in big) / len(big)
        ok = [correct(r, v) for r in check]
        t = paired_permutation_test(ok, base_ok)["p_value"] if v != "V0" else 1.0
        tb = paired_permutation_test([w[r["uid"]] * correct(r, v) for r in check], base_w)["p_value"] if v != "V0" else 1.0
        per = {}
        worse = []
        for s in sources:
            sub = [r for r in check if r["source"] == s]
            a = [correct(r, v) for r in sub]
            b = [correct(r, "V0") for r in sub]
            ps = paired_permutation_test(a, b)["p_value"] if v != "V0" and a != b else 1.0
            per[s] = {"n": len(sub), "acc": sum(a) / len(a), "acc_true": None, "bal": bal(sub, v), "p_vs_V0": ps}
            tr = [correct(r, v) for r in sub if r["gold"] == "true"]
            per[s]["acc_true"] = sum(tr) / len(tr) if tr else None
            if len(sub) >= MIN_N and sum(a) < sum(b) and ps < 0.05:
                worse.append(s)
        temps = fit_temperatures([as_pred(r, v) for r in fit], log=lambda *_: None)
        e = ece([(c, ok_) for c, ok_, _ in scored([as_pred(r, v) for r in check], TypeTemperature(temps))])
        mean_pt = sum(r["V"][v] for r in check) / len(check)
        res["variants"][v] = {"macro_balanced": macro, "p_macro_vs_V0": tb, "accuracy": sum(ok) / len(ok), "p_accuracy_vs_V0": t,
                              "sources_significantly_worse": worse, "noul_T": temps["noul"], "ece_after_refit": e,
                              "mean_p_true": mean_pt, "per_source": per}
        print(f"{v:4s} macro-bal {macro:.4f} (p {tb:.4f})  acc {sum(ok)/len(ok):.4f} (p {t:.3f})  ECE {e:.4f} (T {temps['noul']:.2f})"
              f"  mean p(true) {mean_pt:.3f}  worse: {worse or '-'}")

    print("\nper source (check half): accuracy on true items / balanced accuracy")
    print(f"{'source':24s} {'n':>4s} " + " ".join(f"{v:>11s}" for v in REPORTED))
    for s in sources:
        n = res["variants"]["V0"]["per_source"][s]["n"]
        cells = []
        for v in REPORTED:
            d = res["variants"][v]["per_source"][s]
            cells.append(f"{(d['acc_true'] or 0):.2f}/{d['bal']:.2f}")
        print(f"{s[:24]:24s} {n:4d} " + " ".join(f"{c:>11s}" for c in cells))

    v0 = res["variants"]["V0"]
    best = max(ELIGIBLE, key=lambda v: res["variants"][v]["macro_balanced"])
    b = res["variants"][best]
    conds = {
        "macro_balanced_higher": b["macro_balanced"] > v0["macro_balanced"],
        "macro_improvement_significant": b["p_macro_vs_V0"] < 0.05 / len(ELIGIBLE),
        "accuracy_not_significantly_lower": not (b["accuracy"] < v0["accuracy"] and b["p_accuracy_vs_V0"] < 0.05),
        "no_source_significantly_worse": not b["sources_significantly_worse"],
        "ece_not_worse": b["ece_after_refit"] <= v0["ece_after_refit"] + 0.005,
    }
    res.update({"chosen": best, "conditions": conds, "adopt": all(conds.values())})
    print(f"\nchosen {best}: " + ", ".join(f"{k}={v}" for k, v in conds.items()) + f"  => adopt {res['adopt']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
