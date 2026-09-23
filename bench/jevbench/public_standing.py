"""Where we stand against the leaderboard on the 231 PUBLIC items only.

The site publishes every system's per-task outcome for the public items
(results/v1.2/jevbench-v1.2-per-task.json). Scoring each system on exactly
the items we can run locally is the only like-for-like comparison there is;
the judge tier (146 items, 28% of Intelligence) has no public items at all.

    .venv/bin/python scripts/public_standing.py results/jevbench/p2-r2/run.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

PER_TASK_URL = ("https://raw.githubusercontent.com/fstandhartinger/jevbench/main/"
                "results/v1.2/jevbench-v1.2-per-task.json")


def our_outcomes(run: dict) -> dict[str, bool]:
    # argmax == expected, exactly as jevbench.scoring.score_task and
    # scripts/score_jevbench.py do it. `probs` carries the task's own labels;
    # `harness_probs` is the wire format (noul relabelled yes/no) and must
    # not be scored against `expected`.
    return {r["id"]: max(r["probs"], key=r["probs"].get) == r["expected"] for r in run["records"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--versus", action="append", default=None,
                    help="paired comparison against this leaderboard system (substring of its display name), per tier")
    ap.add_argument("--per-task", default=None, help="URL or file:// of the per-task outcomes JSON; default: the published v1.3 file")
    args = ap.parse_args()
    import urllib.request
    with urllib.request.urlopen(args.per_task or PER_TASK_URL) as f:
        pt = json.loads(f.read())
    tiers = {t["id"]: t["tier"] for t in pt["tasks"]}
    ids_by_tier = defaultdict(list)
    for tid, tier in tiers.items():
        ids_by_tier[tier].append(tid)

    rows = []
    for key, sysd in pt["systems"].items():
        if sysd.get("partial"):
            continue
        acc = {}
        for tier, ids in ids_by_tier.items():
            oc = [sysd["public_tasks"].get(i, ["n"])[0] for i in ids]
            acc[tier] = sum(o == "c" for o in oc) / len(ids)
        rows.append((sysd["display"], acc, sysd))

    run = json.loads(pathlib.Path(args.run).read_text())
    ours = our_outcomes(run)
    our_acc = {tier: sum(ours.get(i, False) for i in ids) / len(ids) for tier, ids in ids_by_tier.items()}

    def pub_intel(acc):  # public-only proxy: same tier weights, judge dropped and renormalised
        w = {"easy": 0.14, "standard": 0.28, "hard": 0.30}
        return sum(w[t] * acc[t] for t in w) / sum(w.values())

    rows.sort(key=lambda r: -pub_intel(r[1]))
    print(f"{'system':46s} {'easy48':>7s} {'std72':>7s} {'hard111':>8s} {'pubIntel':>9s}")
    for name, acc, _ in rows[: args.top]:
        print(f"{name[:46]:46s} {acc['easy']:7.3f} {acc['standard']:7.3f} {acc['hard']:8.3f} {pub_intel(acc):9.3f}")
    label = run["_provenance"].get("label") or run["_provenance"].get("system", "this run")
    print(f"{'>>> ' + label:46s} {our_acc['easy']:7.3f} {our_acc['standard']:7.3f} "
          f"{our_acc['hard']:8.3f} {pub_intel(our_acc):9.3f}")
    better = sum(pub_intel(a) > pub_intel(our_acc) for _, a, _ in rows)
    print(f"\nrank on public items by accuracy alone: {better + 1} of {len(rows) + 1}")

    # item-level vs each of the top 3 on the public subset
    print("\nitems vs top systems (public only):  we-win / they-win / both-wrong")
    for name, acc, sysd in rows[:5]:
        theirs = {i: sysd["public_tasks"].get(i, ["n"])[0] == "c" for i in tiers}
        ww = sum(ours.get(i, False) and not theirs[i] for i in tiers)
        tw = sum(theirs[i] and not ours.get(i, False) for i in tiers)
        bw = sum((not theirs[i]) and not ours.get(i, False) for i in tiers)
        by = defaultdict(lambda: [0, 0])
        for i in tiers:
            if ours.get(i, False) != theirs[i]:
                by[tiers[i]][0 if ours.get(i, False) else 1] += 1
        print(f"  {name[:40]:40s} {ww:3d} / {tw:3d} / {bw:3d}   " + "  ".join(f"{t}: +{v[0]}/-{v[1]}" for t, v in by.items()))

    for name in args.versus or []:
        match = [(n, a, d) for n, a, d in rows if name.lower() in n.lower()]
        if not match:
            print(f"\nno leaderboard system matches {name!r}")
            continue
        n, _, sysd = match[0]
        from quire.stats import paired_permutation_test
        print(f"\n{label} vs {n} (the maintainers' own run of it), paired on the public items:")
        for tier in ("easy", "standard", "hard", None):
            ids = [i for i in tiers if tier is None or tiers[i] == tier]
            a = [float(ours.get(i, False)) for i in ids]
            b = [float(sysd["public_tasks"].get(i, ["n"])[0] == "c") for i in ids]
            t = paired_permutation_test(a, b)
            print(f"  {tier or 'total':9s} {int(sum(a)):4d} vs {int(sum(b)):4d} of {len(ids):3d}   "
                  f"{sum(x > y for x, y in zip(a, b))} better / {sum(x < y for x, y in zip(a, b))} worse   p = {t['p_value']:.3f}")


if __name__ == "__main__":
    main()
