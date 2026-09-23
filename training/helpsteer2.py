"""Convert HelpSteer2 into clear-cut answer-adequacy yes/no items.

HelpSteer2 (nvidia/HelpSteer2, CC BY 4.0) rates responses 0-4 on
helpfulness and correctness. Only the unambiguous ends become items:

    true   helpfulness 4 and correctness 4
    false  helpfulness <= 1 and correctness <= 1

Everything in between is dropped: a middling rating is not a fact one
forward pass can be trained to assert. Each prompt has two rated responses;
where one is a clear true and the other a clear false, both are kept as a
contrastive pair sharing a group id. The rest are filled from singles until
the classes are balanced.

Items whose prompt or response shares an 8-gram with any public JevBench item
are dropped (--jevbench).

    python training/helpsteer2.py --src path/to/HelpSteer2 --out data/synth/helpsteer2 \
        --jevbench path/to/jevbench --per-class 1500
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import pathlib
import random
import re

QUESTION = "The response answers the request helpfully and correctly."


def clear_label(row) -> bool | None:
    if row["helpfulness"] == 4 and row["correctness"] == 4:
        return True
    if row["helpfulness"] <= 1 and row["correctness"] <= 1:
        return False
    return None


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jevbench_ngrams(root: pathlib.Path) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for path in sorted((root / "datasets" / "public").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                grams |= ngrams(line)
    return grams


def convert(rows, per_class: int, seed: int, banned: set) -> tuple[list[dict], dict]:
    by_prompt = collections.defaultdict(list)
    for i, row in enumerate(rows):
        label = clear_label(row)
        if label is not None:
            by_prompt[row["prompt"]].append((i, row, label))
    stats = collections.Counter()
    pairs, singles = [], []
    for prompt, group in by_prompt.items():
        if banned and ngrams(prompt) & banned:
            stats["dropped_overlap"] += len(group)
            continue
        group = [g for g in group if not (banned and ngrams(g[1]["response"]) & banned)]
        labels = {g[2] for g in group}
        if len(group) == 2 and labels == {True, False}:
            pairs.append(group)
        else:
            singles.extend(group)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rng.shuffle(singles)
    chosen = [g for pair in pairs[:per_class] for g in pair]
    count = collections.Counter(g[2] for g in chosen)
    for g in singles:
        if count[g[2]] < per_class:
            chosen.append(g)
            count[g[2]] += 1
    stats.update(pairs=min(len(pairs), per_class), true=count[True], false=count[False])
    out = []
    for i, row, label in chosen:
        out.append({
            "state": {"request": row["prompt"], "response": row["response"]},
            "questions": {"q0": {"type": "noul", "instructions": QUESTION,
                                 "label": label, "src": "helpsteer2"}},
            "_meta": {"id": f"hs2-{i}", "source": "helpsteer2",
                      "group_id": "hs2-" + hashlib.sha1(row["prompt"].encode()).hexdigest()[:12]},
        })
    rng.shuffle(out)
    return out, dict(stats)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=pathlib.Path, required=True, help="directory holding train/validation.jsonl.gz")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--jevbench", type=pathlib.Path, help="JevBench checkout; drops 8-gram overlaps")
    p.add_argument("--per-class", type=int, default=1500)
    p.add_argument("--dev-per-class", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", type=pathlib.Path, action="append", default=[],
                   help="an earlier output's jsonl; its items are never drawn again (fresh evaluation sets)")
    p.add_argument("--only-split", choices=["train", "dev"], default=None,
                   help="write just this split (e.g. a fresh set drawn from HelpSteer2's train file)")
    args = p.parse_args()
    banned = jevbench_ngrams(args.jevbench) if args.jevbench else set()
    old = [json.loads(l) for path in args.exclude for l in open(path) if l.strip()]
    excluded = {r["_meta"]["id"] for r in old}
    excluded_prompts = {r["state"]["request"] for r in old}  # a prompt's other response is not fresh either
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "nvidia/HelpSteer2 (CC BY 4.0)", "question": QUESTION, "seed": args.seed,
                "excluded_from": [str(x) for x in args.exclude],
                "jevbench_filter": bool(banned), "splits": {}}
    for split, src, n in (("train", "train", args.per_class), ("dev", "validation", args.dev_per_class)):
        if args.only_split and split != args.only_split:
            continue
        rows = [json.loads(l) for l in gzip.open(args.src / f"{src}.jsonl.gz")]
        if split == "train" and excluded:
            # keep row indices stable (they are the item ids); blank excluded rows out
            rows = [dict(r, helpfulness=2, correctness=2) if f"hs2-{i}" in excluded or r["prompt"] in excluded_prompts else r for i, r in enumerate(rows)]
        out, stats = convert(rows, n, args.seed, banned)
        (args.out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
        manifest["splits"][split] = {"rows": len(out), **stats}
        print(split, len(out), stats)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
