"""Load kev's frozen decision suites into one normalised shape.

kev ships three question types. They are all "score each of N declared options"
underneath, so they are normalised to that here:

  choice  criteria is {option_id: description or null}     -> N options
  noul    criteria is {"true": ..., "false": ...}          -> 2 options
  score   criteria is an ordered list of level descriptions -> N options

`score` is ordinal and its natural metric is not accuracy, but the readout
question -- how do you turn hidden states into one score per option -- is the
same, so it is carried through and reported separately.

Data: github.com/jaredpalmer/kev under Apache-2.0, checksums in each manifest.
Not vendored; `kev-data/` is git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

DATA = pathlib.Path(__import__("os").environ.get("KEV_DATA", "kev-data"))  # a checkout of jaredpalmer/kev's data
SUITES = {
    "decision-v7": "v7/decision-v7",
    "transfer-v4": "v4/transfer-v4",
    "transfer-v9": "v9/transfer-v9",
}


@dataclass(frozen=True)
class Item:
    """One question against one state."""

    uid: str
    suite: str
    split: str
    source: str
    kind: str                     # choice | noul | score
    state: str
    instructions: str
    option_ids: tuple[str, ...]
    descriptions: tuple[str, ...] # aligned with option_ids; "" where null
    label: int                    # index into option_ids
    group_id: str
    soft: tuple | None = None     # exact gold distribution over option_ids, when the item has one


def flatten_state(state) -> str:
    """States arrive as a string, a dict of labelled fields, or a transcript.

    All three shapes appear in kev's suites (3201 / 4242 / 213 rows). The
    rendering is fixed here and shared by every arm, so it cannot favour one.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(f"{k}: {v}" for k, v in state.items())
    if isinstance(state, list):
        parts = []
        for turn in state:
            if isinstance(turn, dict) and "content" in turn:
                parts.append(f"{turn.get('role', 'turn')}: {turn['content']}")
            else:
                parts.append(str(turn))
        return "\n".join(parts)
    raise ValueError(f"unsupported state type {type(state).__name__}")


def _options(question) -> tuple[tuple[str, ...], tuple[str, ...]]:
    criteria = question.get("criteria")
    if criteria is None:
        # `noul` may omit descriptions entirely; the options are still true/false.
        if question["type"] != "noul":
            raise ValueError(f"{question['type']} question has no criteria")
        return ("true", "false"), ("", "")
    if isinstance(criteria, list):                      # score
        ids = tuple(str(i) for i in range(len(criteria)))
        return ids, tuple(str(c) for c in criteria)
    ids = tuple(criteria)
    return ids, tuple("" if criteria[k] is None else str(criteria[k]) for k in ids)


def _label_index(question, option_ids: tuple[str, ...]) -> int | None:
    label = question.get("label")
    if label is None:
        return None
    if question["type"] == "noul":
        return option_ids.index("true" if label else "false")
    if question["type"] == "score":
        return int(label)
    key = str(label)
    return option_ids.index(key) if key in option_ids else None


def load(suite: str, split: str, data: pathlib.Path = DATA) -> list[Item]:
    path = data / SUITES[suite] / f"{split}.jsonl"
    items: list[Item] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("_meta") or {}
        state = flatten_state(row["state"])
        for qid, question in row["questions"].items():
            ids, descriptions = _options(question)
            label = _label_index(question, ids)
            if label is None or len(ids) < 2:
                continue            # unlabelled or degenerate; never silently scored
            uid = f"{suite}/{split}/{meta.get('id', len(items))}/{qid}"
            items.append(Item(
                uid=uid, suite=suite, split=split,
                source=question.get("src") or meta.get("source") or "unknown",
                kind=question["type"], state=state,
                instructions=question["instructions"],
                option_ids=ids, descriptions=descriptions, label=label,
                group_id=str(meta.get("group_id", uid)),
                soft=(tuple(float(question["gold_probs"].get(o, 0.0)) for o in ids)
                      if isinstance(question.get("gold_probs"), dict) else None),
            ))
    return items


def digest(items: list[Item]) -> str:
    joined = "\x00".join(f"{i.uid}:{i.label}:{len(i.option_ids)}" for i in items)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]
