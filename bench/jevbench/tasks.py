"""JevBench v1.2 public tasks, and the conversion into this engine's schema.

JevBench (github.com/fstandhartinger/jevbench, MIT) publishes 231 of its 534
decisions: 48 easy, 72 standard ("original") and 111 hard. The judge tier (146)
and 109 hard items are held out, so anything computed here is a public-subset
score and is labelled as such.

The conversion mirrors how the harness's own adapters send each type, so
results are comparable: `noul` becomes the two options true/false, `score` becomes the
ordered level indices, and every description is prefixed with its option id.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from quire.schema import Question

# The harness keys a noul's distribution "yes"/"no"; this engine returns "true"/"false".
HARNESS_KEYS = {"true": "yes", "false": "no"}


def harness_labels(probs: dict) -> dict:
    """A run record's distribution, keyed the way JevBench's gold distributions are."""
    return {HARNESS_KEYS.get(k, k): v for k, v in probs.items()}


TIERS = {"easy": "easy", "standard": "original", "hard": "hard"}
# jevbench/composite_v12.py: Intelligence is a weighted mean over tiers.
TIER_WEIGHTS = {"hard": 0.30, "easy": 0.14, "standard": 0.28, "judge": 0.28}


@dataclass(frozen=True)
class Task:
    id: str
    tier: str
    family: str
    state: str
    kind: str
    instructions: str
    criteria: dict[str, str]
    option_ids: tuple[str, ...]
    expected: str

    def to_question(self) -> Question:
        # score items are choices over declared levels. noul keeps its kind so
        # the engine can use the binary word readout when enabled.
        kind = "noul" if self.kind == "noul" else "choice"
        return Question(instructions=self.instructions, criteria=dict(self.criteria), kind=kind)


def _options(question) -> tuple[dict[str, str], tuple[str, ...]]:
    kind, criteria = question["type"], question.get("criteria")
    if kind == "noul":
        ids = ("true", "false")
        descriptions = [(criteria or {}).get(k) or f"The proposition is {k}." for k in ids]
    elif kind == "choice":
        ids = tuple(criteria)
        descriptions = [criteria[k] or k for k in ids]
    elif kind == "score":
        ids = tuple(str(i) for i in range(len(criteria)))
        descriptions = list(criteria)
    else:
        raise ValueError(f"unknown question type {kind!r}")
    return {i: f"{i}: {d}" for i, d in zip(ids, descriptions)}, ids


def load(root: pathlib.Path) -> list[Task]:
    tasks: list[Task] = []
    for tier, filename in TIERS.items():
        path = root / "datasets" / "public" / f"{filename}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            criteria, ids = _options(row["question"])
            expected = str(row["expected"])   # score items carry an int level
            if row["question"]["type"] == "noul":
                # The harness reports noul over yes/no; the gold is "yes"/"no".
                expected = {"yes": "true", "no": "false"}.get(str(expected), str(expected))
            if expected not in ids:
                raise ValueError(f"{row['id']}: expected {expected!r} not in {ids}")
            tasks.append(Task(
                id=row["id"], tier=tier, family=row.get("family") or "?",
                state=row["state"], kind=row["question"]["type"],
                instructions=row["question"]["instructions"],
                criteria=criteria, option_ids=ids, expected=expected,
            ))
    return tasks


def to_harness_probs(kind: str, probabilities: dict[str, float]) -> dict[str, float]:
    """The harness scores noul over yes/no; everything else over its own ids."""
    if kind == "noul":
        return {"yes": probabilities["true"], "no": probabilities["false"]}
    return dict(probabilities)
