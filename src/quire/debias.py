"""Contextual calibration (Zhao et al. 2021).

Score the schema against a content-free state, then subtract those log-probs
from the real ones. What remains is the part of the signal that came from the
state rather than from the model's positional habits.

The prior depends on the SHAPE of the schema -- instructions, option count and
option descriptions -- but not on the state, so it caches across every request
using that shape. Typically several points of accuracy for near-zero cost.

The descriptions belong in the key because the content-free probe renders them:
see `Engine._score_content_free`, which blanks the STATE and nothing else. An
earlier `shape_key` hashed the instructions alone, so two schemas that differed
only in their option wording collided and the second silently reused the
first's prior. The perturbed split in `bench/perturb.py` is precisely that
case, which put the collision underneath the format-overfitting claim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

CONTENT_FREE_STATE = "N/A"


def debias(logits: np.ndarray, prior_logits: np.ndarray) -> np.ndarray:
    """Subtract the content-free prior from the observed logits."""
    return np.asarray(logits, dtype=float) - np.asarray(prior_logits, dtype=float)


class PriorCache:
    """Memoises the content-free prior per schema shape."""

    def __init__(self, scorer: Callable[[str], np.ndarray]) -> None:
        self._scorer = scorer
        self._cache: dict[str, np.ndarray] = {}

    def get(self, shape_key: str) -> np.ndarray:
        if shape_key not in self._cache:
            self._cache[shape_key] = self._scorer(shape_key)
        return self._cache[shape_key]

    def clear(self) -> None:
        self._cache.clear()


def shape_key(question, order: list[str] | None = None) -> str:
    """Identify a schema shape. Two questions with this key share a prior.

    Everything the content-free probe renders must be in the digest, or the
    cache returns a prior measured on a different prompt. That means the option
    descriptions as well as the instructions, in the order they are rendered.

    `order` is that rendering order. Omitted, it is the question's canonical
    one, which is what a single shared prior per schema assumes. Passing a
    rotation gives that rotation its own prior -- see `Engine.per_rotation_prior`
    for why the two differ and what it costs.

    Uses a stable digest rather than hash(): Python randomises string hashing
    per process, so hash() would yield a different key every run. The prior is
    only cached in-process today, but the key also identifies which prior a
    fitted calibration belongs to, and calibration, prompt template and model
    revision are versioned together as one artifact.
    """
    parts = [question.instructions]
    parts += [question.criteria[option_id] for option_id in (order or question.option_ids)]
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{question.kind}:{question.n_options}:{digest}"
