"""Typed question and answer records.

Three kinds, mirroring Jev's primitives:

  choice -- exactly one of N; probabilities are a softmax over option logits.
  noul   -- independent binary judgements; scores are NOT normalised across
            options, because multi-label selection is not a competition
            between candidates.
  score  -- an ordinal rating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("choice", "noul", "score")


@dataclass
class Question:
    """One typed question over a shared state."""

    instructions: str
    # option id -> description. Jev calls this `criteria`. Insertion order is
    # the canonical order; permutation ensembling rotates copies of it.
    criteria: dict[str, str]
    kind: str = "choice"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind != "noul" and len(self.criteria) < 2:
            raise ValueError(
                f"a {self.kind} question needs at least 2 options, got {len(self.criteria)}"
            )
        if not self.criteria:
            raise ValueError("a question needs at least 1 option")

    @property
    def option_ids(self) -> list[str]:
        return list(self.criteria)

    @property
    def n_options(self) -> int:
        return len(self.criteria)


@dataclass
class Answer:
    """The result of one question.

    `aleatoric` and `epistemic` are deliberately separate. A flat distribution
    can mean "the world is genuinely 50/50" (accept it) or "the model has no
    idea" (escalate). Those demand opposite responses and one scalar cannot
    carry both -- which is the documented limitation of Jev's `confidence`.
    """

    choice: str
    probabilities: dict[str, float]
    aleatoric: float  # normalised entropy of the averaged distribution
    epistemic: float  # disagreement across option orderings
    margin: float  # top1 - top2; the abstention signal
    latency_ms: float = 0.0
    # Tokens the engine actually read: the shared state once per decide()
    # call, plus this question's suffixes (one per ordering). A request's
    # honest input count is state_tokens + the sum of suffix_tokens.
    state_tokens: int = 0
    suffix_tokens: int = 0
    raw_logits: dict[str, float] = field(default_factory=dict)
    # One distribution per permutation ordering, before averaging. Needed for
    # order-invariance metrics, which measure spread across orderings directly
    # rather than reading it back out of the aleatoric/epistemic split.
    per_permutation: list[dict[str, float]] = field(default_factory=list)
