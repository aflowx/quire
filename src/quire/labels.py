"""Single-token option labels, derived from the tokenizer in use.

Labels are single characters (A, B, C...) rather than option names. Measured
at +10.0 pp [+8.3, +11.7] over named options: named options vary in token
length, which biases toward short ones, and carry lexical priors from
pretraining. Single characters are uniform-length with a uniform prior.

The pool is DERIVED, never assumed. Qwen3.5's vocabulary is 248320 tokens and
multimodal; which characters are single tokens there is an empirical question.
"""

from __future__ import annotations

CANDIDATE_POOL: list[str] = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [str(d) for d in range(10)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)


def build_label_pool(tokenizer) -> list[str]:
    """Return the candidates that encode to exactly one token with a leading space.

    The leading space matters: the prompt ends with "Answer:" so the token the
    model actually emits at the answer slot is " A", not "A".
    """
    pool = [c for c in CANDIDATE_POOL if _is_single_token(tokenizer, c)]
    if not pool:
        raise RuntimeError(
            "no single-token labels found for this tokenizer; the logit readout "
            "cannot be defined. Check that the tokenizer is the one paired with "
            "the model."
        )
    return pool


def label_token_ids(tokenizer, labels: list[str], bare: bool = False) -> list[int]:
    """Token id for each label, in order. Assumes build_label_pool has run.

    `bare` scores "A" rather than " A": the slot is the first token of the
    assistant turn, with nothing before it on the line.
    """
    ids = []
    for label in labels:
        encoded = tokenizer.encode(label if bare else " " + label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"label {label!r} is not single-token for this tokenizer "
                f"(encoded to {len(encoded)} tokens)"
            )
        ids.append(encoded[0])
    return ids


def pick_labels(n: int, pool: list[str]) -> list[str]:
    """The first n labels from the pool.

    Guards both ends. A negative n would otherwise slice as "all but the last
    |n|" and silently return a plausible but wrong list, which is the exact
    failure mode this module exists to prevent.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n > len(pool):
        raise ValueError(
            f"{n} options exceeds the {len(pool)} single-token labels available; "
            "use hierarchical routing or an embedding prefilter"
        )
    return pool[:n]


def _is_single_token(tokenizer, char: str) -> bool:
    try:
        return len(tokenizer.encode(" " + char, add_special_tokens=False)) == 1
    except Exception:
        return False
