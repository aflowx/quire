"""Shared-prefix helpers used by both engines."""

from __future__ import annotations


def share_question_prefix(prefix_ids: list[int], plans: list) -> tuple[list[int], list]:
    """Move a lone question's shared suffix tokens into the prefix.

    `plans` is [(question, labels, [(order, suffix_ids), ...]), ...]. Only a
    single question with two or more orderings is changed: the longest common
    token prefix of its suffixes (always leaving at least one token per
    suffix) is appended to `prefix_ids` and cut from every suffix.
    """
    if len(plans) != 1 or len(plans[0][2]) < 2:
        return prefix_ids, plans
    q, labels, suffixes = plans[0]
    seqs = [s for _, s in suffixes]
    n = common_prefix_len(seqs)
    if n == 0:
        return prefix_ids, plans
    return prefix_ids + seqs[0][:n], [(q, labels, [(o, s[n:]) for o, s in suffixes])]


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Longest common token prefix, always leaving at least one token in every sequence."""
    n, limit = 0, min(len(s) for s in seqs) - 1
    while n < limit and all(s[n] == seqs[0][n] for s in seqs):
        n += 1
    return n
