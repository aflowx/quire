"""Evaluation metrics. Pure numpy so they are testable without weights.

An uncalibrated 0.85 model and a calibrated 0.85 model are different products:
only the second can be thresholded, gated or escalated. That is why ECE and
Brier are computed on every run rather than on request.
"""

from __future__ import annotations

import numpy as np


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error.

    Bin predictions by top-1 confidence, then average |accuracy - confidence|
    weighted by bin population.
    """
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        total += weight * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score. A proper scoring rule; lower is better.

    Ranges [0, 2]: 0 is a perfect one-hot prediction, 2 is maximal confidence
    in the wrong class. "Proper" means its optimum IS the true conditional
    probability, which is why it is trained and reported against.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def balanced_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-class recall. Class imbalance is endemic in routing."""
    recalls = []
    for cls in np.unique(labels):
        mask = labels == cls
        recalls.append((predictions[mask] == cls).mean())
    return float(np.mean(recalls))


def order_invariance(runs: list[np.ndarray]) -> float:
    """Worst-case shift in the ENSEMBLE winner's probability across orderings.

    Kept for continuity, but read it with two caveats, because on its own it
    ranks stability backwards in the regime it exists to police:

    1. It follows one class -- whichever the ensemble picked -- so it cannot
       see an answer flip. A flip happens when the distribution is near-flat,
       which is exactly when the winner's probability barely moves: an example
       that answers A, B, A, B at 0.50/0.49 scores 0.01, while a rock-steady
       example drifting 0.95 -> 0.25 without ever flipping scores 0.70.
    2. It is a max over examples, so it can only grow as rows are added. Never
       compare it between runs of different size.

    Use `answer_flip_rate` for the question this was meant to answer.
    """
    stacked = np.stack(runs)  # (n_runs, n_examples, n_options)
    top_class = stacked.mean(axis=0).argmax(axis=1)
    per_run = stacked[:, np.arange(stacked.shape[1]), top_class]
    return float((per_run.max(axis=0) - per_run.min(axis=0)).max())


def answer_flip_rate(runs: list[np.ndarray]) -> float:
    """Fraction of examples whose argmax is not unanimous across orderings.

    This is the position-bias measure that matters: a model that answers
    differently when the options are rotated has learned the layout, not the
    task, and the decision -- not the probability attached to it -- is what
    downstream code consumes. Being a mean rather than a max, it is also
    comparable between runs of different size.

    0.0 means every ordering produced the same answer on every example.
    """
    stacked = np.stack(runs)  # (n_runs, n_examples, n_options)
    if stacked.shape[0] < 2:
        return 0.0
    choices = stacked.argmax(axis=2)  # (n_runs, n_examples)
    return float((choices != choices[0]).any(axis=0).mean())


def oos_auroc(scores: np.ndarray, is_oos: np.ndarray) -> float:
    """AUROC for using a confidence score to detect out-of-scope inputs.

    A higher score should mean more in-scope. Computed from the rank statistic
    so there is no sklearn dependency. Returns NaN when either class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    is_oos = np.asarray(is_oos, dtype=bool)
    n_oos = int(is_oos.sum())
    n_in = int((~is_oos).sum())
    if n_oos == 0 or n_in == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so a constant score scores exactly 0.5.
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()

    rank_sum_in = ranks[~is_oos].sum()
    return float((rank_sum_in - n_in * (n_in + 1) / 2) / (n_in * n_oos))


def auroc_permutation_test(
    scores: np.ndarray, is_oos: np.ndarray, n: int = 20000, seed: int = 0
) -> dict[str, float]:
    """Is this AUROC distinguishable from chance at THIS class balance?

    AUROC has a fixed 0.5 null, which makes it tempting to read any departure
    as signal. The spread around that null is not fixed: it is governed by the
    smaller class. At 75 vs 19 the null 95% interval runs roughly [0.355,
    0.646], so a baseline at 0.40 is at chance, not below it -- and three
    baselines landing on the same side of 0.5 is what noise looks like, not
    three independent findings.

    Shuffles the labels, which holds the score vector and the class sizes fixed
    and destroys only the pairing. Returns the two-sided p-value against |A-0.5|
    and the null interval.
    """
    scores = np.asarray(scores, dtype=float)
    is_oos = np.asarray(is_oos, dtype=bool)
    observed = oos_auroc(scores, is_oos)
    rng = np.random.default_rng(seed)
    null = np.array(
        [oos_auroc(scores, rng.permutation(is_oos)) for _ in range(n)]
    )
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {
        "auroc": observed,
        "p_value": float((np.abs(null - 0.5) >= abs(observed - 0.5)).mean()),
        "null_low": float(lo),
        "null_high": float(hi),
        "n_permutations": n,
    }


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUROC under the natural convention: a HIGHER score means `positive`.

    `oos_auroc` is oriented the other way -- it was written for a confidence
    score where higher means in-scope -- and silently returns 1 - AUROC if you
    hand it a detector score with positive labels. This wrapper exists so that
    detector-shaped scenarios (guardrails, anything scoring "how likely is this
    the flagged class") cannot pick up that inversion.
    """
    return oos_auroc(scores, ~np.asarray(positive, dtype=bool))


def auroc_bootstrap_ci(
    scores: np.ndarray, is_oos: np.ndarray, n: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for AUROC by stratified bootstrap.

    Resamples within each class so every replicate keeps the original class
    sizes; an unstratified bootstrap of a 75/19 split occasionally draws almost
    no minority rows and widens the interval for no reason.
    """
    scores = np.asarray(scores, dtype=float)
    is_oos = np.asarray(is_oos, dtype=bool)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(is_oos), np.flatnonzero(~is_oos)
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(oos_auroc(scores[idx], is_oos[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def auroc_paired_bootstrap(
    a: np.ndarray, b: np.ndarray, is_oos: np.ndarray, n: int = 10000, seed: int = 0
) -> dict[str, float]:
    """AUROC(a) - AUROC(b) on the SAME rows, with a stratified bootstrap.

    Two scorers evaluated on one set of rows share most of their sampling
    noise, so comparing their separate intervals badly understates what the
    data can resolve. Resampling rows once per replicate and scoring both keeps
    the pairing. p is two-sided: twice the smaller tail of the difference.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    is_oos = np.asarray(is_oos, dtype=bool)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(is_oos), np.flatnonzero(~is_oos)
    diffs = np.empty(n)
    for k in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[k] = oos_auroc(a[idx], is_oos[idx]) - oos_auroc(b[idx], is_oos[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": oos_auroc(a, is_oos) - oos_auroc(b, is_oos),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
    }


def abstention_stats(
    scores: np.ndarray, is_oos: np.ndarray, threshold: float
) -> dict[str, float]:
    """What a given margin threshold buys and costs.

    Gate on the margin rather than on low top-1: a single readout position
    cannot reconsider, so top-1 alone is a poor abstention signal.
    """
    scores = np.asarray(scores, dtype=float)
    is_oos = np.asarray(is_oos, dtype=bool)
    abstained = scores < threshold
    return {
        "threshold": float(threshold),
        # Of the genuinely out-of-scope inputs, how many did we refuse?
        "oos_recall": float(abstained[is_oos].mean()) if is_oos.any() else float("nan"),
        # Of the in-scope inputs, how many did we still answer?
        "in_scope_retention": float((~abstained)[~is_oos].mean())
        if (~is_oos).any()
        else float("nan"),
    }
