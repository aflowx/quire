"""Paired comparison of two selectors scored on the same questions.

Seventeen questions is few, and most per-question differences are zero, so a
difference in means says little on its own. The sign-flip permutation test asks
the right question directly: if the two selectors were interchangeable, each
per-question difference would be as likely negative as positive, so how often
does randomly flipping signs produce a mean difference this large? It assumes
nothing about the distribution and is exact at this sample size.
"""

from __future__ import annotations

import numpy as np

EXACT_LIMIT = 20  # 2**20 sign patterns is still instant; beyond that, sample


def paired_permutation_test(
    a: list[float], b: list[float], n_samples: int = 200_000, seed: int = 0
) -> dict[str, float]:
    """Two-sided test of mean(a - b) == 0 under sign-flipping."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be equal-length, non-empty, one-dimensional")
    diff = a - b
    observed = diff.mean()
    nonzero = diff[diff != 0]  # zeros are unchanged by a sign flip

    if len(nonzero) == 0:
        p_value = 1.0
    elif len(nonzero) <= EXACT_LIMIT:
        patterns = np.arange(2 ** len(nonzero))[:, None] >> np.arange(len(nonzero)) & 1
        sums = ((patterns * 2 - 1) * nonzero).sum(axis=1)
        p_value = float((np.abs(sums) >= abs(nonzero.sum()) - 1e-12).mean())
    else:
        # Chunked: hundreds of non-tied rows times 200k samples would not fit.
        rng = np.random.default_rng(seed)
        extreme, chunk = 0, 10_000
        for _ in range(0, n_samples, chunk):
            signs = rng.choice([-1.0, 1.0], size=(chunk, len(nonzero)))
            extreme += int((np.abs((signs * nonzero).sum(axis=1)) >= abs(nonzero.sum()) - 1e-12).sum())
        # +1: the observed labelling is itself one of the sign patterns.
        p_value = (extreme + 1) / (n_samples + 1)

    return {
        "mean_difference": float(observed),
        "p_value": p_value,
        "n": int(len(diff)),
        "a_better": int((diff > 0).sum()),
        "b_better": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }
