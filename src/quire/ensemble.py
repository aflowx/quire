"""Permutation ensembling, and splitting uncertainty into its two kinds.

Jev exposes one `confidence` scalar described as reflecting probability spread.
That cannot distinguish two situations demanding opposite responses:

  genuinely ambiguous  -> flat -> accept it, the world is uncertain
  out of distribution  -> flat -> escalate, the distribution is meaningless

So report both:

  aleatoric -- normalised entropy of the AVERAGED distribution. Irreducible.
  epistemic -- mutual information between the run-ensemble and the prediction,
               i.e. total entropy minus mean per-run entropy. Reducible.

Route on both: high aleatoric + low epistemic means trust it and apply the
default. High epistemic means escalate regardless of the top probability.
"""

from __future__ import annotations

import numpy as np


def combine(runs: list[np.ndarray]) -> dict:
    """Combine per-ordering distributions into one answer.

    Each run must already be re-indexed back to canonical option order.
    """
    stacked = np.stack([np.asarray(r, dtype=float) for r in runs])
    mean = stacked.mean(axis=0)

    total_entropy = _entropy(mean)
    mean_entropy = float(np.mean([_entropy(r) for r in stacked]))
    n_options = mean.shape[0]
    max_entropy = np.log(n_options) if n_options > 1 else 1.0

    ordered = np.sort(mean)[::-1]
    margin = float(ordered[0] - ordered[1]) if n_options > 1 else float(ordered[0])

    return {
        "probabilities": mean,
        "aleatoric": float(total_entropy / max_entropy),
        # Mutual information, normalised. Zero when every run agrees.
        "epistemic": float(max(0.0, total_entropy - mean_entropy) / max_entropy),
        "margin": margin,
    }


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    return float(-(p * np.log(p)).sum())
