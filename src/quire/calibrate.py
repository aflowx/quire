"""Temperature scaling (Guo et al. 2017).

Raw softmax over option logits is badly overconfident -- it will say 0.95 and
be right 70% of the time. One parameter, fitted by minimising NLL on a held-out
split, fixes most of it for ~500 rows and a second of CPU.

Because the transform is monotone it cannot change any argmax, so accuracy is
untouched and it is free to apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax the logits at the given temperature."""
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)
) -> float:
    """Minimise negative log-likelihood over a single scalar temperature."""

    def nll(temperature: float) -> float:
        probs = apply_temperature(logits, temperature)
        picked = probs[np.arange(len(labels)), labels]
        return float(-np.log(np.clip(picked, 1e-12, None)).mean())

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


@dataclass
class BucketedTemperature:
    """A temperature per option count, plus a global fallback.

    With runtime-defined schemas there is no fixed class set to calibrate per
    class -- the classes change every request. Option count is the strongest
    stable bucketing key, because overconfidence grows with cardinality.
    """

    temperatures: dict[int, float] = field(default_factory=dict)
    global_temperature: float = 1.0

    def fit(self, buckets: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        for n_options, (logits, labels) in buckets.items():
            self.temperatures[n_options] = fit_temperature(logits, labels)
        # Fallback for cardinalities never seen at fit time. Buckets have
        # different option counts so their logits cannot be concatenated;
        # use the best-populated bucket's temperature instead.
        largest = max(buckets, key=lambda k: len(buckets[k][1]))
        self.global_temperature = self.temperatures[largest]

    def temperature_for(self, n_options: int) -> float:
        return self.temperatures.get(n_options, self.global_temperature)

    def transform(self, logits: np.ndarray, n_options: int) -> np.ndarray:
        return apply_temperature(logits, self.temperature_for(n_options))
