"""One temperature per answer type, applied to a final distribution.

    p'(o) is proportional to p(o) ** (1 / T[type])

A temperature below 1 sharpens and above 1 softens. It divides log-
probabilities by a positive number, so it can never change the top answer.
The map is fitted on held-out data (bench/calibration/fit.py), never on
benchmark items.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

TYPES = ("choice", "noul", "score")
# The released map: fitted on held-out synthetic and kev development items, never on benchmark items.
DEFAULT = pathlib.Path(__file__).parent / "data" / "type_temperature.json"


@dataclass
class TypeTemperature:
    temperatures: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in TYPES})
    source: str = ""

    @classmethod
    def default(cls) -> "TypeTemperature":
        return cls.load(DEFAULT)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "TypeTemperature":
        d = json.loads(pathlib.Path(path).read_text())
        return cls(temperatures={t: float(d["temperatures"].get(t, 1.0)) for t in TYPES}, source=str(path))

    def apply(self, probabilities: dict[str, float], kind: str) -> dict[str, float]:
        t = self.temperatures.get(kind, 1.0)
        if t == 1.0:
            return dict(probabilities)
        logs = {o: math.log(max(p, 1e-12)) / t for o, p in probabilities.items()}
        top = max(logs.values())
        exp = {o: math.exp(v - top) for o, v in logs.items()}
        z = sum(exp.values())
        return {o: v / z for o, v in exp.items()}
