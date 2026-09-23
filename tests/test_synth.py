"""The synthetic scenario generators: deterministic, twins flip the gold, distributions are exact."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))

from synth import adequacy, multi_hop, policy, probability, routing, temporal  # noqa: E402
from synth.common import rng_for  # noqa: E402

FAMILIES = {"policy": policy, "multi_hop": multi_hop, "probability": probability,
            "temporal_numeric": temporal, "routing": routing, "adequacy": adequacy}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_same_seed_same_scenarios(family):
    mod = FAMILIES[family]
    a = [s.row() for i in range(5) for s in mod.build(rng_for(7, family, i), i, 7)]
    b = [s.row() for i in range(5) for s in mod.build(rng_for(7, family, i), i, 7)]
    assert a == b


@pytest.mark.parametrize("family", ["policy", "multi_hop", "temporal_numeric", "adequacy"])
def test_a_twin_changes_one_fact_and_flips_the_gold(family):
    mod = FAMILIES[family]
    pairs = [mod.build(rng_for(3, family, i), i, 3) for i in range(40)]
    twins = [p for p in pairs if len(p) == 2]
    assert twins, "no twins generated"
    for base, twin in twins:
        assert twin.twin_of == base.id
        assert twin.expected != base.expected
        assert twin.state != base.state


def test_probability_gold_is_a_distribution_with_a_clear_argmax():
    for i in range(60):
        s = probability.build(rng_for(5, "probability", i), i, 5)[0]
        p = s.gold_probs
        assert abs(sum(p.values()) - 1) < 1e-3
        assert 0.55 <= max(p.values()) <= 0.86
        assert s.expected == max(p, key=p.get)
