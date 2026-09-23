import numpy as np

from quire.ensemble import combine


def test_agreeing_confident_runs_give_low_aleatoric_and_low_epistemic():
    runs = [np.array([0.95, 0.05]), np.array([0.94, 0.06])]
    result = combine(runs)
    assert result["aleatoric"] < 0.35
    assert result["epistemic"] < 0.05


def test_agreeing_flat_runs_are_aleatoric_not_epistemic():
    """The world is genuinely 50/50. Apply the business default; do not escalate."""
    runs = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    result = combine(runs)
    assert result["aleatoric"] > 0.95
    assert result["epistemic"] < 0.01


def test_disagreeing_confident_runs_are_epistemic():
    """Each run is sure, and they are sure of opposite things. Escalate."""
    runs = [np.array([0.99, 0.01]), np.array([0.01, 0.99])]
    result = combine(runs)
    assert result["epistemic"] > 0.5


def test_the_two_uncertainties_separate_the_cases_they_exist_to_separate():
    """This is the whole point of the module, asserted directly.

    Both situations produce an identical flat averaged distribution, so any
    single spread-based confidence scalar would score them the same. Only
    epistemic tells them apart.
    """
    ambiguous = combine([np.array([0.5, 0.5]), np.array([0.5, 0.5])])
    out_of_distribution = combine([np.array([1.0, 0.0]), np.array([0.0, 1.0])])

    assert np.allclose(ambiguous["probabilities"], out_of_distribution["probabilities"])
    assert abs(ambiguous["aleatoric"] - out_of_distribution["aleatoric"]) < 1e-9
    assert out_of_distribution["epistemic"] > ambiguous["epistemic"] + 0.5


def test_epistemic_never_exceeds_aleatoric():
    """Mutual information cannot exceed total entropy. A violation means the
    normalisation or the subtraction is wrong."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_runs, n_options = rng.integers(1, 5), rng.integers(2, 8)
        raw = rng.random((n_runs, n_options))
        runs = [r / r.sum() for r in raw]
        result = combine(runs)
        assert result["epistemic"] <= result["aleatoric"] + 1e-9


def test_a_single_run_has_exactly_zero_epistemic():
    """With one run there is no disagreement to measure."""
    result = combine([np.array([0.6, 0.3, 0.1])])
    assert result["epistemic"] == 0.0


def test_both_uncertainties_are_normalised_to_unit_range():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n_runs, n_options = rng.integers(1, 5), rng.integers(2, 10)
        raw = rng.random((n_runs, n_options))
        runs = [r / r.sum() for r in raw]
        result = combine(runs)
        assert 0.0 <= result["aleatoric"] <= 1.0 + 1e-9
        assert 0.0 <= result["epistemic"] <= 1.0 + 1e-9


def test_combine_averages_the_distributions():
    runs = [np.array([0.8, 0.2]), np.array([0.4, 0.6])]
    assert np.allclose(combine(runs)["probabilities"], [0.6, 0.4])


def test_combine_returns_a_normalised_distribution():
    runs = [np.array([0.8, 0.2]), np.array([0.4, 0.6]), np.array([0.1, 0.9])]
    assert abs(combine(runs)["probabilities"].sum() - 1.0) < 1e-9


def test_margin_is_top1_minus_top2():
    runs = [np.array([0.7, 0.2, 0.1])]
    assert abs(combine(runs)["margin"] - 0.5) < 1e-9


def test_margin_is_computed_on_the_averaged_distribution_not_per_run():
    """Two runs that each look decisive but disagree must yield a small margin."""
    runs = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]
    assert abs(combine(runs)["margin"]) < 1e-9
