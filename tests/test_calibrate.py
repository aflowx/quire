import numpy as np

from quire.calibrate import BucketedTemperature, apply_temperature, fit_temperature
from quire.metrics import ece


def _overconfident(n=600, seed=0):
    """Logits that are directionally right but far too peaked."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, size=n)
    logits = rng.normal(0, 1, size=(n, 3))
    logits[np.arange(n), labels] += 2.0
    return logits * 4.0, labels  # the *4 is the miscalibration


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_fit_temperature_recovers_a_value_above_one_when_overconfident():
    logits, labels = _overconfident()
    assert fit_temperature(logits, labels) > 1.5


def test_fit_temperature_stays_near_one_when_already_calibrated():
    """Labels drawn FROM softmax(logits) make those logits the true conditional
    distribution by construction, so the NLL-optimal temperature is 1.

    Note this cannot be built by adding a fixed boost to the true class: that
    yields ~88% accuracy against ~0.79 confidence, i.e. underconfident data,
    for which the optimal temperature is correctly below 1.
    """
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1.5, size=(4000, 3))
    probs = _softmax(logits)
    labels = np.array([rng.choice(3, p=p) for p in probs])
    assert 0.85 < fit_temperature(logits, labels) < 1.2


def test_fit_temperature_goes_below_one_when_underconfident():
    """The mirror of the overconfident case: too-flat logits need sharpening."""
    rng = np.random.default_rng(5)
    logits = rng.normal(0, 1.5, size=(2000, 3))
    probs = _softmax(logits)
    labels = np.array([rng.choice(3, p=p) for p in probs])
    assert fit_temperature(logits * 0.25, labels) < 0.9


def test_temperature_scaling_reduces_ece():
    logits, labels = _overconfident()
    raw = _softmax(logits)
    scaled = apply_temperature(logits, fit_temperature(logits, labels))
    assert ece(scaled, labels) < ece(raw, labels)


def test_temperature_scaling_never_changes_an_argmax():
    """Monotone, so it is free to apply: accuracy cannot move."""
    logits, labels = _overconfident()
    raw = _softmax(logits)
    scaled = apply_temperature(logits, 3.7)
    assert np.array_equal(raw.argmax(axis=1), scaled.argmax(axis=1))


def test_temperature_scaling_preserves_the_whole_ranking_not_just_the_top():
    """Monotonicity is a claim about every option, not only the winner.

    Downstream code gates on the top1-top2 margin, so the runner-up matters
    too. A transform that reordered anything below the top would break that
    while still passing an argmax-only check.
    """
    logits, _ = _overconfident(n=200, seed=5)
    raw_order = _softmax(logits).argsort(axis=1)
    scaled_order = apply_temperature(logits, 2.9).argsort(axis=1)
    assert np.array_equal(raw_order, scaled_order)


def test_higher_temperature_softens_and_lower_sharpens():
    logits, _ = _overconfident(n=100, seed=7)
    baseline = apply_temperature(logits, 1.0).max(axis=1)
    assert (apply_temperature(logits, 4.0).max(axis=1) < baseline).all()
    assert (apply_temperature(logits, 0.5).max(axis=1) > baseline).all()


def test_apply_temperature_returns_normalised_rows():
    logits, _ = _overconfident(n=50, seed=11)
    for temperature in (0.5, 1.0, 3.0):
        rows = apply_temperature(logits, temperature).sum(axis=1)
        assert np.allclose(rows, 1.0)


def test_bucketed_temperature_fits_per_option_count():
    logits3, labels3 = _overconfident(n=400, seed=1)
    logits5 = np.concatenate([logits3, logits3[:, :2] * 0.1], axis=1)

    model = BucketedTemperature()
    model.fit({3: (logits3, labels3), 5: (logits5, labels3)})

    assert set(model.temperatures) == {3, 5}
    assert model.temperature_for(3) != model.temperature_for(5)


def test_bucketed_temperature_falls_back_to_global_for_unseen_cardinality():
    logits, labels = _overconfident()
    model = BucketedTemperature()
    model.fit({3: (logits, labels)})
    assert model.temperature_for(97) == model.global_temperature


def test_bucketed_global_fallback_comes_from_the_best_populated_bucket():
    """Buckets have different option counts and cannot be pooled, so the
    fallback borrows from whichever bucket has the most evidence."""
    small_logits, small_labels = _overconfident(n=50, seed=2)
    big_logits, big_labels = _overconfident(n=900, seed=4)

    model = BucketedTemperature()
    model.fit({3: (small_logits, small_labels), 7: (big_logits, big_labels)})

    assert model.global_temperature == model.temperature_for(7)


def test_bucketed_transform_uses_the_matching_bucket():
    logits3, labels3 = _overconfident(n=400, seed=1)
    model = BucketedTemperature()
    model.fit({3: (logits3, labels3)})

    direct = apply_temperature(logits3, model.temperature_for(3))
    assert np.allclose(model.transform(logits3, 3), direct)


def test_fit_temperature_recovers_a_known_analytic_optimum():
    """The boost construction has a closed-form answer; check we land on it.

    For logits_i = z_i + b*1[i=y] with z ~ N(0,1) and a uniform prior, the
    exact posterior is p(y|logits) proportional to exp(b*l_y), i.e.
    softmax(b*logits). So the calibrated temperature is 1/b.

    This is the strongest check on the optimiser: the other tests assert a
    direction, this one asserts the value.
    """
    for boost in (2.0, 4.0):
        rng = np.random.default_rng(11)
        n = 4000
        labels = rng.integers(0, 3, size=n)
        logits = rng.normal(0, 1, size=(n, 3))
        logits[np.arange(n), labels] += boost
        fitted = fit_temperature(logits, labels)
        assert abs(fitted - 1.0 / boost) < 0.1, (
            f"boost={boost}: fitted {fitted:.4f}, analytic optimum {1 / boost:.4f}"
        )
