import numpy as np
import pytest

from quire.metrics import (
    answer_flip_rate,
    auroc,
    auroc_permutation_test,
    balanced_accuracy,
    brier_score,
    ece,
    oos_auroc,
    order_invariance,
)


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    probs = np.array([[0.5, 0.5]] * 100)
    labels = np.array([0] * 50 + [1] * 50)
    assert ece(probs, labels, n_bins=10) < 1e-9


def test_ece_is_large_for_confidently_wrong_predictions():
    probs = np.array([[0.99, 0.01]] * 100)
    labels = np.array([1] * 100)
    assert ece(probs, labels, n_bins=10) > 0.95


def test_ece_is_stable_across_bin_counts_when_perfectly_calibrated():
    """Bin count is a free parameter; a calibrated set must score ~0 for any of them."""
    probs = np.array([[0.5, 0.5]] * 100)
    labels = np.array([0] * 50 + [1] * 50)
    for n_bins in (5, 10, 15, 50):
        assert ece(probs, labels, n_bins=n_bins) < 1e-9, f"failed at n_bins={n_bins}"


def test_brier_is_zero_for_perfect_one_hot_predictions():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert brier_score(probs, labels) == 0.0


def test_brier_penalises_confident_errors_more_than_hedged_ones():
    labels = np.array([0])
    confident_wrong = brier_score(np.array([[0.0, 1.0]]), labels)
    hedged = brier_score(np.array([[0.5, 0.5]]), labels)
    assert confident_wrong > hedged


def test_brier_stays_within_its_theoretical_bounds():
    """Multiclass Brier is in [0, 2]; 2 is maximally confident and wrong."""
    labels = np.array([0])
    assert brier_score(np.array([[0.0, 1.0]]), labels) == 2.0
    assert 0.0 <= brier_score(np.array([[0.3, 0.7]]), labels) <= 2.0


def test_balanced_accuracy_ignores_class_imbalance():
    # 90 of class 0 all correct, 10 of class 1 all wrong.
    preds = np.array([0] * 100)
    labels = np.array([0] * 90 + [1] * 10)
    assert abs(balanced_accuracy(preds, labels) - 0.5) < 1e-9


def test_balanced_accuracy_handles_a_class_never_predicted():
    """Routing datasets routinely contain classes the model never picks."""
    preds = np.array([0, 0, 0, 0])
    labels = np.array([0, 0, 1, 2])
    # class 0 recall 1.0, class 1 recall 0.0, class 2 recall 0.0
    assert abs(balanced_accuracy(preds, labels) - 1 / 3) < 1e-9


def test_order_invariance_is_zero_when_permutations_agree():
    runs = [np.array([[0.7, 0.3]]), np.array([[0.7, 0.3]])]
    assert order_invariance(runs) == 0.0


def test_order_invariance_reports_max_shift_across_permutations():
    runs = [np.array([[0.9, 0.1]]), np.array([[0.4, 0.6]])]
    assert abs(order_invariance(runs) - 0.5) < 1e-9


def test_order_invariance_reports_the_worst_example_not_the_average():
    """One badly position-biased example must not be hidden by stable ones."""
    runs = [
        np.array([[0.7, 0.3], [0.9, 0.1]]),
        np.array([[0.7, 0.3], [0.2, 0.8]]),
    ]
    assert abs(order_invariance(runs) - 0.7) < 1e-9


def test_order_invariance_handles_more_than_two_runs():
    runs = [
        np.array([[0.9, 0.1]]),
        np.array([[0.6, 0.4]]),
        np.array([[0.3, 0.7]]),
    ]
    assert abs(order_invariance(runs) - 0.6) < 1e-9


def test_answer_flip_rate_is_zero_when_every_ordering_agrees():
    runs = [np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([[0.7, 0.3], [0.4, 0.6]])]
    assert order_invariance(runs) == pytest.approx(0.2)
    assert answer_flip_rate(runs) == 0.0


def test_answer_flip_rate_catches_a_flip_that_order_invariance_misses():
    """The case that motivated the metric.

    Example 0 flips its answer on every run; example 1 keeps the same answer
    throughout but its probability swings 0.61. order_invariance follows the
    ENSEMBLE winner's probability, so it reports the stable example's swing and
    scores the unstable one at 0.01 -- exactly backwards.

    Three options, not two: with two, any drop past 0.5 IS a flip, so a large
    swing and a stable answer cannot coexist.
    """
    runs = [
        np.array([[0.50, 0.49, 0.01], [0.95, 0.03, 0.02]]),
        np.array([[0.49, 0.50, 0.01], [0.34, 0.33, 0.33]]),
    ]
    assert order_invariance(runs) == pytest.approx(0.61)
    assert answer_flip_rate(runs) == pytest.approx(0.5)


def test_answer_flip_rate_is_one_when_every_example_flips():
    runs = [np.array([[0.6, 0.4], [0.6, 0.4]]), np.array([[0.4, 0.6], [0.4, 0.6]])]
    assert answer_flip_rate(runs) == 1.0


def test_answer_flip_rate_counts_an_example_once_however_many_runs_disagree():
    runs = [
        np.array([[0.7, 0.2, 0.1]]),
        np.array([[0.2, 0.7, 0.1]]),
        np.array([[0.1, 0.2, 0.7]]),
    ]
    assert answer_flip_rate(runs) == 1.0


def test_answer_flip_rate_is_a_mean_so_it_does_not_grow_with_the_row_count():
    """order_invariance is a max over examples and can only rise as rows are
    added, which makes it uncomparable across runs of different size. This
    metric has to stay flat when stable rows are appended."""
    flipping = [np.array([[0.6, 0.4]]), np.array([[0.4, 0.6]])]
    stable = [np.array([[0.9, 0.1]]), np.array([[0.9, 0.1]])]
    small = [np.concatenate([f, s]) for f, s in zip(flipping, stable)]
    large = [np.concatenate([f] + [s] * 3) for f, s in zip(flipping, stable)]
    assert answer_flip_rate(small) == pytest.approx(0.5)
    assert answer_flip_rate(large) == pytest.approx(0.25)
    assert order_invariance(small) == order_invariance(large)


def test_answer_flip_rate_is_zero_for_a_single_run():
    assert answer_flip_rate([np.array([[0.5, 0.5]])]) == 0.0


def test_auroc_permutation_test_calls_a_strong_separation_significant():
    scores = np.concatenate([np.linspace(0.6, 1.0, 40), np.linspace(0.0, 0.4, 10)])
    is_oos = np.concatenate([np.zeros(40, bool), np.ones(10, bool)])
    out = auroc_permutation_test(scores, is_oos, n=2000)
    assert out["auroc"] == 1.0
    assert out["p_value"] < 0.01


def test_auroc_permutation_test_calls_a_modest_departure_chance():
    """The TC39 case. At 75 vs 19 the null interval is wide enough that 0.40
    is not distinguishable from 0.50, so three baselines landing below chance
    is what noise looks like rather than three findings."""
    rng = np.random.default_rng(0)
    scores = rng.random(94)
    is_oos = np.zeros(94, bool)
    is_oos[:19] = True
    out = auroc_permutation_test(scores, is_oos, n=4000)
    assert out["p_value"] > 0.05
    assert out["null_low"] < 0.40 and out["null_high"] > 0.60


def test_auroc_permutation_test_null_interval_narrows_as_the_rare_class_grows():
    """The spread is governed by the smaller class, which is why a fixed 0.5
    null cannot be read without one."""
    rng = np.random.default_rng(1)
    def width(n_rare, n_total):
        is_oos = np.zeros(n_total, bool)
        is_oos[:n_rare] = True
        out = auroc_permutation_test(rng.random(n_total), is_oos, n=2000)
        return out["null_high"] - out["null_low"]
    assert width(40, 200) < width(8, 200)


def test_auroc_is_oriented_so_that_a_higher_score_means_positive():
    """The guardrail scenario's first run reported every system below 0.5
    because oos_auroc has the opposite convention. This pins the fix."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    positive = np.array([True, True, False, False])
    assert auroc(scores, positive) == 1.0
    assert oos_auroc(scores, positive) == 0.0


def test_auroc_and_oos_auroc_are_reflections_of_each_other():
    rng = np.random.default_rng(0)
    scores = rng.random(60)
    positive = rng.random(60) < 0.3
    assert auroc(scores, positive) == pytest.approx(1 - oos_auroc(scores, positive))
