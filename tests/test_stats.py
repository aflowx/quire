import pytest

from quire.stats import paired_permutation_test


def test_identical_selectors_are_not_distinguishable():
    out = paired_permutation_test([0.5, 1.0, 0.2], [0.5, 1.0, 0.2])
    assert out["p_value"] == 1.0 and out["ties"] == 3


def test_a_consistent_advantage_on_many_questions_is_significant():
    a = [1.0] * 12
    b = [0.5] * 12
    out = paired_permutation_test(a, b)
    # Only the all-plus and all-minus patterns are as extreme: 2 / 2**12.
    assert out["p_value"] == pytest.approx(2 / 2**12)
    assert out["a_better"] == 12


def test_an_advantage_on_three_questions_cannot_be_significant():
    """The floor this benchmark keeps bumping into: with k non-tied questions
    the smallest attainable two-sided p is 2 / 2**k, so three wins out of
    seventeen with fourteen ties can never clear 0.05 however large they are."""
    a = [1.0] * 3 + [0.7] * 14
    b = [0.0] * 3 + [0.7] * 14
    out = paired_permutation_test(a, b)
    assert out["p_value"] == pytest.approx(0.25)
    assert out["ties"] == 14


def test_mixed_wins_and_losses_cancel():
    out = paired_permutation_test([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0])
    assert out["mean_difference"] == 0.0 and out["p_value"] == 1.0


def test_the_test_is_symmetric_in_its_arguments():
    a, b = [0.9, 0.8, 0.4, 1.0, 0.6], [0.5, 0.9, 0.1, 0.7, 0.2]
    assert paired_permutation_test(a, b)["p_value"] == paired_permutation_test(b, a)["p_value"]


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [1.0, 2.0])


def test_the_sampled_path_agrees_with_the_exact_one_near_the_boundary():
    """Above 20 non-tied pairs the test samples sign patterns instead of
    enumerating them. 21 identical wins has an exact p of 2 / 2**21."""
    out = paired_permutation_test([1.0] * 21, [0.0] * 21, n_samples=50_000)
    assert out["p_value"] < 1e-3


def test_the_sampled_path_does_not_invent_significance():
    a = [1.0, 0.0] * 30
    b = [0.0, 1.0] * 30
    assert paired_permutation_test(a, b, n_samples=50_000)["p_value"] > 0.9
