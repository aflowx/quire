import pytest

from quire.schema import Answer, Question


def test_question_requires_at_least_two_options():
    with pytest.raises(ValueError, match="at least 2 options"):
        Question(instructions="pick", criteria={"a": "only one"})


def test_question_option_ids_are_ordered_and_stable():
    q = Question(
        instructions="Which department?",
        criteria={"billing": "money", "tech": "broken things", "other": "none of these"},
    )
    assert q.option_ids == ["billing", "tech", "other"]
    assert q.n_options == 3


def test_noul_question_allows_a_single_option():
    q = Question(instructions="Is it needed?", criteria={"keep": "required"}, kind="noul")
    assert q.n_options == 1


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        Question(instructions="x", criteria={"a": "1", "b": "2"}, kind="regression")


def test_answer_probabilities_sum_to_one_for_choice():
    a = Answer(
        choice="billing",
        probabilities={"billing": 0.7, "tech": 0.3},
        aleatoric=0.61,
        epistemic=0.02,
        margin=0.4,
    )
    assert a.choice == "billing"
    assert abs(sum(a.probabilities.values()) - 1.0) < 1e-9
