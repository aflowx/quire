import numpy as np
import pytest

from quire.engine import Engine
from quire.schema import Question

MODEL_REPO = "mlx-community/Qwen3.5-0.8B-8bit"


@pytest.fixture(scope="module")
def engine():
    return Engine(model_repo=MODEL_REPO, n_permutations=3)


@pytest.mark.model
def test_decide_returns_one_answer_per_question(engine):
    questions = [
        Question("Which department?", {"billing": "money", "tech": "broken things"}),
        Question("Is it urgent?", {"yes": "urgent", "no": "not urgent"}),
    ]
    answers = engine.decide("My card was charged twice.", questions)
    assert len(answers) == 2


@pytest.mark.model
def test_probabilities_are_a_distribution_over_the_declared_options(engine):
    q = Question("Which department?", {"billing": "money", "tech": "broken things"})
    answer = engine.decide("My card was charged twice.", [q])[0]
    assert set(answer.probabilities) == {"billing", "tech"}
    assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-6


@pytest.mark.model
def test_choice_is_drawn_from_the_declared_options(engine):
    q = Question("Which department?", {"billing": "money", "tech": "broken things"})
    answer = engine.decide("My card was charged twice.", [q])[0]
    assert answer.choice in {"billing", "tech"}


@pytest.mark.model
def test_choice_is_the_argmax_of_the_reported_probabilities(engine):
    """The headline answer and the distribution must not disagree."""
    q = Question("Which?", {"a": "first thing", "b": "second thing", "c": "third thing"})
    answer = engine.decide("some state", [q])[0]
    assert answer.choice == max(answer.probabilities, key=answer.probabilities.get)


@pytest.mark.model
def test_an_obvious_question_is_answered_correctly_and_confidently(engine):
    q = Question(
        "Which department should handle this?",
        {"billing": "payment and invoice problems", "tech": "software bugs and outages"},
    )
    answer = engine.decide("I was charged twice for my subscription.", [q])[0]
    assert answer.choice == "billing"
    assert answer.margin > 0.2


@pytest.mark.model
def test_several_questions_in_one_call_get_their_own_answers(engine):
    """Answers must line up with the questions that produced them.

    decide() flattens every question's permutations into one fan-out and slices
    the rows back apart. An off-by-one there would attach each answer to the
    wrong question while every distribution still looked well-formed.
    """
    questions = [
        Question("What colour is the sky?", {"blue": "blue", "green": "green"}),
        Question("What colour is grass?", {"blue": "blue", "green": "green"}),
    ]
    answers = engine.decide("The sky is blue. Grass is green.", questions)
    assert answers[0].choice == "blue"
    assert answers[1].choice == "green"


@pytest.mark.model
def test_confidences_are_in_range(engine):
    q = Question("Which?", {"a": "first", "b": "second", "c": "third"})
    answer = engine.decide("ambiguous", [q])[0]
    assert 0.0 <= answer.aleatoric <= 1.0
    assert 0.0 <= answer.epistemic <= 1.0


@pytest.mark.model
def test_a_single_permutation_yields_zero_epistemic(engine):
    """Epistemic uncertainty measures disagreement BETWEEN runs."""
    single = Engine(model_repo=MODEL_REPO, n_permutations=1)
    q = Question("Which?", {"a": "first", "b": "second", "c": "third"})
    answer = single.decide("some state", [q])[0]
    assert answer.epistemic == 0.0


@pytest.mark.model
def test_per_permutation_distributions_are_retained(engine):
    q = Question("Which?", {"a": "first", "b": "second", "c": "third"})
    answer = engine.decide("some state", [q])[0]
    assert len(answer.per_permutation) == 3
    for run in answer.per_permutation:
        assert set(run) == {"a", "b", "c"}
        assert abs(sum(run.values()) - 1.0) < 1e-6


@pytest.mark.model
def test_debiasing_actually_changes_the_distribution(engine):
    """If the prior were trivial, debiasing would be dead code."""
    plain = Engine(model_repo=MODEL_REPO, n_permutations=1, use_debias=False)
    debiased = Engine(model_repo=MODEL_REPO, n_permutations=1, use_debias=True)
    q = Question("Which?", {"a": "first", "b": "second", "c": "third"})

    a = plain.decide("some state", [q])[0].probabilities
    b = debiased.decide("some state", [q])[0].probabilities
    assert max(abs(a[k] - b[k]) for k in a) > 1e-6


@pytest.mark.model
def test_the_content_free_prior_is_computed_once_per_schema_shape(engine):
    """It depends only on the schema, not the state, so it must cache."""
    probe = Engine(model_repo=MODEL_REPO, n_permutations=1, use_debias=True)
    calls = []
    original = probe._score_content_free

    def counting(key):
        calls.append(key)
        return original(key)

    probe._priors = type(probe._priors)(counting)

    q = Question("Which?", {"a": "first", "b": "second"})
    probe.decide("state one", [q])
    probe.decide("a completely different state", [q])
    assert len(calls) == 1, f"prior recomputed {len(calls)} times for one schema"


@pytest.mark.model
def test_different_instructions_get_their_own_prior(engine):
    """shape_key digests the instructions, so the scorer must honour them."""
    probe = Engine(model_repo=MODEL_REPO, n_permutations=1, use_debias=True)
    calls = []
    original = probe._score_content_free

    def counting(key):
        calls.append(key)
        return original(key)

    probe._priors = type(probe._priors)(counting)

    probe.decide("s", [Question("Which one?", {"a": "1", "b": "2"})])
    probe.decide("s", [Question("Pick the best:", {"a": "1", "b": "2"})])
    assert len(calls) == 2, "two different instruction texts shared one prior"
