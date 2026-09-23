"""The wide-choice path: expand to per-option yes/no questions, collapse back."""

import pytest

from quire import wide
from quire.schema import Answer, Question


def _q(n):
    return Question("Which one?", {f"o{i}": f"option {i}" for i in range(n)})


def _yes(p):
    return Answer(choice="yes" if p >= 0.5 else "no", probabilities={"yes": p, "no": 1 - p},
                  aleatoric=0.0, epistemic=0.0, margin=abs(2 * p - 1), state_tokens=100, suffix_tokens=10)


def test_only_choices_beyond_the_cap_are_wide():
    assert not wide.is_wide(_q(26), cap=26)
    assert wide.is_wide(_q(27), cap=26)
    noul = Question("True?", {"true": "yes", "false": "no"}, kind="noul")
    assert not wide.is_wide(noul, cap=1)


def test_expand_asks_one_binary_question_per_option_in_order():
    q = _q(30)
    subs = wide.expand(q)
    assert len(subs) == 30
    assert all(s.option_ids == ["yes", "no"] for s in subs)
    assert "option 7" in subs[7].instructions and "Which one?" in subs[7].instructions


def test_collapse_normalises_yes_probabilities_into_one_distribution():
    q = _q(3)
    a = wide.collapse(q, [_yes(0.2), _yes(0.6), _yes(0.2)], latency_ms=1.0)
    assert a.choice == "o1"
    assert sum(a.probabilities.values()) == pytest.approx(1.0)
    assert a.probabilities["o1"] == pytest.approx(0.6)
    # the state is read once; every sub-question adds its own suffix
    assert a.state_tokens == 100 and a.suffix_tokens == 30


def test_decide_routes_wide_questions_through_one_inner_call():
    calls = []

    def inner(state, questions):
        calls.append(len(questions))
        out = []
        for q in questions:
            if q.option_ids == ["yes", "no"]:
                out.append(_yes(0.9 if "option 28" in q.instructions else 0.1))
            else:
                out.append(Answer(choice=q.option_ids[0], probabilities={o: 1 / q.n_options for o in q.option_ids},
                                  aleatoric=1.0, epistemic=0.0, margin=0.0))
        return out

    narrow, broad = _q(4), _q(40)
    answers = wide.decide(inner, "state", [narrow, broad], cap=26)
    assert calls == [41]                      # 1 narrow + 40 sub-questions, one call
    assert set(answers[0].probabilities) == set(narrow.option_ids)
    assert answers[1].choice == "o28"
