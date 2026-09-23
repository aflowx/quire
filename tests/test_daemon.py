import pytest
from fastapi.testclient import TestClient

from quire.daemon import build_app
from quire.schema import Answer


class StubEngine:
    """Stands in for the real engine so the API contract is tested without weights."""

    model_repo = "stub"

    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        return [
            Answer(
                choice=q.option_ids[0],
                probabilities={o: 1.0 / q.n_options for o in q.option_ids},
                aleatoric=0.5,
                epistemic=0.1,
                margin=0.2,
                latency_ms=1.0,
            )
            for q in questions
        ]


def _client(engine=None):
    return TestClient(build_app(engine or StubEngine()))


def test_health_reports_the_loaded_model():
    body = _client().get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "stub"


def test_decide_returns_one_answer_per_question():
    body = _client().post(
        "/decide",
        json={
            "state": "the build failed",
            "questions": [
                {"instructions": "Which?", "criteria": {"a": "first", "b": "second"}},
                {"instructions": "Urgent?", "criteria": {"yes": "y", "no": "n"}},
            ],
        },
    ).json()
    assert len(body["answers"]) == 2


def test_decide_returns_the_full_distribution_not_just_the_argmax():
    """You cannot calibrate, debug or re-threshold without the distributions."""
    body = _client().post(
        "/decide",
        json={
            "state": "s",
            "questions": [{"instructions": "Which?", "criteria": {"a": "1", "b": "2"}}],
        },
    ).json()
    answer = body["answers"][0]
    assert set(answer["probabilities"]) == {"a", "b"}
    assert "aleatoric" in answer and "epistemic" in answer and "margin" in answer


def test_decide_passes_the_state_through_verbatim():
    """The state is the expensive shared input; mangling it would be silent."""
    engine = StubEngine()
    client = TestClient(build_app(engine))
    state = "ERROR at line 42\n  with trailing whitespace   "
    client.post(
        "/decide",
        json={"state": state, "questions": [{"instructions": "W?", "criteria": {"a": "1", "b": "2"}}]},
    )
    assert engine.calls[0][0] == state


def test_decide_preserves_option_order():
    """Option order is canonical: ensembling rotates it and re-indexes back."""
    engine = StubEngine()
    client = TestClient(build_app(engine))
    client.post(
        "/decide",
        json={
            "state": "s",
            "questions": [{"instructions": "W?", "criteria": {"zeta": "1", "alpha": "2", "mid": "3"}}],
        },
    )
    assert engine.calls[0][1][0].option_ids == ["zeta", "alpha", "mid"]


def test_decide_rejects_a_choice_question_with_one_option():
    response = _client().post(
        "/decide",
        json={"state": "s", "questions": [{"instructions": "Which?", "criteria": {"a": "1"}}]},
    )
    assert response.status_code == 422


def test_decide_rejects_an_unknown_question_kind():
    response = _client().post(
        "/decide",
        json={
            "state": "s",
            "questions": [
                {"instructions": "W?", "criteria": {"a": "1", "b": "2"}, "kind": "regression"}
            ],
        },
    )
    assert response.status_code == 422


def test_decide_accepts_a_noul_question_with_one_option():
    """Noul is an independent binary judgement, so one option is legitimate."""
    response = _client().post(
        "/decide",
        json={
            "state": "s",
            "questions": [{"instructions": "Keep?", "criteria": {"keep": "needed"}, "kind": "noul"}],
        },
    )
    assert response.status_code == 200


def test_client_and_daemon_agree_on_the_payload_shape():
    """Guards against the client and server drifting apart."""
    import json
    from unittest.mock import patch

    from quire import client as client_module

    engine = StubEngine()
    app_client = TestClient(build_app(engine))

    captured = {}

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        captured["body"] = request.data
        response = app_client.post("/decide", json=json.loads(request.data))
        return FakeResponse(response.content)

    with patch.object(client_module.urllib.request, "urlopen", fake_urlopen):
        result = client_module.decide(
            "some state", [{"instructions": "W?", "criteria": {"a": "1", "b": "2"}}]
        )

    assert len(result["answers"]) == 1
    assert set(result["answers"][0]["probabilities"]) == {"a", "b"}


def _system_one_client():
    """A client whose engine returns a fixed distribution, so the route's
    response SHAPE is tested without weights."""
    from fastapi.testclient import TestClient

    from quire.daemon import build_app
    from quire.schema import Answer

    class FakeEngine:
        model_repo = "fake"
        n_permutations = 2
        use_debias = False
        single_turn = True
        close_thinking = False

        def decide(self, state, questions):
            out = []
            for q in questions:
                ids = q.option_ids
                step = 1.0 / sum(range(1, len(ids) + 1))
                probs = {o: step * (len(ids) - i) for i, o in enumerate(ids)}
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}
                ranked = sorted(probs.values(), reverse=True)
                out.append(Answer(choice=max(probs, key=probs.get), probabilities=probs,
                                  aleatoric=0.0, epistemic=0.0,
                                  margin=ranked[0] - ranked[1], latency_ms=1.0))
            return out

    return TestClient(build_app(FakeEngine()))


def test_system_one_reports_noul_as_a_single_probability():
    """JevBench's typesafe adapter reads answer["noul"] as P(yes) and nothing
    else, so a noul answer must carry that key and a numeric value."""
    client = _system_one_client()
    body = {"state": "s", "questions": {"decision": {
        "type": "noul", "instructions": "is it?",
        "criteria": {"true": "yes it is", "false": "no it is not"}}}}
    answer = client.post("/v1/systemone", json=body).json()["answers"]["decision"]
    assert answer["type"] == "noul"
    assert isinstance(answer["noul"], float) and 0.0 <= answer["noul"] <= 1.0


def test_system_one_accepts_a_noul_question_with_no_criteria():
    client = _system_one_client()
    body = {"state": "s", "questions": {"decision": {"type": "noul", "instructions": "is it?"}}}
    assert client.post("/v1/systemone", json=body).status_code == 200


def test_system_one_returns_choice_probabilities_keyed_by_option_id():
    client = _system_one_client()
    body = {"state": "s", "questions": {"decision": {
        "type": "choice", "instructions": "which?",
        "criteria": {"billing": "money", "shipping": None}}}}
    answer = client.post("/v1/systemone", json=body).json()["answers"]["decision"]
    assert set(answer["probabilities"]) == {"billing", "shipping"}
    assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6
    assert answer["choice"] in answer["probabilities"]


def test_system_one_scores_are_the_mean_level_index():
    """TypeSafe's score is the probability-weighted level index, and the legend
    maps each index back to its description."""
    client = _system_one_client()
    body = {"state": "s", "questions": {"decision": {
        "type": "score", "instructions": "how bad?",
        "criteria": ["calm", "annoyed", "furious"]}}}
    answer = client.post("/v1/systemone", json=body).json()["answers"]["decision"]
    expected = sum(int(i) * p for i, p in answer["probabilities"].items())
    assert abs(answer["score"] - expected) < 1e-9
    assert answer["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}


def test_system_one_rejects_a_degenerate_question():
    client = _system_one_client()
    for bad in ({"type": "choice", "instructions": "q", "criteria": {"only": "one"}},
                {"type": "score", "instructions": "q", "criteria": ["single"]},
                {"type": "rating", "instructions": "q"}):
        assert client.post("/v1/systemone",
                           json={"state": "s", "questions": {"decision": bad}}).status_code == 422


def test_system_one_answers_several_questions_against_one_state():
    client = _system_one_client()
    body = {"state": "s", "questions": {
        "a": {"type": "noul", "instructions": "is it?"},
        "b": {"type": "choice", "instructions": "which?",
              "criteria": {"x": "ex", "y": "why"}}}}
    answers = client.post("/v1/systemone", json=body).json()["answers"]
    assert set(answers) == {"a", "b"}
    assert answers["a"]["type"] == "noul" and answers["b"]["type"] == "choice"
