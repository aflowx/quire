"""The endpoint must report the tokens it read; JevBench prices from it."""

from fastapi.testclient import TestClient

from quire.daemon import build_app
from quire.engine import Engine

MODEL = "mlx-community/Qwen3.5-4B-MLX-8bit"


def test_usage_counts_state_once_plus_suffixes():
    engine = Engine(model_repo=MODEL, n_permutations=1)
    client = TestClient(build_app(engine))
    body = {"state": "Policy: trial users may export CSV. A trial user asks for CSV.",
            "questions": {"q": {"type": "noul", "instructions": "Is the action permitted?",
                                "criteria": {"true": "permitted", "false": "not permitted"}}}}
    r = client.post("/v1/systemone", json=body).json()
    used = r["usage"]["input_tokens"]
    assert used > 20, r["usage"]
    # state once + one suffix; a second question adds only its suffix
    body["questions"]["q2"] = body["questions"]["q"]
    r2 = client.post("/v1/systemone", json=body).json()
    assert used < r2["usage"]["input_tokens"] < 2 * used


def test_object_states_are_accepted_and_rendered_as_json():
    """JevBench sends some states as JSON objects; the server must accept them."""
    engine = Engine(model_repo=MODEL, n_permutations=1)
    client = TestClient(build_app(engine))
    body = {"state": {"request": "Return [3] as JSON.", "response": "{\"value\": 3}"},
            "questions": {"q": {"type": "noul", "instructions": "Does the response satisfy the request?"}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["answers"]["q"]["noul"] <= 1.0


def test_render_state_uses_json_not_a_python_repr():
    from quire.prompt import render_state
    text = render_state({"a": ["x", "y"], "b": None})
    assert '"a"' in text and "null" in text and "None" not in text
