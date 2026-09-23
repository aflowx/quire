import os
import subprocess
import sys

import numpy as np

from quire.debias import CONTENT_FREE_STATE, PriorCache, debias, shape_key
from quire.schema import Question


def test_debias_removes_a_uniform_shift():
    logits = np.array([2.0, 1.0, 0.5])
    prior = np.array([1.0, 1.0, 1.0])
    assert np.allclose(debias(logits, prior), np.array([1.0, 0.0, -0.5]))


def test_debias_cancels_a_label_specific_bias():
    """The model loves position A by +3. After debiasing, B should win."""
    prior = np.array([3.0, 0.0, 0.0])
    logits = np.array([3.5, 1.0, 0.0])
    assert debias(logits, prior).argmax() == 1


def test_debias_leaves_the_answer_alone_when_there_is_no_bias():
    prior = np.zeros(3)
    logits = np.array([0.2, 2.0, 0.1])
    assert np.allclose(debias(logits, prior), logits)


def test_prior_cache_computes_once_per_schema_shape():
    calls = []

    def fake_scorer(key):
        calls.append(key)
        return np.zeros(3)

    cache = PriorCache(fake_scorer)
    cache.get("choice:3")
    cache.get("choice:3")
    cache.get("choice:5")

    assert calls == ["choice:3", "choice:5"]


def test_prior_cache_returns_the_same_array_for_a_repeated_shape():
    cache = PriorCache(lambda _: np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(cache.get("k"), cache.get("k"))


def test_prior_cache_clear_forces_recomputation():
    calls = []
    cache = PriorCache(lambda k: calls.append(k) or np.zeros(2))
    cache.get("k")
    cache.clear()
    cache.get("k")
    assert calls == ["k", "k"]


def test_shape_key_separates_different_cardinalities_and_instructions():
    two = Question("Which?", {"a": "1", "b": "2"})
    three = Question("Which?", {"a": "1", "b": "2", "c": "3"})
    other = Question("Pick one:", {"a": "1", "b": "2"})
    assert shape_key(two) != shape_key(three)
    assert shape_key(two) != shape_key(other)


def test_shape_key_covers_everything_the_content_free_probe_renders():
    """This test used to assert the opposite, on the reasoning that "the prior
    depends on shape, not content -- that is what makes it cacheable". The
    sentence is plausible and the implementation does not match it:
    `_score_content_free` blanks the STATE and renders the real descriptions,
    so for this cache the descriptions ARE shape. The state is the only thing
    the prior is independent of, and the state is the only thing left out.

    The collision this allowed landed under the format-overfitting claim: the
    perturbed split in bench/perturb.py restyles descriptions while keeping the
    instructions and option count, so it reused the authored split's prior.
    """
    a = Question("Which?", {"a": "money problems", "b": "broken things"})
    b = Question("Which?", {"a": "entirely different", "b": "text here"})
    assert shape_key(a) != shape_key(b)


def test_shape_key_is_stable_across_hash_seeds():
    """hash() is randomised per process; a fitted prior must stay identifiable.

    The calibration artifact, the prompt template and the model revision are
    one versioned unit. A key that changes every run silently breaks that.
    """
    code = (
        "from quire.debias import shape_key\n"
        "from quire.schema import Question\n"
        "print(shape_key(Question('Which?', {'a': '1', 'b': '2'})))\n"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, check=True,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] and outputs[0] == outputs[1], (
        f"shape_key varies with PYTHONHASHSEED: {outputs}"
    )


def test_content_free_state_is_actually_content_free():
    assert CONTENT_FREE_STATE.strip() != ""
    assert len(CONTENT_FREE_STATE) < 10


def test_shape_key_ignores_the_option_ids_because_the_prompt_does():
    """Renaming ids must NOT invalidate the prior: question_suffix never
    renders them, so the probe's prompt is byte-identical either way."""
    base = Question("pick one", {"a": "money problems", "b": "broken things"})
    renamed = Question("pick one", {"opt0": "money problems", "opt1": "broken things"})
    assert shape_key(base) == shape_key(renamed)


def test_shape_key_separates_schemas_whose_options_are_ordered_differently():
    """The probe renders options in canonical order, so canonical order is part
    of the prompt it measured."""
    forward = Question("pick one", {"a": "money problems", "b": "broken things"})
    reverse = Question("pick one", {"b": "broken things", "a": "money problems"})
    assert shape_key(forward) != shape_key(reverse)
