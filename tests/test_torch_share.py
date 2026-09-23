"""The torch engine's shared-question path against its old path, on a tiny random
Qwen3.5-architecture model on CPU (no weights downloaded; fp32).

Needs torch and a local Qwen3.5 tokenizer (MLX snapshot or HF cache); skipped otherwise.
"""

import dataclasses
import glob
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from quire.labels import build_label_pool  # noqa: E402
from quire.schema import Question  # noqa: E402
from quire.torch_engine import TorchEngine  # noqa: E402


def _tokenizer_path():
    pats = [os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-8bit/snapshots/*"),
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")]
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


@pytest.fixture(scope="module")
def engine():
    path = _tokenizer_path()
    if path is None:
        pytest.skip("no local Qwen3.5 tokenizer")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    full = AutoConfig.from_pretrained(path)
    text = full.text_config if hasattr(full, "text_config") else full
    cfg = type(text)(**{**text.to_dict(), "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
                        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                        "linear_key_head_dim": 16, "linear_value_head_dim": 16})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    eng = TorchEngine.__new__(TorchEngine)
    for f in dataclasses.fields(TorchEngine):
        if f.default is not dataclasses.MISSING:
            setattr(eng, f.name, f.default)
    eng.tokenizer = AutoTokenizer.from_pretrained(path)
    eng.model = model
    eng.label_pool = build_label_pool(eng.tokenizer)
    return eng


CASES = [
    ("The refund window is 30 days. The order was placed 12 days ago.",
     Question(instructions="Is the refund still allowed?", criteria={"true": "Allowed.", "false": "Not allowed."}, kind="noul")),
    ("Ticket: the app crashes when I upload a photo from the gallery.",
     Question(instructions="Which team should handle this ticket?",
              criteria={"mobile": "Mobile app", "billing": "Billing", "web": "Web", "infra": "Infrastructure"}, kind="choice")),
]


@pytest.mark.parametrize("state,question", CASES)
def test_shared_question_matches_old_path(engine, state, question):
    engine.share_question = False
    a = engine.decide(state, [question])[0]
    engine.share_question = True
    b = engine.decide(state, [question])[0]
    for k in a.probabilities:
        assert abs(a.probabilities[k] - b.probabilities[k]) < 1e-4
    assert a.choice == b.choice
    # the same tokens are read; fewer are computed and reported
    assert b.state_tokens > a.state_tokens
    assert b.state_tokens + b.suffix_tokens < a.state_tokens + a.suffix_tokens


def test_multi_question_request_is_unchanged(engine):
    qs = [c[1] for c in CASES]
    engine.share_question = False
    a = engine.decide(CASES[0][0], qs)
    engine.share_question = True
    b = engine.decide(CASES[0][0], qs)
    for x, y in zip(a, b):
        assert x.probabilities == pytest.approx(y.probabilities, abs=1e-6)
        assert x.state_tokens == y.state_tokens
