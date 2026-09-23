from quire.prompt import question_suffix, rotate, state_prefix
from quire.schema import Question


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


def test_state_prefix_contains_the_state_verbatim():
    prefix = state_prefix(FakeTokenizer(), "ERROR at line 42", style="plain")
    assert "ERROR at line 42" in prefix


def test_state_prefix_does_not_open_the_assistant_turn():
    assert not state_prefix(FakeTokenizer(), "s", style="plain").endswith("<assistant>")


def test_question_suffix_ends_so_next_token_is_the_answer():
    q = Question(instructions="Which?", criteria={"a": "first", "b": "second"})
    suffix = question_suffix(FakeTokenizer(), q, ["a", "b"], ["A", "B"], style="plain")
    assert suffix.endswith("Answer:")


def test_question_suffix_has_no_trailing_whitespace():
    """The label convention depends on this.

    Labels are looked up as " A", " B" -- the space-prefixed token. That is
    only the token the model emits if the prompt ends flush against the answer
    slot. A trailing space here would make the next token "A" (a different id),
    and the readout would silently score the wrong tokens.
    """
    q = Question(instructions="Which?", criteria={"a": "first", "b": "second"})
    suffix = question_suffix(FakeTokenizer(), q, ["a", "b"], ["A", "B"], style="plain")
    assert suffix == suffix.rstrip(), "trailing whitespace breaks the ' A' label convention"


def test_question_suffix_labels_options_by_letter_not_name():
    q = Question(instructions="Which?", criteria={"billing": "money", "tech": "things"})
    suffix = question_suffix(FakeTokenizer(), q, ["billing", "tech"], ["A", "B"], style="plain")
    assert "A. money" in suffix
    assert "B. things" in suffix


def test_question_suffix_renders_options_in_the_given_order_not_dict_order():
    """Permutation ensembling depends on `order` overriding insertion order."""
    q = Question(instructions="Which?", criteria={"billing": "money", "tech": "things"})
    suffix = question_suffix(FakeTokenizer(), q, ["tech", "billing"], ["A", "B"], style="plain")
    assert "A. things" in suffix
    assert "B. money" in suffix


def test_rotate_produces_distinct_cyclic_orderings():
    assert rotate(["a", "b", "c"], 0) == ["a", "b", "c"]
    assert rotate(["a", "b", "c"], 1) == ["b", "c", "a"]
    assert rotate(["a", "b", "c"], 2) == ["c", "a", "b"]


def test_rotate_handles_an_empty_list():
    assert rotate([], 1) == []


def test_the_answer_slot_sits_inside_an_open_think_block_by_default():
    """Qwen3.5's template opens a <think> block with the generation prompt, so
    the readout position is inside unclosed reasoning. That looks like a bug and
    was measured as one: closing it (scripts/ab_thinking.py, kev transfer-v4,
    764 items) moved accuracy +0.0092 (p = 0.40, not distinguishable) while ECE
    went 0.0448 -> 0.1032. Reading inside the open block is BETTER CALIBRATED,
    so the default stays and this test pins the reason."""

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            tail = "<think>\n\n</think>\n\n" if kwargs.get("enable_thinking") is False else "<think>\n"
            return messages[0]["content"] + "|assistant\n" + tail

    question = Question("pick", {"a": "first", "b": "second"})
    default = question_suffix(FakeTokenizer(), question, ["a", "b"], ["A", "B"], style="plain")
    closed = question_suffix(FakeTokenizer(), question, ["a", "b"], ["A", "B"],
                             close_thinking=True, style="plain")
    assert default.endswith("<think>\nAnswer:")
    assert closed.endswith("<think>\n\n</think>\n\nAnswer:")


def test_close_thinking_falls_back_when_the_template_rejects_the_flag():
    class OldTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[0]["content"] + "|assistant\n"

    out = question_suffix(OldTokenizer(), Question("pick", {"a": "1", "b": "2"}),
                          ["a", "b"], ["A", "B"], close_thinking=True, style="plain")
    assert out.endswith("Answer:")


def test_single_turn_produces_one_well_formed_user_turn():
    """The engine's original split rendered a COMPLETE user turn for the state
    and then opened another for the question, so the model saw two consecutive
    user turns with no assistant turn between them. single_turn closes the one
    the prefix opened instead."""

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            body = messages[0]["content"]
            tail = "<|im_end|>\n<|im_start|>assistant\n" if add_generation_prompt else "<|im_end|>\n"
            return f"<|im_start|>user\n{body}{tail}"

    tok = FakeTokenizer()
    question = Question("pick", {"a": "first", "b": "second"})

    split = state_prefix(tok, "S", style="plain") + question_suffix(tok, question, ["a", "b"], ["A", "B"], style="plain")
    assert split.count("<|im_start|>user") == 2, "the original shape is two user turns"

    joined = (state_prefix(tok, "S", single_turn=True, style="plain")
              + question_suffix(tok, question, ["a", "b"], ["A", "B"], single_turn=True, style="plain"))
    assert joined.count("<|im_start|>user") == 1
    assert joined.count("<|im_end|>") == 1
    assert "<state>\nS\n</state>" in joined and joined.endswith("Answer:")


def test_single_turn_keeps_the_state_as_a_shared_prefix():
    """The fan-out depends on every question sharing the same leading tokens."""

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            tail = "<|im_end|>\n<|im_start|>assistant\n" if add_generation_prompt else "<|im_end|>\n"
            return f"<|im_start|>user\n{messages[0]['content']}{tail}"

    prefix = state_prefix(FakeTokenizer(), "shared state", single_turn=True, style="plain")
    for instructions in ("first question", "second question"):
        suffix = question_suffix(FakeTokenizer(), Question(instructions, {"a": "1", "b": "2"}),
                                 ["a", "b"], ["A", "B"], single_turn=True, style="plain")
        assert (prefix + suffix).startswith(prefix)


# -- the quire style ------------------------------------------------------

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    """The real Qwen3.5 chat template: the quire style depends on its system role and thinking flag."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-4B-MLX-8bit")
    except Exception as exc:  # not cached / offline
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {exc}")

def test_quire_style_frames_the_question_as_a_test_and_ends_in_an_open_assistant_turn(tokenizer):
    q = Question(instructions="Is the request permitted?", criteria={"t": "permitted", "f": "not permitted"})
    prefix = state_prefix(tokenizer, "STATE", style="quire")
    suffix = question_suffix(tokenizer, q, ["f", "t"], ["A", "B"], style="quire")
    text = prefix + suffix
    assert "Each question is a test to run against the state" in text
    assert "<state>\nSTATE\n</state>" in prefix
    assert "<test>Is the request permitted?</test>" in suffix
    assert "A. not permitted\nB. permitted" in suffix           # the given order, not dict order
    assert text.rstrip().endswith("</think>")                  # thinking closed; next token is the answer
    assert "STATE" not in suffix                                # the state lives only in the shared prefix


def test_quire_style_prefix_is_identical_across_questions(tokenizer):
    a = state_prefix(tokenizer, "same state", style="quire")
    b = state_prefix(tokenizer, "same state", style="quire")
    assert a == b and a.endswith("</state>\n")
