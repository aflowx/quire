"""Prompt assembly.

Split into two pieces on purpose:

  state_prefix   -- everything that does NOT vary across questions. Prefilled
                    once; its cache is replicated across the fan-out.
  question_suffix -- the per-question tail, ending mid-turn so that the very
                    next token IS the answer.

Option descriptions are rendered against letter labels, never option names.

Two styles:

  quire  (default) a system instruction that frames each question as a test
         run against the state; the state, the test and the lettered options
         in tags; thinking closed; the answer is the first assistant token,
         scored as a bare letter ("A").
  plain  the original rendering: one user turn, the state in <state> tags,
         lettered options, ending flush with "Answer:" (no trailing space),
         scored as a space-prefixed letter (" A").
"""

from __future__ import annotations

from .schema import Question

QUIRE_SYSTEM = (
    "Each question is a test to run against the state. Work out whether the state passes it, "
    "then give the letter of the option that matches the result. Letter only."
)
STYLES = ("quire", "plain")


def render_state(state) -> str:
    """The state as prompt text. The System One format allows a string, a JSON
    object or a JSON array; non-strings are rendered as JSON, never as a
    language-specific repr."""
    if isinstance(state, str):
        return state
    import json
    return json.dumps(state, ensure_ascii=False, indent=2)


def first_token_style(style: str) -> bool:
    """Styles whose answer is the first assistant token, scored as a bare letter."""
    return style == "quire"


def _quire_frame(tokenizer) -> tuple[str, str]:
    """(head, tail) around the user content, thinking closed."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": QUIRE_SYSTEM}, {"role": "user", "content": "\x00"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    head, tail = rendered.split("\x00", 1)
    return head, tail


def state_prefix(tokenizer, state, single_turn: bool = False, style: str = "quire") -> str:
    """The shared, expensive part of the prompt. Prefilled once per request.

    `single_turn=False` (the original) renders a COMPLETE user turn, so the
    question suffix then opens a second one and the model sees two consecutive
    user turns with no assistant turn between them -- a shape no chat model is
    trained on. `single_turn=True` opens the turn and leaves it open for the
    suffix to close, which keeps the state as a shared prefix (so the fan-out
    still works) while producing a well-formed conversation.
    """
    state = render_state(state)
    if style == "quire":
        head, _ = _quire_frame(tokenizer)
        return f"{head}<state>\n{state}\n</state>\n"
    if single_turn:
        opened = tokenizer.apply_chat_template(
            [{"role": "user", "content": "\x00"}], tokenize=False, add_generation_prompt=False,
        )
        head = opened.split("\x00")[0]
        return f"{head}<state>\n{state}\n</state>\n\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<state>\n{state}\n</state>\n"}],
        tokenize=False,
        add_generation_prompt=False,
    )


def apply_template(tokenizer, body: str, close_thinking: bool) -> str:
    """Render one user turn with the assistant turn opened.

    Qwen3.5's chat template OPENS a `<think>` block when the generation prompt
    is added, so by default the answer slot lands *inside* an unclosed
    reasoning block -- the next token there is meant to be reasoning prose, not
    an answer. `close_thinking` passes `enable_thinking=False`, which emits
    `<think>\n\n</think>` and puts the slot where a direct answer belongs.

    Tokenizers whose template does not take the flag fall back to the default;
    the caller measures the difference rather than assuming it.
    """
    if close_thinking:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
    )


# Labels for the binary word readout: the first option is "yes".
WORD_LABELS = ["yes", "no"]


def question_suffix(
    tokenizer, question: Question, order: list[str], labels: list[str],
    close_thinking: bool = False, single_turn: bool = False, style: str = "quire",
) -> str:
    """The per-question tail, ending exactly where the next token is the answer.

    Under the quire style it ends with the assistant turn opened, so the next
    token is a bare letter; under plain it ends with "Answer:".

    `order` overrides the question's insertion order -- permutation ensembling
    passes a rotated ordering, and the caller re-indexes the result back to
    canonical order afterwards.
    """
    if style == "quire":
        _, tail = _quire_frame(tokenizer)
        lines = "\n".join(f"{label}. {question.criteria[option_id]}" for label, option_id in zip(labels, order))
        return f"<test>{question.instructions}</test>\n<options>\n{lines}\n</options>{tail}"
    if labels == WORD_LABELS:
        # Binary readout on the words themselves. There is no letter layer for
        # position bias to act on, so `order` is always the canonical one.
        yes_id, no_id = order
        body = (f"{question.instructions}\n\nAnswer yes if: {question.criteria[yes_id]}\n"
                f"Answer no if: {question.criteria[no_id]}\n\nReply yes or no.")
    else:
        lines = "\n".join(
            f"{label}. {question.criteria[option_id]}"
            for label, option_id in zip(labels, order)
        )
        body = f"{question.instructions}\n\n{lines}\n\nReply with one letter."
    if single_turn:
        # Close the turn the prefix opened, rather than starting a second one.
        rendered = apply_template(tokenizer, "\x00", close_thinking)
        tail = rendered.split("\x00", 1)[1]
        return f"{body}{tail}Answer:"
    return apply_template(tokenizer, body, close_thinking) + "Answer:"


def rotate(items: list[str], n: int) -> list[str]:
    """Cyclic rotation by n. Used to generate option orderings for ensembling."""
    if not items:
        return []
    n = n % len(items)
    return items[n:] + items[:n]
