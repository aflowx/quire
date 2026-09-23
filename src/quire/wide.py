"""Wide choices: more options than the single-token label pool can name.

Jev caps a choice at 255 options and, above what one slot can read, "does a
2-stage system of scoring independently then making an explicit choice".
This is that, as one fan-out: every option becomes a yes/no sub-question over
the same prefilled state, the sub-questions run as one batch, and the
yes-probabilities are normalised into the choice distribution. Nothing is
truncated and no option is filtered, which is what benchmarks that score
coverage require.
"""

from __future__ import annotations

import numpy as np

from .schema import Answer, Question

YES, NO = "yes", "no"


def is_wide(question: Question, cap: int) -> bool:
    return question.kind != "noul" and question.n_options > cap


def expand(question: Question) -> list[Question]:
    """One binary sub-question per option, in canonical option order."""
    return [
        Question(
            instructions=f"{question.instructions}\nCandidate under consideration: {question.criteria[o]}",
            criteria={YES: "This candidate is the correct choice for the question.",
                      NO: "This candidate is not the correct choice."},
            kind="choice",
        )
        for o in question.option_ids
    ]


def collapse(question: Question, subs: list[Answer], latency_ms: float) -> Answer:
    """Normalise p(yes) across options into one distribution."""
    p_yes = np.array([a.probabilities[YES] for a in subs], dtype=float)
    p_yes = np.clip(p_yes, 1e-6, None)
    probs = p_yes / p_yes.sum()
    order = np.argsort(-probs)
    top1, top2 = probs[order[0]], probs[order[1]] if len(probs) > 1 else 0.0
    ent = float(-(probs * np.log(probs)).sum() / np.log(len(probs)))
    return Answer(
        choice=question.option_ids[int(order[0])],
        probabilities=dict(zip(question.option_ids, probs.tolist())),
        aleatoric=ent,
        epistemic=float(np.mean([a.epistemic for a in subs])),
        margin=float(top1 - top2),
        latency_ms=latency_ms,
        state_tokens=subs[0].state_tokens,
        suffix_tokens=sum(a.suffix_tokens for a in subs),
    )


def decide(inner_decide, state, questions: list[Question], cap: int) -> list[Answer]:
    """Run `inner_decide(state, flat_questions)` with wide choices expanded, then collapse.

    Any decider with the Engine.decide signature gets the wide path this way;
    all sub-questions still go to the inner decider in one call, so they share
    its prefilled state.
    """
    import time
    started = time.perf_counter()
    flat: list[Question] = []
    groups: list[tuple[int, int, bool]] = []
    for q in questions:
        if is_wide(q, cap):
            subs = expand(q)
            groups.append((len(flat), len(subs), True))
            flat.extend(subs)
        else:
            groups.append((len(flat), 1, False))
            flat.append(q)
    answers = inner_decide(state, flat)
    ms = (time.perf_counter() - started) * 1000
    return [collapse(q, answers[s:s + n], ms) if w else answers[s] for q, (s, n, w) in zip(questions, groups)]
