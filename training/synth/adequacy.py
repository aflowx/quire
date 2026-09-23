"""adequacy scenarios (judge tier): does the response satisfy the request's explicit constraints?

Constraints are checkable by code, so gold is exact: word counts, JSON shape,
required/forbidden strings, numeric answers with units, case, length caps.
About half of responses satisfy everything; the rest break exactly one
constraint, often subtly. Twins fix or break that one constraint.
"""

from __future__ import annotations

import json
import random

from .common import CITIES, Scenario, est_tokens

WORDS = ["done", "approved", "ready", "shipped", "pending", "confirmed", "rejected", "later", "now", "ok", "all", "clear"]


def _json_array(r):
    n = r.randint(1, 4)
    nums = [r.randint(1, 40) for _ in range(n)]
    req = f"Return a JSON array containing exactly these integers in this order: {', '.join(map(str, nums))}. Output only the array."
    variants = [
        (json.dumps(nums), True, "correct"),
        (json.dumps({"values": nums}), False, "object instead of array"),
        (json.dumps(nums[::-1]) if n > 1 else json.dumps(nums + [0]), False, "wrong order/extra element"),
        (json.dumps([str(x) for x in nums]), False, "strings instead of integers"),
        ("Here you go: " + json.dumps(nums), False, "extra text outside the array"),
    ]
    return req, variants


def _exact_words(r):
    k = r.randint(2, 4)
    req = f"Reply with exactly {k} words."
    good = " ".join(r.sample(WORDS, k)).capitalize() + "."
    bad_more = " ".join(r.sample(WORDS, k + 1)).capitalize() + "."
    bad_less = " ".join(r.sample(WORDS, k - 1)).capitalize() + "." if k > 1 else "Ok"
    return req, [(good, True, "correct"), (bad_more, False, f"{k + 1} words"), (bad_less, False, f"{k - 1} words")]


def _unit_answer(r):
    km = r.randint(3, 400)
    req = f"State the distance from the depot to {r.choice(CITIES)} in kilometres, using the reference. Reference: the route is {km} km."
    return req, [(f"{km} km", True, "correct"), (f"{km} miles", False, "wrong unit"), (f"{km + 1} km", False, "off by one"),
                 (f"{km * 1000} m", False, "metres, not kilometres as asked")]


def _yes_no_from_reference(r):
    day = r.choice(["Sunday", "Monday", "Saturday"])
    closed = r.random() < 0.5
    ref = f"Reference: the branch is {'closed' if closed else 'open'} on {day}s."
    req = f"Is the branch open on {day}? Answer yes or no using the reference. {ref}"
    right = "no" if closed else "yes"
    wrong = "yes" if closed else "no"
    return req, [(right.capitalize() + ".", True, "correct"), (wrong.capitalize() + ".", False, "contradicts the reference"),
                 (f"{right.capitalize()}, and it is also {'closed' if not closed else 'open'} on {day}s.", False, "self-contradictory")]


def _earliest(r):
    a, b, c = sorted(r.sample(range(360, 1200, 5), 3))
    fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"  # noqa: E731
    labels = r.sample(["A", "B", "C"], 3)
    times = dict(zip(labels, [a, b, c]))
    earliest = min(times, key=times.get)
    latest = max(times, key=times.get)
    ref = "Reference departures: " + ", ".join(f"{k} at {fmt(v)}" for k, v in sorted(times.items())) + "."
    req = f"Which train departs earliest? Name it. {ref}"
    return req, [(f"Train {earliest}.", True, "correct"), (f"Train {latest}.", False, "latest, not earliest"),
                 (f"Train {[k for k in times if k not in (earliest, latest)][0]}.", False, "middle departure")]


def _length_cap(r):
    cap = r.choice([40, 60, 80])
    good = "Order confirmed; dispatch tomorrow." if cap >= 40 else "Confirmed."
    while len(good) > cap:
        good = good[: cap - 1].rstrip() + "."
    bad = good + " We appreciate your patience and will send tracking details as soon as the carrier scans the parcel."
    req = f"Confirm the order in at most {cap} characters."
    return req, [(good, True, "correct"), (bad, False, f"{len(bad)} characters, over the cap")]


def _must_include(r):
    ticket = f"TCK-{r.randint(1000, 9999)}"
    req = f"Acknowledge the report. Your reply must include the ticket number {ticket} and must not promise a date."
    return req, [(f"Thanks — logged as {ticket}; we'll update you when there is progress.", True, "correct"),
                 ("Thanks — logged; we'll update you when there is progress.", False, "ticket number missing"),
                 (f"Thanks — logged as {ticket}; it will be fixed by Friday.", False, "promises a date")]


GENS = [_json_array, _exact_words, _unit_answer, _yes_no_from_reference, _earliest, _length_cap, _must_include]


def build(r: random.Random, index: int, seed: int, long: bool = False) -> list[Scenario]:
    gen = r.choice(GENS)
    req, variants = gen(r)
    good = [v for v in variants if v[1]]
    bad = [v for v in variants if not v[1]]
    pick_good = r.random() < 0.5
    resp, ok, why = r.choice(good) if pick_good else r.choice(bad)
    other = r.choice(bad) if pick_good else r.choice(good)
    def render(response):
        if r.random() < 0.5:
            return json.dumps({"request": req, "response": response}, ensure_ascii=False)
        return f"Request: {req}\nResponse: {response}"
    state = render(resp)
    q = r.choice(["Does the response fully and correctly satisfy the request, including every explicit constraint?",
                  "Judge the response against the request: is it correct, complete and within all stated constraints?"])
    crit = {"true": "Correct, complete, and every explicit constraint is met.",
            "false": "Wrong, incomplete, or it breaks at least one explicit constraint."}
    base = Scenario(id=f"synth-adequacy-{seed}-{index}", family="adequacy", kind="noul", state=state, instructions=q,
                    criteria=crit, expected="true" if ok else "false", rationale=why, n_tokens_est=est_tokens(state),
                    tags=[gen.__name__.strip("_")])
    twin = Scenario(id=base.id + "-twin", family="adequacy", kind="noul", state=render(other[0]), instructions=q, criteria=crit,
                    expected="true" if other[1] else "false", rationale=f"Twin: {other[2]}", twin_of=base.id, flipped="response",
                    n_tokens_est=base.n_tokens_est, tags=base.tags)
    return [base, twin]
