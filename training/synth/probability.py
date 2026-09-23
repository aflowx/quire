"""probability scenarios: countable evidence fixes the probability; gold_probs is exact.

Three mechanisms, each rendered differently:
  log     -- a table of past cases with attributes and outcomes plus a precise
             definition of "comparable"; p = fraction of comparable cases with the outcome.
  rates   -- independent stated per-component failure rates; P(at least one) = 1 - prod(1 - p_i).
  draw    -- sampling without replacement from a stated pool; hypergeometric.
Top probability is kept in [0.55, 0.85] so calibration matters and the argmax is well-defined.
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

from .common import CITIES, COMPANIES, Scenario, est_tokens, pad_to


LEVEL = {"level": "hard"}


def _log(r: random.Random) -> tuple[list[str], str, Fraction, str, str, str]:
    domain = r.choice([
        ("shipment", "arrived late", ["carrier", "service", "origin"], {"carrier": ["Kolt", "Brisa", "Nordvik"], "service": ["Express", "Standard"], "origin": CITIES[:6]}),
        ("loan application", "was approved", ["channel", "segment", "region"], {"channel": ["broker", "direct", "partner"], "segment": ["retail", "SME", "corporate"], "region": ["North", "South", "East"]}),
        ("support ticket", "was reopened", ["queue", "priority", "product"], {"queue": ["Tier 1", "Tier 2", "Escalations"], "priority": ["P1", "P2", "P3"], "product": ["Ledger", "Payroll", "Portal"]}),
    ])
    noun, outcome, attrs, values = domain
    n = r.randint(10, 16) if LEVEL["level"] == "easy" else r.randint(28, 48)
    rows = [{a: r.choice(values[a]) for a in attrs} for _ in range(n)]
    # the comparable definition fixes two attributes; make sure enough rows match
    fixed = r.sample(attrs, 2)
    target = {a: r.choice(values[a]) for a in fixed}
    for row in rows[: max(10, n // 3)]:
        row.update(target)
    r.shuffle(rows)
    comparable = [row for row in rows if all(row[a] == target[a] for a in fixed)]
    k = len(comparable)
    # choose outcomes so the comparable fraction lands in [0.55, 0.85] or its complement
    want = r.uniform(0.56, 0.84)
    yes = max(1, min(k - 1, round(want * k)))
    if Fraction(yes, k) == Fraction(1, 2):
        yes += 1
    outcomes = [True] * yes + [False] * (k - yes)
    r.shuffle(outcomes)
    it = iter(outcomes)
    for row in rows:
        row["_y"] = next(it) if all(row[a] == target[a] for a in fixed) else r.random() < r.uniform(0.2, 0.8)
    lines = [f"| {noun.upper()[:3]}-{4400 + i} | " + " | ".join(row[a] for a in attrs) + f" | {'yes' if row['_y'] else 'no'} |"
             for i, row in enumerate(rows)]
    table = (f"| id | {' | '.join(attrs)} | {outcome}? |\n|---|" + "---|" * (len(attrs) + 1) + "\n" + "\n".join(lines))
    new_case = {a: (target[a] if a in fixed else r.choice(values[a])) for a in attrs}
    definition = (f"Definition. A past {noun} is *comparable* to the new one when its {fixed[0]} and {fixed[1]} both match "
                  f"the new {noun}'s. Other attributes are not part of the definition.")
    sections = [f"Historical {noun} log ({len(rows)} records)\n" + table, definition,
                f"New {noun}: " + ", ".join(f"{a} {new_case[a]}" for a in attrs) + "."]
    p = Fraction(yes, k)
    rationale = f"Comparable = {fixed[0]}={target[fixed[0]]} and {fixed[1]}={target[fixed[1]]}: {k} records, {yes} {outcome} → p = {yes}/{k} = {float(p):.4f}."
    q = f"Will the new {noun} be one that {outcome}? Give probabilities that reflect the log and the definition of comparable."
    return sections, q, p, rationale, "log", (f"The new {noun} {outcome}.", f"The new {noun} does not turn out to be one that {outcome}.")


def _rates(r: random.Random) -> tuple[list[str], str, Fraction, str, str, str]:
    company = r.choice(COMPANIES)
    parts = r.sample(["power supply", "primary pump", "controller board", "coolant loop", "network link", "backup battery"], 2 if LEVEL["level"] == "easy" else r.randint(2, 4))
    rates = [Fraction(r.choice([2, 5, 8, 10, 12, 15, 20, 25]), 100) for _ in parts]
    p_none = math.prod(1 - x for x in rates)
    p_any = 1 - p_none
    # keep within band by resampling
    tries = 0
    while not (Fraction(55, 100) <= max(p_any, 1 - p_any) <= Fraction(85, 100)) and tries < 50:
        rates = [Fraction(r.choice([2, 5, 8, 10, 12, 15, 20, 25, 30]), 100) for _ in parts]
        p_none = math.prod(1 - x for x in rates); p_any = 1 - p_none; tries += 1
    lines = "\n".join(f"- {p}: {float(x) * 100:.0f}% probability of a fault during the window" for p, x in zip(parts, rates))
    sections = [f"{company} — maintenance window risk sheet",
                "Component fault probabilities for the 72-hour window, assessed independently (faults in different components are independent events):\n" + lines,
                "A superseded version of this sheet listed the controller board at 40%; that figure was corrected after the firmware update and no longer applies.",
                "The window is declared *disrupted* if at least one listed component faults."]
    q = "Will the window be disrupted? Give probabilities that reflect the stated rates."
    rationale = "P(no fault) = " + " × ".join(f"(1−{float(x):.2f})" for x in rates) + f" = {float(p_none):.4f}; P(disrupted) = {float(p_any):.4f}."
    return sections, q, p_any, rationale, "rates", ("At least one listed component faults; the window is disrupted.", "No listed component faults; the window is not disrupted.")


def _draw(r: random.Random) -> tuple[list[str], str, Fraction, str, str, str]:
    total = r.randint(6, 10) if LEVEL["level"] == "easy" else r.randint(10, 24)
    bad = r.randint(2, max(3, total // 3))
    n = 2 if LEVEL["level"] == "easy" else r.randint(2, 4)
    # P(no defective in n draws without replacement)
    p_none = Fraction(math.comb(total - bad, n), math.comb(total, n))
    p_any = 1 - p_none
    tries = 0
    while not (Fraction(55, 100) <= max(p_any, 1 - p_any) <= Fraction(85, 100)) and tries < 100:
        total = r.randint(10, 24); bad = r.randint(2, max(3, total // 3)); n = r.randint(2, 4)
        p_none = Fraction(math.comb(total - bad, n), math.comb(total, n)); p_any = 1 - p_none; tries += 1
    lot = f"LOT-{r.randint(1000, 9999)}"
    sections = [f"Inspection record for {lot}",
                f"The lot contains {total} units. The supplier's declaration, confirmed by the incoming audit, states that exactly {bad} of them are out of specification.",
                f"Sampling plan in force: {n} units are drawn at random without replacement and tested. The lot is rejected if any tested unit is out of specification.",
                f"An earlier plan drew {n + 1} units; it was withdrawn last quarter and does not apply."]
    q = "Will the lot be rejected? Give probabilities that reflect the sampling plan and the declared counts."
    rationale = f"P(no defective in {n} of {total} with {bad} bad) = C({total - bad},{n})/C({total},{n}) = {float(p_none):.4f}; P(rejected) = {float(p_any):.4f}."
    return sections, q, p_any, rationale, "draw", ("At least one tested unit is out of specification; the lot is rejected.", "No tested unit is out of specification; the lot is accepted.")


def build(r: random.Random, index: int, seed: int, long: bool = False, level: str = "hard") -> list[Scenario]:
    LEVEL["level"] = level
    mech = r.choice([_log, _log, _rates, _draw])
    sections, q, p, rationale, tag, prop = mech(r)
    target = r.randint(1200, 2500) if long else r.randint(350, 900)
    body = pad_to(r, sections, target, heading="Note")
    state = "\n\n".join(body)
    p_yes = float(p)
    prop_true, prop_false = prop
    # Ask the complement half the time so the family is not skewed to "true":
    # same evidence, the question and the two propositions swap.
    if r.random() < 0.5:
        q = (q.replace("Will the window be disrupted?", "Will the window pass without any component fault?")
              .replace("Will the lot be rejected?", "Will the lot be accepted?"))
        if "be one that" in q:
            q = q.replace("be one that", "be one that does NOT turn out to be one that")
        prop_true, prop_false = prop_false, prop_true
        p_yes = 1 - p_yes
        rationale += " (question asks the complement)"
    expected = "true" if p_yes > 0.5 else "false"
    criteria = {"true": prop_true, "false": prop_false}
    base = Scenario(id=(f"synth-prob-{seed}-{index}" if level == "hard" else f"synth-prob-{level}-{seed}-{index}"), family="probability", kind="noul", state=state, instructions=q,
                    criteria=criteria, expected=expected, rationale=rationale,
                    gold_probs={"true": round(p_yes, 4), "false": round(1 - p_yes, 4)},
                    n_tokens_est=est_tokens(state), tags=[tag, "long" if long else "short", level])
    return [base]  # twins for this family come from regeneration; flipping counts would mean re-rendering the table
