"""long_policy-style scenarios: a policy document with interacting clauses and a case.

World: a request with attributes; a base rule (thresholds), an exception that
rescues one failing condition for a class of requesters, an amendment that
changes a threshold from an effective date, a definition that changes how one
attribute is computed, and a prohibition. The evaluator decides; the prose
scatters the clauses across a long document with irrelevant sections.
"""

from __future__ import annotations

import datetime as dt
import random

from .common import COMPANIES, DEPARTMENTS, Scenario, est_tokens, pad_to, person

DOMAINS = {
    "expense": {
        "action": "reimbursement of the claimed expense",
        "unit": "€", "amount_name": "claim amount",
        "categories": ["travel", "equipment", "training", "client hospitality", "software", "conference"],
        "prohibited": ["gifts", "fines", "personal subscriptions"],
        "exception_class": ("field engineers", "a field engineer"),
        "policy_title": "Expense Reimbursement Policy",
    },
    "discount": {
        "action": "the requested discount",
        "unit": "%", "amount_name": "discount rate",
        "categories": ["renewal", "new logo", "expansion", "pilot", "public sector", "reseller"],
        "prohibited": ["loss-leader", "unlisted bundle", "retroactive credit"],
        "exception_class": ("strategic accounts", "a strategic account"),
        "policy_title": "Commercial Discount Approval Policy",
    },
    "leave": {
        "action": "the requested leave",
        "unit": " days", "amount_name": "number of days requested",
        "categories": ["annual", "study", "unpaid", "carer", "sabbatical", "volunteering"],
        "prohibited": ["leave during a declared freeze period", "leave overlapping a disciplinary review"],
        "exception_class": ("shift workers", "a shift worker"),
        "policy_title": "Leave Approval Policy",
    },
}


def _fmt(unit: str, v: int) -> str:
    return f"{unit}{v:,}" if unit == "€" else f"{v}{unit}"


def build(r: random.Random, index: int, seed: int, long: bool = True) -> list[Scenario]:
    dom_name = r.choice(list(DOMAINS))
    d = DOMAINS[dom_name]
    company = r.choice(COMPANIES)
    unit = d["unit"]

    # thresholds
    base_limit = {"expense": r.choice([500, 750, 1000, 1500, 2000]),
                  "discount": r.choice([10, 12, 15, 20]),
                  "leave": r.choice([5, 8, 10, 12])}[dom_name]
    amended_limit = int(base_limit * r.choice([0.6, 0.75, 1.25, 1.5]))
    min_tenure = r.choice([3, 6, 12])
    notice_days = r.choice([5, 10, 14, 21])
    eff = dt.date(2026, r.randint(2, 8), r.randint(1, 28))
    amendment_first = r.random() < 0.5  # whether the amendment section appears before or after the base rule

    # the case
    cat = r.choice(d["categories"] + d["prohibited"]) if r.random() < 0.15 else r.choice(d["categories"])
    is_prohibited = cat in d["prohibited"]
    req_date = eff + dt.timedelta(days=r.randint(-60, 60))
    limit_in_force = amended_limit if req_date >= eff else base_limit
    amount = int(limit_in_force * r.choice([0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.6]))
    # definition twist: tenure counts from contract start, NOT from probation end
    # Keep tenure clear of the month boundary the rule tests, so "at least N
    # months" has one defensible reading whether counted in days or calendar months.
    start = req_date - dt.timedelta(days=30 * r.randint(1, 30))
    while abs((req_date - start).days - 30 * min_tenure) < 25:
        start = req_date - dt.timedelta(days=30 * r.randint(1, 30))
    probation_end = start + dt.timedelta(days=90)
    tenure_months = (req_date - start).days // 30
    tenure_from_probation = max(0, (req_date - probation_end).days // 30)
    notice = r.choice([notice_days - 3, notice_days - 1, notice_days, notice_days + 2, notice_days + 10])
    in_exception_class = r.random() < 0.35
    requester = person(r)

    def evaluate(amount, tenure_months, notice, is_prohibited, in_exc, limit):
        if is_prohibited:
            return False, "a prohibited category"
        cond_amount = amount <= limit
        cond_tenure = tenure_months >= min_tenure
        cond_notice = notice >= notice_days
        # the exception waives the notice condition for the exception class, nothing else
        if not cond_notice and in_exc:
            cond_notice = True
        ok = cond_amount and cond_tenure and cond_notice
        why = []
        if not cond_amount: why.append(f"{d['amount_name']} {_fmt(unit, amount)} exceeds the limit in force {_fmt(unit, limit)}")
        if not cond_tenure: why.append(f"tenure {tenure_months} months is under the {min_tenure}-month minimum (counted from contract start)")
        if not cond_notice: why.append(f"notice {notice} days is under the {notice_days}-day requirement")
        return ok, "; ".join(why) if why else "every condition holds"

    ok, why = evaluate(amount, tenure_months, notice, is_prohibited, in_exception_class, limit_in_force)

    # --- render -----------------------------------------------------------
    exc_plural, exc_single = d["exception_class"]
    sections = []
    sections.append(f"{d['policy_title']} — {company}\nVersion 4.2. This policy applies to all staff and contractors.")
    sections.append(
        "Section 1. Definitions\n"
        f"1.1 \"Tenure\" means the period elapsed since the individual's contract start date. For the avoidance of doubt, "
        f"tenure is not counted from the end of the probation period.\n"
        f"1.2 \"Notice\" means the number of calendar days between the date the request is lodged and the date the {dom_name} would take effect.\n"
        f"1.3 \"{exc_plural.capitalize()}\" are staff whose role is designated as such in the HR system.")
    base_rule = (
        "Section 3. Conditions for approval\n"
        f"3.1 {d['action'].capitalize()} is approved when all of the following hold: (a) the {d['amount_name']} does not exceed "
        f"{_fmt(unit, base_limit)}; (b) the requester's tenure is at least {min_tenure} months; (c) notice of at least {notice_days} days is given.\n"
        f"3.2 Requests in the following categories are not eligible under any circumstances: {', '.join(d['prohibited'])}.\n"
        f"3.3 Requests failing any condition in 3.1 are declined unless an exception in Section 5 applies.")
    exception = (
        "Section 5. Exceptions\n"
        f"5.1 For {exc_plural}, condition 3.1(c) (notice) is waived. No other condition is waived by this section.\n"
        f"5.2 Section 5.1 does not apply to categories listed in 3.2.")
    amendment = (
        f"Section 7. Amendment A-{r.randint(11, 39)}\n"
        f"7.1 With effect from {eff.strftime('%d %B %Y')}, the limit in condition 3.1(a) is {_fmt(unit, amended_limit)}. "
        f"Requests lodged before that date are assessed against the previous limit.")
    procedure = ("Section 4. Procedure\n"
                 "4.1 Requests are lodged through the portal and routed to the line manager. 4.2 The manager records the decision "
                 "and the clause relied upon. 4.3 Declined requests may be resubmitted once the failing condition is met.")
    order = [base_rule, procedure, exception, amendment] if not amendment_first else [amendment, base_rule, procedure, exception]
    sections.extend(order)
    case = (
        "Case file\n"
        f"Requester: {requester}, {r.choice(DEPARTMENTS)}" + (f", designated {exc_single}" if in_exception_class else "") + ".\n"
        f"Contract start date: {start.strftime('%d %B %Y')}. Probation ended: {probation_end.strftime('%d %B %Y')}.\n"
        f"Request lodged: {req_date.strftime('%d %B %Y')}. Category: {cat}. {d['amount_name'].capitalize()}: {_fmt(unit, amount)}. "
        f"Notice given: {notice} days.\n"
        f"Note from the requester: \"My manager said this should be fine since I finished probation {tenure_from_probation} months ago.\"")
    target = r.randint(2000, 5500) if long else r.randint(400, 1200)
    body = pad_to(r, sections, target, heading="Section")
    state = "\n\n".join(body + [case])

    q_variants = [
        "Applying the policy as written to the case file, may the request be approved?",
        "Is the request in the case file permitted under this policy? Conditions not shown to be met are treated as unmet.",
        "Does the policy, including its amendments and exceptions, allow this request?",
    ]
    instructions = r.choice(q_variants)
    criteria = {"true": "The request satisfies every applicable condition, or a stated exception covers the failing one, and no prohibition applies.",
                "false": "At least one applicable condition fails without an exception covering it, or a prohibition applies."}
    rationale = (f"Limit in force on {req_date.strftime('%d %b')} is {_fmt(unit, limit_in_force)} (amendment effective {eff.strftime('%d %b')}). "
                 f"Tenure from contract start = {tenure_months} months. {why}.")
    base = Scenario(id=f"synth-policy-{seed}-{index}", family="policy", kind="noul", state=state, instructions=instructions,
                    criteria=criteria, expected="true" if ok else "false", rationale=rationale,
                    n_tokens_est=est_tokens(state), tags=[dom_name, "long" if long else "short",
                                                          "amendment_applies" if req_date >= eff else "amendment_pending"])

    # --- twin: flip the decisive fact through the same evaluator ------------
    twin = None
    if not is_prohibited:
        # choose the smallest change that flips the outcome
        candidates = []
        for new_amount in (int(limit_in_force * 0.9), int(limit_in_force * 1.1)):
            ok2, why2 = evaluate(new_amount, tenure_months, notice, False, in_exception_class, limit_in_force)
            if ok2 != ok:
                candidates.append(("amount", new_amount, ok2, why2))
        for new_notice in (notice_days - 2, notice_days + 1):
            ok2, why2 = evaluate(amount, tenure_months, new_notice, False, in_exception_class, limit_in_force)
            if ok2 != ok:
                candidates.append(("notice", new_notice, ok2, why2))
        if candidates:
            what, val, ok2, why2 = r.choice(candidates)
            if what == "amount":
                new_case = case.replace(f"{d['amount_name'].capitalize()}: {_fmt(unit, amount)}", f"{d['amount_name'].capitalize()}: {_fmt(unit, val)}")
            else:
                new_case = case.replace(f"Notice given: {notice} days", f"Notice given: {val} days")
            assert new_case != case
            twin = Scenario(id=base.id + "-twin", family="policy", kind="noul", state="\n\n".join(body + [new_case]),
                            instructions=instructions, criteria=criteria, expected="true" if ok2 else "false",
                            rationale=f"Twin: {what} changed to {val}. {why2}.", twin_of=base.id, flipped=what,
                            n_tokens_est=base.n_tokens_est, tags=base.tags)
    return [base] + ([twin] if twin else [])
