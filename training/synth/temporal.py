"""temporal_numeric scenarios: cutoffs, time zones, business days, running totals, >= vs >."""

from __future__ import annotations

import datetime as dt
import random

from .common import CITIES, COMPANIES, Scenario, est_tokens, pad_to, person

TZ = {"Lisbon": 1, "Bergen": 2, "Tallinn": 3, "Geneva": 2, "Kraków": 2, "Cork": 1, "Riga": 3, "Porto": 1, "Turku": 3}


LEVEL = {"level": "hard"}


def _cancellation(r: random.Random):
    """Free cancellation if cancelled at least H hours before departure, in the departure city's local time."""
    dep_city, req_city = r.sample(list(TZ), 2)
    if LEVEL["level"] == "easy":
        req_city = dep_city  # same zone: no conversion
    H = r.choice([24, 48, 72])
    dep_local = dt.datetime(2026, r.randint(3, 11), r.randint(2, 27), r.choice([6, 9, 14, 18, 22]), r.choice([0, 15, 30, 45]))
    dep_utc = dep_local - dt.timedelta(hours=TZ[dep_city])
    # cancellation time expressed in the requester's local zone, within a few hours of the cutoff
    delta_h = H + (r.choice([-6, -3, 3, 6]) if LEVEL["level"] == "easy" else r.choice([-3, -1, -0.25, 0, 0.5, 2, 6]))
    cancel_utc = dep_utc - dt.timedelta(hours=delta_h)
    cancel_local = cancel_utc + dt.timedelta(hours=TZ[req_city])
    hours_before = (dep_utc - cancel_utc).total_seconds() / 3600
    ok = hours_before >= H
    sections = [
        f"Booking terms — {r.choice(COMPANIES)} travel desk",
        f"Clause 6. A booking may be cancelled free of charge if the cancellation is received at least {H} hours before the scheduled departure. "
        f"Times are compared in the local time of the departure city. A cancellation received exactly {H} hours before departure qualifies.",
        f"Clause 9. Time zones used by this desk (UTC offsets in effect): " + ", ".join(f"{c} UTC+{o}" for c, o in TZ.items()) + ".",
        "Clause 11. The previous edition of these terms used a 36-hour rule; it was withdrawn and does not apply to bookings made this year.",
        f"Booking B-{r.randint(10000, 99999)}: departure from {dep_city} on {dep_local.strftime('%d %B %Y at %H:%M')} ({dep_city} local time).",
        f"Cancellation request received on {cancel_local.strftime('%d %B %Y at %H:%M')} {req_city} local time, from {person(r)} in {req_city}.",
    ]
    q = "Does this cancellation qualify as free of charge under the booking terms?"
    crit = {"true": "The cancellation was received at or before the free-cancellation cutoff.", "false": "The cancellation was received after the cutoff."}
    rationale = (f"Departure {dep_utc.strftime('%d %b %H:%M')} UTC; cancellation {cancel_utc.strftime('%d %b %H:%M')} UTC; "
                 f"{hours_before:.2f} h before vs {H} h required → {'qualifies' if ok else 'does not qualify'}.")
    # twin: shift the cancellation by enough to cross the cutoff
    shift = dt.timedelta(hours=(H - hours_before) + (1 if not ok else -1) * 0.0) if False else None
    new_cancel_utc = dep_utc - dt.timedelta(hours=H + (2 if not ok else -2))
    new_local = new_cancel_utc + dt.timedelta(hours=TZ[req_city])
    twin_sections = sections[:-1] + [sections[-1].replace(cancel_local.strftime('%d %B %Y at %H:%M'), new_local.strftime('%d %B %Y at %H:%M'))]
    twin_ok = not ok
    return sections, q, crit, ok, rationale, twin_sections, twin_ok, "cancellation_time", "cancellation"


def _running_total(r: random.Random):
    """Monthly cap on approvals; the new request is checked against the running total, >= vs > matters."""
    cap = r.choice([5000, 8000, 12000, 20000])
    strict = (r.random() < 0.5) and LEVEL["level"] != "easy"  # easy: always "must not exceed"
    n = r.randint(4, 9)
    prior = [r.choice([250, 400, 750, 900, 1200, 1500, 2200]) for _ in range(n)]
    total = sum(prior)
    # place the new amount so total + new is near the cap; sometimes exactly at cap
    new = r.choice([cap - total - 600, cap - total + 600]) if LEVEL["level"] == "easy" else r.choice([cap - total, cap - total - 100, cap - total + 100, cap - total - 350])
    if new <= 0:
        new = r.choice([100, 250]); prior = prior[:-1]; total = sum(prior); new = max(50, min(new, cap - total))
    after = total + new
    ok = (after < cap) if strict else (after <= cap)
    rule_word = "must remain strictly below" if strict else "must not exceed"
    rows = "\n".join(f"- {(dt.date(2026, 9, 1) + dt.timedelta(days=i * 3)).strftime('%d %b')}: €{a:,} ({r.choice(['tooling', 'travel', 'contractor', 'licences', 'catering'])})" for i, a in enumerate(prior))
    sections = [
        f"{r.choice(COMPANIES)} — monthly discretionary spend control",
        f"Rule 3. The cumulative approved discretionary spend in a calendar month, including the request under consideration, {rule_word} €{cap:,}.",
        "Rule 4. Amounts are counted on the approval date, before tax. Rejected requests do not count.",
        "Approved this month so far:\n" + rows,
        f"New request: €{new:,} for {r.choice(['a replacement printer', 'venue hire', 'a training course', 'spare parts'])}, submitted today.",
    ]
    q = "May the new request be approved under Rule 3?"
    crit = {"true": "Approving it keeps the month within the control.", "false": "Approving it would breach the control."}
    rationale = f"Prior total €{total:,} + €{new:,} = €{after:,}; cap €{cap:,} ({'strict <' if strict else '<='}) → {'ok' if ok else 'breach'}."
    new2 = new + (-200 if not ok else 200)
    twin_sections = sections[:-1] + [sections[-1].replace(f"€{new:,}", f"€{new2:,}")]
    after2 = total + new2
    twin_ok = (after2 < cap) if strict else (after2 <= cap)
    return sections, q, crit, ok, rationale, twin_sections, twin_ok, "amount", "running_total"


def _business_days(r: random.Random):
    """A deadline in business days from a trigger date; weekends and a listed holiday excluded."""
    trigger = dt.date(2026, r.randint(1, 11), r.randint(1, 26))
    days = r.choice([3, 5, 10])
    holiday = trigger + dt.timedelta(days=r.randint(1, 6))
    while holiday.weekday() >= 5:
        holiday += dt.timedelta(days=1)
    if LEVEL["level"] == "easy":  # holiday listed but outside the window
        holiday = trigger + dt.timedelta(days=40)
    # compute deadline: count `days` business days after the trigger
    d = trigger; count = 0
    while count < days:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d != holiday:
            count += 1
    deadline = d
    submitted = deadline + dt.timedelta(days=r.choice([-2, -1, 0, 1, 2]))
    ok = submitted <= deadline
    sections = [
        f"{r.choice(COMPANIES)} — complaints procedure",
        f"Step 2. A response must be sent within {days} business days of the day the complaint is logged. The day of logging is day zero. "
        "Business days exclude Saturdays, Sundays and the public holidays listed in Appendix A.",
        f"Appendix A. Public holidays this year include {holiday.strftime('%A %d %B %Y')}.",
        f"Complaint C-{r.randint(100, 999)} was logged on {trigger.strftime('%A %d %B %Y')}. The response was sent on {submitted.strftime('%A %d %B %Y')}.",
    ]
    q = "Was the response sent within the Step 2 deadline?"
    crit = {"true": "Sent on or before the deadline.", "false": "Sent after the deadline."}
    rationale = f"{days} business days after {trigger} (skipping weekends and {holiday}) = {deadline}; sent {submitted} → {'on time' if ok else 'late'}."
    new_sub = deadline + dt.timedelta(days=(2 if ok else -1))
    twin_sections = sections[:-1] + [sections[-1].replace(submitted.strftime('%A %d %B %Y'), new_sub.strftime('%A %d %B %Y'))]
    return sections, q, crit, ok, rationale, twin_sections, new_sub <= deadline, "sent_date", "business_days"


def build(r: random.Random, index: int, seed: int, long: bool = False, level: str = "hard") -> list[Scenario]:
    LEVEL["level"] = level
    mech = r.choice([_cancellation, _running_total, _business_days])
    sections, q, crit, ok, rationale, twin_sections, twin_ok, flipped, tag = mech(r)
    target = r.randint(900, 2000) if long else r.randint(300, 800)
    body = pad_to(r, sections, target, heading="Clause")
    # keep the case (last section) last
    state = "\n\n".join(body)
    base = Scenario(id=(f"synth-temporal-{seed}-{index}" if level == "hard" else f"synth-temporal-{level}-{seed}-{index}"), family="temporal_numeric", kind="noul", state=state, instructions=q,
                    criteria=crit, expected="true" if ok else "false", rationale=rationale, n_tokens_est=est_tokens(state),
                    tags=[tag, "long" if long else "short", level])
    # twin shares the filler: replace the changed case section inside the padded body
    tbody = [twin_sections[-1] if s == sections[-1] else s for s in body]
    twin = Scenario(id=base.id + "-twin", family="temporal_numeric", kind="noul", state="\n\n".join(tbody), instructions=q,
                    criteria=crit, expected="true" if twin_ok else "false", rationale=f"Twin: {flipped} moved across the cutoff.",
                    twin_of=base.id, flipped=flipped, n_tokens_est=base.n_tokens_est, tags=base.tags)
    if twin_ok == ok:
        return [base]
    return [base, twin]
