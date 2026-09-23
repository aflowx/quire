"""multi_hop scenarios: the answer needs a chain of lookups across the state.

World: requests -> requesters -> teams -> regions -> a rule table keyed by
region and amount band -> an approver role -> a person, with aliases in
footnotes and a distractor who shares one link. The evaluator follows the
chain; the prose scatters each table into its own section.
"""

from __future__ import annotations

import random

from .common import CITIES, COMPANIES, DEPARTMENTS, Scenario, est_tokens, pad_to, person

ROLES = ["Regional Director", "Finance Controller", "Head of Procurement", "Site Lead", "Compliance Officer",
         "Operations Manager", "Programme Director"]
BANDS = [(0, 5000), (5000, 20000), (20000, 100000), (100000, 10**9)]


def _band(amount: int) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= amount < hi:
            return i
    return len(BANDS) - 1


def build(r: random.Random, index: int, seed: int, long: bool = True, level: str = "hard") -> list[Scenario]:
    """level: easy = one band per region (no amount lookup), no alias, 3 options;
    medium = bands, no alias, 4 options; hard = bands, alias footnote, 5 options."""
    company = r.choice(COMPANIES)
    regions = r.sample(["North", "South", "East", "West", "Central", "Nordic", "Iberia", "Alpine"], 4)
    if level == "easy":
        n_people = r.randint(5, 7)
    elif level == "medium":
        n_people = r.randint(6, 9)
    else:
        n_people = r.randint(9, 16) if long else r.randint(6, 9)
    people = [person(r) for _ in range(n_people)]
    while len(set(people)) < n_people:
        people = [person(r) for _ in range(n_people)]
    teams = [f"{r.choice(DEPARTMENTS)} {r.choice(['A', 'B', 'C', 'North', 'Two'])}" for _ in range(5)]
    while len(set(teams)) < 5:
        teams = [f"{r.choice(DEPARTMENTS)} {r.choice(['A', 'B', 'C', 'North', 'Two'])}" for _ in range(5)]
    team_region = {t: r.choice(regions) for t in teams}
    person_team = {p: r.choice(teams) for p in people}
    # rule table: (region, band) -> role; role -> holder per region
    rule = {(reg, b): r.choice(ROLES) for reg in regions for b in range(len(BANDS))}
    if level == "easy":  # one role per region regardless of amount
        for region in regions:
            role0 = r.choice(ROLES)
            for b in range(len(BANDS)):
                rule[(region, b)] = role0
    holders = {(reg, role): r.choice(people) for reg in regions for role in ROLES}
    # an alias footnote: the holder of the decisive role is referred to by an alias in the table
    requester = r.choice(people)
    amount = r.choice([1200, 3800, 7500, 14000, 26000, 61000, 120000, 240000])
    reg = team_region[person_team[requester]]
    band = _band(amount)
    role = rule[(reg, band)]
    approver = holders[(reg, role)]
    alias = f"{approver.split()[0][0]}. {approver.split()[1]}" if level == "hard" else approver
    # distractor: same first name different person holding a role in another region
    others = [p for p in people if p != approver]
    distractor = r.choice(others)
    # option set: approver + 3-5 others incl. the distractor and the requester's own manager-looking name
    n_extra = {"easy": 1, "medium": 2}.get(level, 3)
    pool = [approver, distractor] + r.sample([p for p in others if p != distractor], min(n_extra, len(others) - 1))
    r.shuffle(pool)

    sections = [f"{company} — Approval Directory and Delegation Rules (internal)"]
    roster = "\n".join(f"- {p}: {person_team[p]}" for p in people)
    sections.append("Section 1. Staff roster (name: team)\n" + roster)
    tr = "\n".join(f"- {t} → {team_region[t]} region" for t in teams)
    sections.append("Section 2. Team to region mapping\n" + tr)
    rows = []
    if level == "easy":
        for region in regions:
            rows.append(f"| {region} | {rule[(region, 0)]} |")
        r.shuffle(rows)
        sections.append("Section 3. Delegation of authority — approver role by region\n| Region | Approving role |\n|---|---|\n" + "\n".join(rows))
    else:
        for (rg, b), ro in rule.items():
            lo, hi = BANDS[b]
            hi_s = f"under {hi:,}" if hi < 10**9 else "and above"
            rows.append(f"| {rg} | {lo:,} {hi_s} | {ro} |")
        r.shuffle(rows)
        sections.append("Section 3. Delegation of authority — approver role by region and amount (EUR)\n| Region | Amount band | Approving role |\n|---|---|---|\n" + "\n".join(rows))
    hrows = []
    for (rg, ro), p in holders.items():
        name = alias if (rg, ro) == (reg, role) else p
        hrows.append(f"- {rg}, {ro}: {name}")
    r.shuffle(hrows)
    sections.append("Section 4. Current role holders\n" + "\n".join(hrows) +
                    (f"\n\nFootnote 4a. Names may appear abbreviated; \"{alias}\" refers to {approver}." if level == "hard" else ""))
    sections.append("Section 5. Amount bands\n" + "Bands are inclusive of the lower bound and exclusive of the upper bound. "
                    "Amounts are in EUR before tax. Requests split across invoices are assessed on the total.")
    request = (f"Request R-{r.randint(1000, 9999)}\nRaised by: {requester}. Amount: EUR {amount:,} (before tax). "
               f"Purpose: {r.choice(['replacement tooling', 'venue hire', 'consultancy days', 'server hardware', 'vehicle lease'])}.\n"
               f"Note: \"{distractor} approved something similar for us last quarter.\"")
    target = r.randint(1800, 4500) if long else r.randint(500, 1200)
    body = pad_to(r, sections, target)
    state = "\n\n".join(body + [request])
    instructions = r.choice(["Who must approve this request under the delegation rules?",
                             "Following the directory and the delegation table, which person is the required approver for request above?",
                             "Identify the person whose approval the request requires."])
    ids = [f"p{i}" for i in range(len(pool))]
    criteria = {i: p for i, p in zip(ids, pool)}
    expected = ids[pool.index(approver)]
    rationale = (f"{requester} → {person_team[requester]} → {reg}; EUR {amount:,} is band {BANDS[band][0]:,}+ → {role}; "
                 f"{reg} {role} = {alias} = {approver}" + (" (footnote 4a)." if level == "hard" else "."))
    base = Scenario(id=(f"synth-multihop-{seed}-{index}" if level == "hard" else f"synth-multihop-{level}-{seed}-{index}"), family="multi_hop", kind="choice", state=state,
                    instructions=instructions, criteria=criteria, expected=expected, rationale=rationale,
                    n_tokens_est=est_tokens(state), tags=["long" if long else "short", f"options{len(pool)}", level])
    # twin: move the amount to a different band so the role (and usually the person) changes
    twin = None
    if level == "easy":
        return [base]
    for new_amount in r.sample([1200, 3800, 7500, 14000, 26000, 61000, 120000, 240000], 8):
        nb = _band(new_amount)
        if nb == band:
            continue
        nrole = rule[(reg, nb)]
        napp = holders[(reg, nrole)]
        if napp != approver and napp in pool:
            new_request = request.replace(f"EUR {amount:,}", f"EUR {new_amount:,}")
            twin = Scenario(id=base.id + "-twin", family="multi_hop", kind="choice", state="\n\n".join(body + [new_request]),
                            instructions=instructions, criteria=criteria, expected=ids[pool.index(napp)],
                            rationale=f"Twin: amount {new_amount:,} → band {nb} → {nrole} → {napp}.", twin_of=base.id,
                            flipped="amount_band", n_tokens_est=base.n_tokens_est, tags=base.tags)
            break
    return [base] + ([twin] if twin else [])
