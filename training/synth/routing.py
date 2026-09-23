"""routing scenarios: 4–9 overlapping handlers; the request name-drops the wrong one."""

from __future__ import annotations

import random

from .common import COMPANIES, Scenario, est_tokens, pad_to, person

HANDLERS = [
    ("billing", "Billing", "invoices, charges and payment methods for active subscriptions"),
    ("refunds", "Refunds & Credits", "returning money already taken: refunds, credit notes, chargebacks"),
    ("access", "Account Access", "sign-in, passwords, two-factor devices and locked accounts"),
    ("security", "Security Incidents", "suspected compromise, unknown sign-ins, leaked credentials"),
    ("shipping", "Shipping", "delivery status, addresses and carrier issues for orders in transit"),
    ("returns", "Returns", "sending goods back after delivery, return labels and eligibility"),
    ("api", "Developer Support", "API keys, rate limits, SDKs and webhook failures"),
    ("outage", "Service Status", "platform-wide incidents affecting many customers at once"),
    ("legal", "Legal & Privacy", "data-deletion requests, subpoenas, terms disputes"),
    ("sales", "Sales", "quotes, plan upgrades before purchase, enterprise pricing"),
    ("hr", "People Team", "employee questions about pay, leave and benefits"),
]

# (request template, correct handler, wrong handler that the request mentions by keyword)
REQUESTS = [
    ("I was charged twice last month and I want the second payment back. Your billing page didn't help.", "refunds", "billing"),
    ("Someone logged into my account from a country I've never been to and changed the email. I can't get in now.", "security", "access"),
    ("My parcel says delivered but it isn't here. I'd like to return it if it ever arrives — can you sort this?", "shipping", "returns"),
    ("The item arrived but it's the wrong size. Shipping was fast though. How do I send it back?", "returns", "shipping"),
    ("Our webhooks stopped firing at 09:00 and the status page says all systems normal. API keys are fine.", "api", "outage"),
    ("Everything is down for us and for two other companies I spoke to — the API returns 503 everywhere.", "outage", "api"),
    ("Please delete all my data under GDPR. I've already cancelled billing so there's nothing to refund.", "legal", "refunds"),
    ("I want to upgrade to the Enterprise plan before we buy; the billing page only shows my current invoice.", "sales", "billing"),
    ("I forgot my password and the reset email never arrives. Nothing suspicious, I just can't sign in.", "access", "security"),
    ("An invoice shows a charge for a seat we removed; we don't want money back, just the next invoice corrected.", "billing", "refunds"),
]


def build(r: random.Random, index: int, seed: int, long: bool = False) -> list[Scenario]:
    company = r.choice(COMPANIES)
    text, correct, wrong = r.choice(REQUESTS)
    n_opts = r.choice([4, 5, 6, 7, 9])
    others = [h for h in HANDLERS if h[0] not in (correct, wrong)]
    chosen = [h for h in HANDLERS if h[0] in (correct, wrong)] + r.sample(others, n_opts - 2)
    r.shuffle(chosen)
    roster = "\n".join(f"- {name}: {desc}." for _, name, desc in chosen)
    rule = ("Routing rule. Route by what the customer needs done, not by the words they use. Money already taken is Refunds & Credits; "
            "a future or current charge is Billing. A sign-in problem with any sign of compromise is Security, otherwise Account Access. "
            "Goods still in transit are Shipping; goods already delivered are Returns. One customer's integration failing is Developer "
            "Support; many customers failing at once is Service Status.")
    sections = [f"{company} support — team roster and routing rule", "Teams\n" + roster, rule,
                f"Incoming request from {person(r)}:\n\"{text}\""]
    target = r.randint(700, 1400) if long else r.randint(250, 600)
    body = pad_to(r, sections, target, heading="Notice")
    # keep the request last
    body = [s for s in body if not s.startswith("Incoming request")] + [sections[-1]]
    state = "\n\n".join(body)
    criteria = {hid: f"{name} — {desc}" for hid, name, desc in chosen}
    q = r.choice(["Which team should this request be routed to?", "Under the routing rule, which team handles this request?",
                  "Choose the team that should own this ticket."])
    base = Scenario(id=f"synth-routing-{seed}-{index}", family="routing", kind="choice", state=state, instructions=q,
                    criteria=criteria, expected=correct, rationale=f"Need is '{correct}' although the text mentions '{wrong}'.",
                    n_tokens_est=est_tokens(state), tags=[f"options{n_opts}", "long" if long else "short"])
    return [base]
