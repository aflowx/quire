from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

FIRST = ["Ada", "Ben", "Chloé", "Dev", "Elin", "Farid", "Greta", "Hugo", "Ines", "Jonas", "Kaia", "Luis",
         "Mira", "Noor", "Oskar", "Priya", "Quinn", "Rosa", "Sven", "Tomás", "Uma", "Viktor", "Wen", "Yara", "Zoë",
         "Amara", "Bao", "Cyrus", "Dalia", "Emeka", "Freya", "Gael", "Hana", "Ivo", "Jun", "Kofi", "Lena", "Mateo",
         "Nadia", "Olu", "Petra", "Rafi", "Sana", "Teo", "Ulla", "Vera", "Wilhelm", "Ximena", "Yusuf", "Zara"]
LAST = ["Lin", "Reed", "Shah", "Vega", "Okafor", "Dubois", "Park", "Ito", "Novak", "Haddad", "Moreau", "Sato",
        "Berg", "Costa", "Nakamura", "Olsen", "Petrov", "Quist", "Rahman", "Silva", "Tanaka", "Umar", "Varga",
        "Weber", "Xu", "Yilmaz", "Zhou", "Adeyemi", "Brandt", "Chen", "Dlamini", "Eriksen", "Ferreira", "Gupta"]
CITIES = ["Bergen", "Lisbon", "Tallinn", "Porto", "Geneva", "Kraków", "Valencia", "Leipzig", "Ghent", "Turku",
          "Ljubljana", "Nantes", "Bilbao", "Aarhus", "Bratislava", "Cork", "Graz", "Utrecht", "Malmö", "Riga"]
COMPANIES = ["Northwind Outfitters", "Halvard Logistics", "Meridian Health Partners", "Kestrel Data",
             "Orbis Mutual", "Tallow & Finch", "Sunward Energy", "Bluecap Robotics", "Ferrous Works",
             "Larkspur Foods", "Quillon Software", "Aster Clinics", "Pinewood Council", "Velour Hotels"]
DEPARTMENTS = ["Finance", "Operations", "Engineering", "Customer Care", "Legal", "People", "Facilities",
               "Security", "Procurement", "Quality", "Logistics", "Clinical Services", "Marketing"]


@dataclass
class Scenario:
    id: str
    family: str
    kind: str                      # noul | choice | score
    state: str
    instructions: str
    criteria: dict | list          # noul: {"true","false"}; choice: {id: desc}; score: [levels]
    expected: str | int            # noul: "true"/"false"; choice: id; score: level index
    rationale: str
    gold_probs: dict | None = None  # probability family only
    twin_of: str | None = None
    flipped: str | None = None     # what the twin changed
    n_tokens_est: int = 0
    tags: list = field(default_factory=list)

    def row(self) -> dict:
        """kev's decision-v7 row shape, which scenarios/readout/corpus.py loads."""
        q = {"type": self.kind, "instructions": self.instructions, "criteria": self.criteria,
             "label": (self.expected == "true") if self.kind == "noul" else self.expected, "src": self.family}
        if self.gold_probs:
            q["gold_probs"] = self.gold_probs
        return {"state": self.state, "questions": {"q": q},
                "_meta": {"id": self.id, "family": self.family, "twin_of": self.twin_of, "flipped": self.flipped,
                          "rationale": self.rationale, "tags": self.tags, "source": "quire-synth-v1"}}


def rng_for(seed: int, family: str, index: int) -> random.Random:
    h = hashlib.sha256(f"{seed}:{family}:{index}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def person(r: random.Random) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def est_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35)


# --- filler: realistic, irrelevant, varied ---------------------------------

_FILLER_TOPICS = {
    "facilities": ["The {dept} floor will be re-carpeted between {d1} and {d2}; desks in bays {n1}–{n2} are to be cleared by the preceding Friday.",
                   "Badge readers on the {city} site will be replaced with contactless units; existing badges continue to work until {d2}.",
                   "The {dept} kitchenette has a new dishwasher. Please rinse mugs before loading; the previous unit failed due to residue build-up.",
                   "Parking permits for the {city} office are renewed annually in {month}; applications received after {d1} are processed in the following cycle."],
    "it": ["Laptops older than {n1} years are eligible for replacement in the {month} refresh; requests go through the {dept} portal.",
           "The VPN client will be updated to version {n1}.{n2} on {d1}. A restart is required; unsaved work should be closed beforehand.",
           "Shared mailboxes are archived after {n1} days of inactivity. Owners are notified twice before archival.",
           "Multi-factor prompts now expire after {n2} minutes rather than {n1}. This does not change the sign-in policy itself."],
    "hr": ["Learning credits for {year} are {n1} hours per person; unused credits do not carry over except where a course spans the year end.",
           "The {dept} team offsite is provisionally {d1}–{d2} in {city}; dietary requirements should be entered by {month} 1.",
           "Performance conversations happen twice a year. The {month} cycle uses the same form as last year with one added question on collaboration.",
           "Employees relocating between {city} and {city2} should update their address within {n2} days for tax reporting."],
    "ops": ["Courier collections from the {city} depot move to {n1}:00 on weekdays; Saturday collections are unchanged.",
            "Vendor {company} has changed its remittance address; invoices dated after {d1} carry the new details.",
            "The {dept} weekly review is moving from Tuesday to Wednesday to avoid clashing with the {city} stand-up.",
            "Stock counts in {month} found {n2} discrepancies, all under {n1} units; no process change is proposed."],
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def filler_paragraphs(r: random.Random, n: int) -> list[str]:
    """`n` unrelated paragraphs of two to four sentences each, from mixed topics."""
    out = []
    for _ in range(n):
        topic = r.choice(list(_FILLER_TOPICS))
        sents = r.sample(_FILLER_TOPICS[topic], k=min(len(_FILLER_TOPICS[topic]), r.randint(2, 4)))
        d1, d2 = sorted(r.sample(range(1, 28), 2))
        month = r.choice(MONTHS)
        text = " ".join(sents).format(
            dept=r.choice(DEPARTMENTS), city=r.choice(CITIES), city2=r.choice(CITIES), d1=f"{d1} {month}",
            d2=f"{d2} {month}", n1=r.randint(2, 9), n2=r.randint(10, 60), month=month, year=r.choice([2025, 2026]),
            company=r.choice(COMPANIES))
        out.append(text)
    return out


def pad_to(r: random.Random, sections: list[str], target_tokens: int, heading: str = "Section") -> list[str]:
    """Interleave filler sections until the estimated length reaches the target."""
    out = list(sections)
    k = len(out) + 10  # numbered after the real sections so no heading number repeats
    while est_tokens("\n\n".join(out)) < target_tokens:
        para = filler_paragraphs(r, r.randint(1, 3))
        out.insert(r.randint(1, max(1, len(out) - 1)), f"{heading} {k}. General notices\n" + "\n".join(para))
        k += 1
    return out


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
