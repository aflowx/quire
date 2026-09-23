"""Does the per-option fan-out lose accuracy where the letter path also applies?

dbpedia14 (14 options): letter path vs forced wide path, paired.
banking77 (77 options): wide path only -- the letter path cannot name 77 options.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quire.engine import Engine  # noqa: E402
from quire.schema import Question  # noqa: E402
from quire.stats import paired_permutation_test  # noqa: E402
import kev_corpus as corpus  # noqa: E402

MODEL = "mlx-community/Qwen3.5-4B-MLX-8bit"


def run(engine, items):
    hits, secs = [], time.perf_counter()
    for it in items:
        q = Question(instructions=it.instructions, criteria={o: (d or o) for o, d in zip(it.option_ids, it.descriptions)}, kind="choice")
        a = engine.decide(it.state, [q])[0]
        hits.append(float(a.choice == it.option_ids[it.label]))
    return hits, (time.perf_counter() - secs) / len(items)


def main() -> None:
    dev = corpus.load("decision-v7", "development")
    dbp = [i for i in dev if i.source == "dbpedia14"]
    bank = [i for i in dev if i.source == "banking77"]
    letter = Engine(model_repo=MODEL, n_permutations=2, use_debias=False, prompt_style="quire", wide_cap=26)
    a, ta = run(letter, dbp)
    letter.wide_cap = 2  # force every choice through the wide path
    b, tb = run(letter, dbp)
    t = paired_permutation_test(b, a)
    print(f"dbpedia14 (14 options, n={len(dbp)}): letter {sum(a)/len(a):.3f} ({ta*1000:.0f} ms)  wide {sum(b)/len(b):.3f} ({tb*1000:.0f} ms)  "
          f"diff {(sum(b)-sum(a))/len(a):+.3f} p={t['p_value']:.3f}  wins/losses {sum(x>y for x,y in zip(b,a))}/{sum(x<y for x,y in zip(b,a))}")
    letter.wide_cap = 26
    c, tc = run(letter, bank)
    print(f"banking77 (77 options, n={len(bank)}): wide {sum(c)/len(c):.3f} ({tc*1000:.0f} ms/item, chance 0.013)")


if __name__ == "__main__":
    main()
