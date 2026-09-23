"""Verify a partially rebuilt Decision Index suite row-by-row.

The frozen suite isn't published (apolinario/decision-index#1), and a partial
rebuild cannot match the whole-file sha256. Every result row of reflex-27b's
full run (Kshetrajna/decision-index-results) carries the run_id and the
payload_sha256 of the frozen request, so each rebuilt request can be checked
against the frozen one individually.

    python scripts/verify_di_rows.py ../decision-index/work/.../selected-rows.jsonl.gz
"""

from __future__ import annotations

import collections
import gzip
import json
import sys

from huggingface_hub import hf_hub_download

REF = ("Kshetrajna/decision-index-results", "runs/reflex-27b-p2-full-r8-p32/results.jsonl.gz")


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    ref = {}
    for r in rows(hf_hub_download(REF[0], REF[1], repo_type="dataset")):
        ref[r["run_id"]] = (r["payload_sha256"], r["catalog_id"], r["dataset"])
    per = collections.defaultdict(lambda: [0, 0, 0])  # match, mismatch, missing-from-ref
    ours = set()
    for r in rows(sys.argv[1]):
        ev = r["_evaluation"]; rid = ev["run_id"]; ours.add(rid)
        b = per[(ev["catalog_id"], ev.get("dataset"))]
        if rid not in ref:
            b[2] += 1
        elif ref[rid][0] == ev["payload_sha256"]:
            b[0] += 1
        else:
            b[1] += 1
    expected = collections.Counter((c, d) for rid, (_, c, d) in ref.items())
    print(f"{'#':>3s} {'benchmark':22s} {'frozen':>7s} {'rebuilt':>7s} {'match':>7s} {'differ':>6s} {'extra':>5s}")
    ok = True
    for (cid, ds), n in sorted(expected.items()):
        if (cid, ds) not in per:
            continue
        m, x, e = per[(cid, ds)]
        flag = "" if (m == n and x == 0 and e == 0) else "  <-- "
        ok &= flag == ""
        print(f"{cid:3d} {str(ds)[:22]:22s} {n:7d} {m + x + e:7d} {m:7d} {x:6d} {e:5d}{flag}")
    print("\nALL REBUILT BENCHMARKS BYTE-IDENTICAL PER REQUEST" if ok else "\nMISMATCHES PRESENT")


if __name__ == "__main__":
    main()
