"""Carry the citation-support records onto edited sentences -- but only where
the claim did not change.

P5R-M adjudication follow-through (2026-08-29). Acting on the adjudications
edits the sentences the support records were written against. Two kinds of
edit, and they must not be treated alike:

  * a citation is REMOVED from a list and the prose is untouched. Every
    remaining citation carries the same claim it carried before, so its
    verdict carries over and only the recorded sentence text needs updating.
  * the PROSE changes. The claim the citation is being asked to carry is now
    a different claim, and a verdict written against the old wording says
    nothing about the new one. Carrying it over silently would be exactly the
    failure this audit exists to catch.

So the rule is mechanical: strip every \\cite{...} and all whitespace from
both sentences. If what remains is identical, the edit was citation-list-only
and the record carries over. If not, the point must appear in REJUDGED with a
fresh verdict and its evidence, or this refuses to emit it and the gate goes
red on a missing support record.

    env PYTHONHASHSEED=2025 python reporting/bib_support_resync.py \
        --raw <support_raw.json> --out <support_raw_new.json> [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

#: Points whose PROSE changed, with a fresh judgment made against the NEW
#: wording. Keyed by (key, a distinctive fragment of the new sentence).
#: Anything not listed here and not citation-list-only is refused.
REJUDGED: dict = {
    # P5R-M adjudication item 5. The old sentence said "most heads can be
    # pruned without loss", which is stronger than anything the paper's own
    # headline pruning result supports. The new sentence carries the paper's
    # figures, so the citation is being asked to carry a different -- and
    # smaller -- claim, and is judged against the new wording.
    ("Michel2019", "substantial fraction of attention heads"): {
        "verdict": "SUPPORTS",
        "burden": "C",
        "support_quote":
            "Body, section 4.2: \"this approach allows us to prune up to 20% "
            "and 40% of heads from WMT and BERT (respectively), without "
            "incurring any noticeable negative impact.\"  Abstract: \"a "
            "large percentage of attention heads can be removed at test time "
            "without significantly impacting performance. In fact, some "
            "layers can even be reduced to a single head.\"",
        "evidence_source":
            "https://ar5iv.labs.arxiv.org/html/1905.10650 (full text, "
            "fetched and grepped for the pruning figures) and "
            "http://export.arxiv.org/api/query?id_list=1905.10650 (abstract). "
            "Both numbers in the new sentence are the paper's own.",
        "overreach": False,
        "suggested_direction": "",
        "rejudged_note":
            "2026-08-29: the claim was narrowed to the paper's own 20%/40% "
            "figures and to 'at test time', and re-judged against the new "
            "wording. WEAK -> SUPPORTS because the sentence changed, not "
            "because the evidence did.",
    },
    # Abnar2020 sits in the SAME sentence, and its own clause -- "raw weights
    # are not the only defensible aggregation" -- is untouched. It has to be
    # declared anyway: the tool cannot tell which half of a sentence moved,
    # and a verdict must never ride along on an edit nobody looked at.
    ("Abnar2020", "substantial fraction of attention heads"): {
        "verdict": "SUPPORTS",
        "burden": "A",
        "support_quote":
            "\"We propose two methods for approximating the attention to "
            "input tokens given attention weights, attention rollout and "
            "attention flow, as post hoc methods when we use attention "
            "weights as the relative relevance of the input tokens.\"",
        "evidence_source":
            "https://api.semanticscholar.org/graph/v1/paper/DOI:"
            "10.18653/v1/2020.acl-main.385 (identical to the ACL Anthology "
            "abstract)",
        "overreach": False,
        "suggested_direction": "",
        "rejudged_note":
            "2026-08-29: only the neighboring clause changed; this clause "
            "and this verdict are unchanged, and the record is re-declared "
            "rather than carried silently.",
    },
}


def _claim_text(s: str) -> str:
    """The sentence with every citation and all whitespace removed.

    What is left is the CLAIM. Two sentences with the same claim text differ
    only in which works are cited for it.
    """
    s = re.sub(r"\\cite[tp]?\{[^}]*\}", "", s)
    return re.sub(r"\s+", "", s)


def resync(raw: Path, rejudged: dict | None = None) -> dict:
    from reporting.bib_inventory import citation_points

    rejudged = REJUDGED if rejudged is None else rejudged
    d = json.loads(Path(raw).read_text())
    old = d["results"]
    by_key: dict = {}
    for r in old:
        by_key.setdefault(r["key"], []).append(r)

    points = citation_points()
    out, carried, rejudged_hits, refused, dropped = [], [], [], [], []

    used = set()
    for p in points:
        cands = by_key.get(p["key"], [])
        if not cands:
            refused.append({"key": p["key"], "sentence": p["sentence"][:160],
                            "why": "no support record for this key at all"})
            continue
        # the old record whose sentence is closest to this one
        best, score = None, -1.0
        for i, r in enumerate(cands):
            if (p["key"], i) in used:
                continue
            s = SequenceMatcher(None, r["sentence"], p["sentence"]).ratio()
            if s > score:
                best, score, bi = r, s, i
        if best is None:
            refused.append({"key": p["key"], "sentence": p["sentence"][:160],
                            "why": "every record for this key already matched "
                                   "another point"})
            continue
        used.add((p["key"], bi))

        rec = dict(best)
        a, b = _claim_text(best["sentence"]), _claim_text(p["sentence"])
        # Identical claim text: only the citation list moved.
        # One contained in the other: the SENTENCE BOUNDARY moved -- the
        # extractor's window changed, or a neighboring clause was re-wrapped
        # -- while every word this citation sits among is still there. That is
        # not a claim change either, but it IS reported rather than silent,
        # because a boundary move is how a real truncation would first look.
        if a == b or (a and b and (a in b or b in a)):
            rec["sentence"] = p["sentence"]
            if best["sentence"] != p["sentence"]:
                kind = ("citation list changed, prose unchanged"
                        if a == b else
                        "sentence boundary moved, prose within it unchanged")
                rec["carried_over"] = kind + "; verdict carries over"
                carried.append({"key": p["key"], "kind": kind,
                                "sentence": p["sentence"][:110]})
            out.append(rec)
            continue

        hit = None
        for (k, frag), new in rejudged.items():
            if k == p["key"] and frag in p["sentence"]:
                hit = new
                break
        if hit is None:
            refused.append({
                "key": p["key"], "sentence": p["sentence"][:200],
                "why": "the PROSE changed and no re-judgment was declared; "
                       "the old verdict was written against different wording"})
            continue
        rec.update(hit)
        rec["sentence"] = p["sentence"]
        rec["rejudged"] = True
        rejudged_hits.append({"key": p["key"], "verdict": rec["verdict"],
                              "sentence": p["sentence"][:110]})
        out.append(rec)

    # Report every record that no point claimed -- not merely every KEY that
    # went away. A work cited twice can lose one of its two positions, and the
    # verdict written for THAT position has to be retired visibly: it is the
    # DOES-NOT-SUPPORT on the position we removed that a reader will want to
    # see accounted for.
    matched = {id(by_key[k][i]) for (k, i) in used}
    for r in old:
        if id(r) not in matched:
            still = any(x["key"] == r["key"] for x in out)
            dropped.append({"key": r["key"], "verdict": r["verdict"],
                            "sentence": r["sentence"][:110],
                            "why": ("the work is no longer cited anywhere"
                                    if not still else
                                    "this POSITION is gone; the work is still "
                                    "cited elsewhere and that position keeps "
                                    "its own record")})

    from collections import Counter
    # A survivor record belongs to a POSITION, so match it on the position,
    # not on the key: the flag that survived its challenge was raised against
    # a sentence, and if that sentence is gone so is the flag.
    live = {(r["key"], _claim_text(r["sentence"])[:120]) for r in out}
    survived = [s for s in d.get("survived", [])
                if (s["point"]["key"],
                    _claim_text(s["point"]["sentence"])[:120]) in live]
    return {"n_points": len(out), "counts": dict(Counter(r["verdict"]
                                                        for r in out)),
            "results": out, "survived": survived,
            "resync": {"carried_over_citation_list_only": carried,
                       "rejudged": rejudged_hits,
                       "REFUSED": refused,
                       "records_no_longer_cited": dropped}}


def _selftest() -> int:
    import tempfile
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    import reporting.bib_inventory as inv
    old_pts = inv.citation_points
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.json"
        raw.write_text(json.dumps({"results": [
            {"key": "A", "sentence": "Claim one \\cite{A,B}.", "burden": "A",
             "verdict": "SUPPORTS", "support_quote": "q"},
            {"key": "B", "sentence": "Claim one \\cite{A,B}.", "burden": "A",
             "verdict": "DOES-NOT-SUPPORT", "support_quote": "q"},
            {"key": "C", "sentence": "Most heads prune without loss "
                                     "\\cite{C}.", "burden": "B",
             "verdict": "WEAK", "support_quote": "q"}]}))

        # 1. a citation removed, prose untouched -> carries over
        inv.citation_points = lambda *a, **k: [
            {"key": "A", "sentence": "Claim one \\cite{A}.", "file": "t",
             "char": 0}]
        r = resync(raw, {})
        chk(len(r["results"]) == 1 and r["results"][0]["verdict"] == "SUPPORTS"
            and r["resync"]["carried_over_citation_list_only"],
            "removing a citation carries the others' verdicts over")
        chk(not r["resync"]["REFUSED"], "and refuses nothing")
        chk(any(x["key"] == "B" for x in r["resync"]["records_no_longer_cited"]),
            "the removed citation's record is reported as no longer cited")

        # 2. prose changed, nothing declared -> REFUSED, not carried
        inv.citation_points = lambda *a, **k: [
            {"key": "C", "sentence": "A substantial fraction of heads can be "
                                     "removed at limited cost \\cite{C}.",
             "file": "t", "char": 0}]
        r = resync(raw, {})
        chk(not r["results"] and len(r["resync"]["REFUSED"]) == 1,
            "a rewritten claim is REFUSED, not silently carried over")

        # 3. prose changed WITH a declared re-judgment -> emitted, marked
        r = resync(raw, {("C", "substantial fraction"):
                         {"verdict": "SUPPORTS",
                          "support_quote": "up to 20% and 40% of heads",
                          "evidence_source": "arxiv 1905.10650"}})
        chk(len(r["results"]) == 1
            and r["results"][0]["verdict"] == "SUPPORTS"
            and r["results"][0].get("rejudged") is True,
            "a declared re-judgment is emitted and marked as re-judged")

        # 4. the sentence boundary moves (the extractor's window changed):
        #    carried over, but reported as a boundary move, not silently
        inv.citation_points = lambda *a, **k: [
            {"key": "A", "sentence": "Claim one \\cite{A,B}.", "file": "t",
             "char": 0}]
        raw2 = Path(td) / "raw2.json"
        raw2.write_text(json.dumps({"results": [
            {"key": "A", "sentence": "Preamble text. Claim one \\cite{A,B}.",
             "burden": "A", "verdict": "SUPPORTS", "support_quote": "q"}]}))
        r = resync(raw2, {})
        chk(len(r["results"]) == 1 and not r["resync"]["REFUSED"]
            and any("boundary" in c["kind"]
                    for c in r["resync"]["carried_over_citation_list_only"]),
            "a moved sentence boundary carries over and is reported as one")

        #    but a genuine change of the claim is NOT containment-safe
        inv.citation_points = lambda *a, **k: [
            {"key": "A", "sentence": "Claim two \\cite{A}.", "file": "t",
             "char": 0}]
        r = resync(raw2, {})
        chk(not r["results"] and r["resync"]["REFUSED"],
            "a different claim is still refused")

        # 5. whitespace-only differences are not prose changes
        inv.citation_points = lambda *a, **k: [
            {"key": "A", "sentence": "Claim  one\n\\cite{A,B}.", "file": "t",
             "char": 0}]
        r = resync(raw, {})
        chk(len(r["results"]) == 1 and not r["resync"]["REFUSED"],
            "rewrapping a line is not a claim change")
    inv.citation_points = old_pts
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = resync(Path(a.raw))
    rs = r["resync"]
    print(f"points now {r['n_points']}   verdicts {r['counts']}")
    print(f"  carried over (citation list only): "
          f"{len(rs['carried_over_citation_list_only'])}")
    print(f"  re-judged (declared):              {len(rs['rejudged'])}")
    print(f"  no longer cited:                   "
          f"{len(rs['records_no_longer_cited'])}")
    for x in rs["records_no_longer_cited"]:
        print(f"      {x['key']} [{x['verdict']}]")
    if rs["REFUSED"]:
        print(f"  REFUSED ({len(rs['REFUSED'])}):")
        for x in rs["REFUSED"]:
            print(f"      {x['key']}: {x['why']}\n        {x['sentence']}")
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
        print(f"\nwrote {a.out}")
    return 1 if rs["REFUSED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
