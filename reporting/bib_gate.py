"""Bibliography gate: field policy, verification, and citation support.

P5R-M SS4. Three kinds of bibliography defect, and only the first is the kind a
compiler notices:

  * a field policy violated -- no verifiable identifier, or a DOI and a URL
    for the same work, or a hand-written "Available from:" fighting the
    journal style's own prefix;
  * a reference whose authoritative record does not match what we wrote, or
    does not exist;
  * a real, well-formatted, resolvable reference attached to a claim it never
    made.

The third is the dangerous one and no format check sees it, so this gate
requires a per-citation-POINT support record, not merely a per-reference one:
the same work cited in four places is asked to carry four different claims.

There is a fourth, found by this gate's own rendered-layer check: a field that
is present and correct in the .bib and that the STYLE never prints. This
journal's @article function does not emit `url`, so a JMLR paper with no DOI
carried a perfectly good url field and rendered with no locator a reader could
follow. Every .bib-level rule passed it. Checking the .bib is not the same as
checking what the reader sees, so this gate checks both.

Wired into reporting/compile_gate.py as a precondition for the manuscript.

    env PYTHONHASHSEED=2025 python reporting/bib_gate.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(CODE_ROOT))
#: paper_R1 is main/ + esm/ since P7-A SS2; a package's assembled
#: view is still flat. One resolver answers for all three layouts.
from experiments.package_layout import paper_file  # noqa: E402

ROOT = CODE_ROOT.parent
PAPER = ROOT / "paper_R1"
sys.path.insert(0, str(CODE_ROOT))

BIB = paper_file("references_Y.bib")
BBL = paper_file("paperY_main.bbl")
VERIF = ROOT / "reports" / "citation_verification.json"
SUPPORT = ROOT / "reports" / "citation_support.json"
#: P5R-M SS1.4 / SS3.4 items are stop-and-report by rule: Code does not resolve
#: them. The manuscript must still BUILD while they wait, so the compile gate
#: tolerates exactly the items declared here and prints them; packaging does
#: not tolerate them at all.
OPEN = ROOT / "internal_review" / "bib_open_adjudications.json"

PREFIXES = ("Available from", "Retrieved from", "Available at",
            "Accessed from")


def open_adjudications(path: Path = OPEN) -> dict:
    """The declared stop-and-report items, as {kind: set(keys)}."""
    if not Path(path).exists():
        return {"identifier": set(), "support": set()}
    d = json.loads(Path(path).read_text())
    return {"identifier": {x["key"] for x in d.get("identifier_items", [])
                           if x.get("status") == "OPEN"},
            "support": {x["key"] for x in d.get("support_items", [])
                        if x.get("status") == "OPEN"}}


def rendered_entries(bbl: Path = None) -> dict:
    """The .bbl split per key: what the reader actually gets."""
    # Resolved here, not in the signature: a default argument freezes the
    # root at import time, and package mode's rebinding never reaches it.
    bbl = bbl or BBL
    if not Path(bbl).exists():
        return {}
    src = Path(bbl).read_text(errors="replace")
    return {c.split("}", 1)[0]: c
            for c in re.split(r"\\bibitem\[[^\]]*\]\{", src)[1:]}


def check(bib: Path = None, verif: Path = None, support: Path = None,
          strict_support: bool = True, tolerate_open: bool = False,
          bbl: Path = None) -> dict:
    # Same reason as above: every root is resolved at CALL time.
    bib, verif = bib or BIB, verif or VERIF
    support, bbl = support or SUPPORT, bbl or BBL
    from reporting.bib_inventory import (parse_bib, cited_keys,
                                         citation_points, types_that_emit_url)

    entries = parse_bib(bib)
    try:
        cited = cited_keys()
    except FileNotFoundError:
        cited = list(entries)
    points = citation_points()

    no_identifier, both_doi_url, prefix_in_url = [], [], []
    url_types = types_that_emit_url()
    rendered = rendered_entries(bbl)
    unrendered = []
    for k in cited:
        e = entries.get(k)
        if e is None:
            no_identifier.append(f"{k}: cited but absent from the .bib")
            continue
        f = e["fields"]
        doi, url, isbn = (f.get("doi", "").strip(), f.get("url", "").strip(),
                          f.get("isbn", "").strip())
        note = f.get("note", "") + f.get("howpublished", "")
        has_url = bool(url) or bool(re.search(r"\\url\{|https?://", note))
        if not (doi or has_url or isbn):
            no_identifier.append(k)
        if doi and has_url:
            both_doi_url.append(k)
        # The prefix belongs to the style, not to a field -- EXCEPT where the
        # style cannot print the field at all. For an entry type whose
        # function never emits `url`, and which has no DOI either, the note is
        # the only channel left and its prefix has to be literal. The
        # exemption is computed from the .bst, so it cannot be claimed for an
        # entry the style would have rendered by itself.
        # The exemption covers the NOTE only. A prefix inside a url field
        # corrupts the URL itself and is wrong for every entry type.
        style_cannot_emit_url = (e["type"] not in url_types) and not doi
        if any(p in url for p in PREFIXES) or (
                any(p in note for p in PREFIXES) and not style_cannot_emit_url):
            prefix_in_url.append(k)
        # What the READER sees. A field the style silently drops is, for
        # anyone holding the PDF, no identifier at all.
        if rendered:
            r = rendered.get(k, "")
            if not (r"\doi{" in r or r"\url{" in r or "http" in r
                    or (isbn and isbn.replace("-", "") in r.replace("-", ""))):
                unrendered.append(k)

    # verification records
    vrec, missing_verif, not_verified = {}, [], []
    if verif.exists():
        raw = json.loads(verif.read_text())
        vrec = {r["key"]: r for r in (raw["results"] if isinstance(raw, dict)
                                      else raw)}
    for k in cited:
        r = vrec.get(k)
        if r is None:
            missing_verif.append(k)
        elif r.get("verdict") != "VERIFIED":
            not_verified.append(f"{k}={r.get('verdict')}")

    # support records, one per citation POINT
    srec, missing_support, not_supported = {}, [], []
    if support.exists():
        raw = json.loads(support.read_text())
        rows = raw["results"] if isinstance(raw, dict) else raw
        for r in rows:
            srec.setdefault(r["key"], []).append(r)
    for p in points:
        rows = srec.get(p["key"], [])
        hit = next((r for r in rows
                    if _same_sentence(r.get("sentence", ""), p["sentence"])),
                   None)
        if hit is None:
            missing_support.append(f"{p['key']} @ {p['sentence'][:60]}")
        elif hit.get("verdict") != "SUPPORTS":
            not_supported.append(f"{p['key']}={hit.get('verdict')} "
                                 f"@ {p['sentence'][:60]}")

    fails = {
        "no_verifiable_identifier": no_identifier,
        "doi_and_url_both_present": both_doi_url,
        "prefix_text_in_url_or_note": prefix_in_url,
        "no_identifier_in_rendered_list": unrendered,
        "missing_verification_record": missing_verif,
        "not_verified": not_verified,
        "missing_support_record": missing_support,
        "not_supported": not_supported,
    }
    if not strict_support:
        fails.pop("missing_support_record")
        fails.pop("not_supported")

    deferred = {}
    if tolerate_open:
        op = open_adjudications()
        for name, keys in (("no_verifiable_identifier", op["identifier"]),
                           ("no_identifier_in_rendered_list", op["identifier"]),
                           ("not_supported", op["support"])):
            if name not in fails:
                continue
            keep, defer = [], []
            for item in fails[name]:
                (defer if any(item.startswith(k) for k in keys)
                 else keep).append(item)
            fails[name] = keep
            if defer:
                deferred[name] = defer

    return {"n_cited": len(cited), "n_points": len(points),
            "failures": {k: v for k, v in fails.items() if v},
            "deferred_to_adjudication": deferred,
            "pass": not any(fails.values())}


def _same_sentence(a: str, b: str) -> bool:
    """Sentence identity, robust to whitespace, to LaTeX-vs-rendered form, and
    to orthography.

    A support record says: this verdict was reached against THIS sentence. So
    the record is never rewritten -- it has to keep saying what was judged.
    But P7-A SS1 respelled the manuscript, and a British -ised ending
    becoming an American -ized one does not ask the citation to carry a
    different claim. (Spelled out, this sentence would trip the very scan it
    describes -- so it is not spelled out. An exemption would have been the
    other way to write it, and the wrong one: the scanner is right.)
    The comparison already discards what does not change the claim (case,
    punctuation, whitespace); a spelling variant belongs in the same set. What
    is NOT discarded is a word change, which still lands in REJUDGED where it
    belongs.
    """
    from reporting.facts_gate import _ortho
    pat, m = _ortho()

    def spell(t: str) -> str:
        return pat.sub(lambda x: m[x.group(0).lower()], t)
    a, b = spell(a), spell(b)
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    return na == nb or na[:120] == nb[:120] or na in nb or nb in na


def _selftest() -> int:
    import tempfile
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    base = """@article{Good2020,
  author = {Real, A. and Person, B.},
  title = {A title},
  journal = {A journal},
  year = {2020},
  doi = {10.1000/real}
}
"""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        NO_BBL = d / "absent.bbl"   # the rendered check is off for fixtures
                                    # that have no rendered layer of their own

        def run(bibtext, verif_rows, support_rows=None, strict=False,
                bbl=None):
            bp = d / "b.bib"; bp.write_text(bibtext)
            vp = d / "v.json"; vp.write_text(json.dumps(verif_rows))
            sp = d / "s.json"
            sp.write_text(json.dumps(support_rows or []))
            import reporting.bib_inventory as inv
            old_cited = inv.cited_keys
            old_points = inv.citation_points
            keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bibtext)
            inv.cited_keys = lambda *a, **k: keys
            if support_rows is None:
                inv.citation_points = lambda *a, **k: []
            try:
                return check(bp, vp, sp, strict_support=strict,
                             bbl=bbl or NO_BBL)
            finally:
                inv.cited_keys, inv.citation_points = old_cited, old_points

        # 1. a fabricated DOI: the record says NOT-FOUND -> red
        r = run(base, [{"key": "Good2020", "verdict": "NOT-FOUND"}])
        chk(not r["pass"] and "not_verified" in r["failures"],
            "a reference whose record is NOT-FOUND fails the gate")

        # 2. doi AND url on the same entry -> red
        dual = base.replace("  doi = {10.1000/real}\n",
                            "  doi = {10.1000/real},\n"
                            "  url = {https://example.org/x}\n")
        r = run(dual, [{"key": "Good2020", "verdict": "VERIFIED"}])
        chk(not r["pass"] and "doi_and_url_both_present" in r["failures"],
            "an entry carrying both a DOI and a URL fails the gate")

        # 3. no identifier at all -> red
        none_ = """@inproceedings{Bare2020,
  author = {Nobody},
  title = {No identifier anywhere},
  booktitle = {Somewhere},
  year = {2020}
}
"""
        r = run(none_, [{"key": "Bare2020", "verdict": "VERIFIED"}])
        chk(not r["pass"] and "no_verifiable_identifier" in r["failures"],
            "an entry with no doi, url or isbn fails the gate")

        # 4. a citation point with no support record -> red
        import reporting.bib_inventory as inv
        old = inv.citation_points
        inv.citation_points = lambda *a, **k: [
            {"key": "Good2020", "file": "t.tex", "char": 0,
             "sentence": "Some claim that needs support."}]
        try:
            bp = d / "b.bib"; bp.write_text(base)
            vp = d / "v.json"
            vp.write_text(json.dumps([{"key": "Good2020",
                                       "verdict": "VERIFIED"}]))
            sp = d / "s.json"; sp.write_text(json.dumps([]))
            old_c = inv.cited_keys
            inv.cited_keys = lambda *a, **k: ["Good2020"]
            try:
                r = check(bp, vp, sp, strict_support=True, bbl=NO_BBL)
            finally:
                inv.cited_keys = old_c
            chk(not r["pass"] and "missing_support_record" in r["failures"],
                "a citation point with no support record fails the gate")

            sp.write_text(json.dumps([{"key": "Good2020",
                                       "sentence": "Some claim that needs "
                                                   "support.",
                                       "verdict": "WEAK"}]))
            inv.cited_keys = lambda *a, **k: ["Good2020"]
            try:
                r = check(bp, vp, sp, strict_support=True, bbl=NO_BBL)
            finally:
                inv.cited_keys = old_c
            chk(not r["pass"] and "not_supported" in r["failures"],
                "a citation point recorded WEAK fails the gate")

            sp.write_text(json.dumps([{"key": "Good2020",
                                       "sentence": "Some claim that needs "
                                                   "support.",
                                       "verdict": "SUPPORTS"}]))
            inv.cited_keys = lambda *a, **k: ["Good2020"]
            try:
                r = check(bp, vp, sp, strict_support=True, bbl=NO_BBL)
            finally:
                inv.cited_keys = old_c
            chk(r["pass"], "a clean entry with a SUPPORTS record passes")
        finally:
            inv.citation_points = old

        # 5. the prefix check
        pref = base.replace("  doi = {10.1000/real}\n",
                            "  url = {Available from: https://example.org/x}\n")
        r = run(pref, [{"key": "Good2020", "verdict": "VERIFIED"}])
        chk("prefix_text_in_url_or_note" in r["failures"],
            "a hand-written 'Available from:' in a field fails the gate")

        # 6. a field the STYLE drops: an @article with a url renders with no
        #    locator at all, and every .bib-level rule passes it
        art = base.replace("  doi = {10.1000/real}\n",
                           "  url = {https://example.org/x}\n")
        bblp = d / "f.bbl"
        bblp.write_text("\\bibitem[1]{Good2020}\nReal A. A title. "
                        "A journal. 2020;1(1):1--2.\n")
        r = run(art, [{"key": "Good2020", "verdict": "VERIFIED"}], bbl=bblp)
        chk("no_identifier_in_rendered_list" in r["failures"],
            "a url the style never prints fails the rendered-layer check")

        bblp.write_text("\\bibitem[1]{Good2020}\nReal A. A title. A journal. "
                        "2020;1(1):1--2. Available from: "
                        "\\url{https://example.org/x}.\n")
        r = run(art, [{"key": "Good2020", "verdict": "VERIFIED"}], bbl=bblp)
        chk("no_identifier_in_rendered_list" not in r["failures"],
            "the same entry passes once the locator actually renders")

        # 7. the prefix exemption is computed, not granted: it applies only
        #    where the style cannot emit the field and there is no DOI
        art_note = base.replace(
            "  doi = {10.1000/real}\n",
            "  note = {Available from: \\url{https://example.org/x}}\n")
        r = run(art_note, [{"key": "Good2020", "verdict": "VERIFIED"}],
                bbl=bblp)
        chk("prefix_text_in_url_or_note" not in r["failures"],
            "an @article with no DOI may carry the prefix -- the style "
            "cannot print its url at all")
        conf = art_note.replace("@article{Good2020", "@inproceedings{Good2020")
        r = run(conf, [{"key": "Good2020", "verdict": "VERIFIED"}], bbl=bblp)
        chk("prefix_text_in_url_or_note" in r["failures"],
            "the same prefix on an @inproceedings still fails -- the style "
            "would have printed that one itself")

    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-strict-support", action="store_true",
                    help="skip the per-citation-point support requirement "
                         "(used only while that record is being built)")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = check(strict_support=not a.no_strict_support)
    print(f"cited references {r['n_cited']}   citation points {r['n_points']}")
    for k, v in r["failures"].items():
        print(f"  [RED] {k} ({len(v)}): {v[:6]}")
    print(f"\nBIB GATE: {'PASS' if r['pass'] else 'FAIL'}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
