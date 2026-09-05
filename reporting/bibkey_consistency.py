"""Does each bib key still name the work its fields describe?

P5R-M adjudication item 2 (2026-08-29). One entry in this bibliography was
keyed `Kim2024ehratt` while its fields describe a paper by Kowsar, Rabbani and
Samad -- no author named Kim. Nothing in a field check sees that: every field
was correct, the DOI resolved, the record matched. The key is not a field, so
no field rule looks at it.

But the key is the one part of an entry that records what the entry was
MINTED for. When a key and its fields disagree, the entry was almost certainly
swapped after the key was written, and the citation in the text -- placed for
the original work -- may now point at a different paper. That is the failure
this bibliography actually had, and the mismatch is its only mechanical
signature. One occurrence means you have to assume there may be a second, so
this checks all of them.

Compares, per cited entry:
  * the surname in the key against the FIRST author's surname;
  * the year in the key against the `year` field.

    env PYTHONHASHSEED=2025 python reporting/bibkey_consistency.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

OUT = ROOT / "reports" / "bibkey_consistency.json"

#: Keys that deliberately do not name their first author, with the reason.
#: Declared here so an intentional key is distinguishable from a swap -- the
#: point of the check is that an unexplained mismatch stays visible.
INTENTIONAL = {
    "SR2026icu": "deliberate anonymized self-reference; the key encodes the "
                 "role the work plays in this manuscript, not its authorship",
}

_KEY = re.compile(r"^([A-Za-z][A-Za-z'\-]*)(\d{4})(.*)$")


def _norm(s: str) -> str:
    """Lowercase, drop everything that is not a letter.

    'Alvarez-Melis', 'van der Schaar' and 'Le Morvan' are one surname each,
    and a key cannot carry the spaces and hyphens, so the comparison has to
    happen with those removed on both sides.
    """
    return re.sub(r"[^a-z]", "", s.lower())


def first_author_surname(fields: dict) -> str | None:
    """The surname of the first author, in either BibTeX name order."""
    raw = (fields.get("author") or fields.get("editor") or "").strip()
    if not raw:
        return None
    first = re.split(r"\s+and\s+", raw)[0].strip().strip("{}")
    if not first:
        return None
    if "," in first:                       # "Alvarez-Melis, David"
        return first.split(",")[0].strip().strip("{}")
    parts = [p for p in first.split() if p]
    if not parts:
        return None
    # "David Alvarez-Melis": the surname is the tail, and a lowercase particle
    # ("van", "de", "le") belongs to it.
    i = len(parts) - 1
    while i > 0 and parts[i - 1][:1].islower():
        i -= 1
    return " ".join(parts[i:]).strip("{}")


def check_key(key: str, fields: dict) -> dict:
    m = _KEY.match(key)
    rec = {"key": key, "key_surname": None, "key_year": None,
           "author_surname": None, "field_year": fields.get("year", ""),
           "surname_verdict": "UNPARSED", "year_verdict": "UNPARSED",
           "note": ""}
    if not m:
        rec["note"] = "key does not parse as <Surname><Year><suffix>"
        return rec
    ksur, kyear, _ = m.groups()
    rec["key_surname"], rec["key_year"] = ksur, kyear

    asur = first_author_surname(fields)
    rec["author_surname"] = asur
    if key in INTENTIONAL:
        rec["surname_verdict"] = "INTENTIONAL"
        rec["note"] = INTENTIONAL[key]
    elif asur is None:
        rec["surname_verdict"] = "NO-AUTHOR-FIELD"
        rec["note"] = ("entry carries no author or editor, so the key cannot "
                       "be checked against one")
    else:
        k, a = _norm(ksur), _norm(asur)
        rec["surname_verdict"] = ("MATCH" if k == a
                                  else "PREFIX" if a.startswith(k) and len(k) >= 3
                                  else "MISMATCH")

    y = str(fields.get("year", "")).strip()
    ym = re.search(r"\d{4}", y)
    if not ym:
        rec["year_verdict"] = "NO-YEAR-FIELD"
    else:
        rec["year_verdict"] = "MATCH" if ym.group(0) == kyear else "MISMATCH"
        rec["field_year"] = ym.group(0)
    return rec


def scan() -> dict:
    from reporting.bib_inventory import parse_bib, cited_keys
    bib = parse_bib()
    try:
        cited = cited_keys()
    except FileNotFoundError:
        cited = list(bib)
    rows = [check_key(k, bib[k]["fields"]) for k in cited if k in bib]
    from collections import Counter
    rec = {
        "n_checked": len(rows),
        "surname": dict(Counter(r["surname_verdict"] for r in rows)),
        "year": dict(Counter(r["year_verdict"] for r in rows)),
        "surname_mismatches": [r for r in rows
                               if r["surname_verdict"] == "MISMATCH"],
        "year_mismatches": [r for r in rows if r["year_verdict"] == "MISMATCH"],
        "unverifiable": [r for r in rows
                         if r["surname_verdict"] in ("NO-AUTHOR-FIELD",
                                                     "UNPARSED")],
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    return rec


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    # the real defect this check was written for
    r = check_key("Kim2024ehratt",
                  {"author": "Kowsar, Ibna and Rabbani, Shourav B. and "
                             "Samad, Manar D.", "year": "2024"})
    chk(r["surname_verdict"] == "MISMATCH",
        "a key naming an author who is not on the paper is a MISMATCH")

    # compound surnames a key cannot spell exactly are NOT mismatches
    for key, auth in (("AlvarezMelis2018", "Alvarez-Melis, David and Jaakkola"),
                      ("LeMorvan2021", "Le Morvan, Marine and Josse, Julie"),
                      ("VanBuuren1999", "van Buuren, Stef and Oudshoorn"),
                      ("Beaulieu-Jones2017", "Beaulieu-Jones, Brett K.")):
        r = check_key(key, {"author": auth, "year": key[-4:] if key[-4:].isdigit()
                            else re.search(r"\d{4}", key).group(0)})
        chk(r["surname_verdict"] in ("MATCH", "PREFIX"),
            f"compound surname survives: {key} vs {auth.split(',')[0]}")

    # first-name-first order
    r = check_key("Vaswani2017", {"author": "Ashish Vaswani and Noam Shazeer",
                                  "year": "2017"})
    chk(r["surname_verdict"] == "MATCH", "'First Last' name order parses")
    r = check_key("Schaar2018", {"author": "Jinsung Yoon and Mihaela van der "
                                           "Schaar", "year": "2018"})
    chk(r["surname_verdict"] == "MISMATCH",
        "a key naming the LAST author, not the first, is a MISMATCH")

    # the year half
    r = check_key("Cao2018", {"author": "Cao, Wei", "year": "2019"})
    chk(r["year_verdict"] == "MISMATCH", "a key year that is not the pub year "
                                         "is a MISMATCH")
    r = check_key("SR2026icu", {"author": "Anonymous, A.", "year": "2026"})
    chk(r["surname_verdict"] == "INTENTIONAL",
        "a declared intentional key is not reported as a mismatch")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    r = scan()
    print(f"checked {r['n_checked']} cited entries")
    print(f"  surname {r['surname']}")
    print(f"  year    {r['year']}")
    for name in ("surname_mismatches", "year_mismatches", "unverifiable"):
        if r[name]:
            print(f"\n{name}:")
            for x in r[name]:
                print(f"  {x['key']:22s} key={x['key_surname']}/{x['key_year']}"
                      f"  fields={x['author_surname']}/{x['field_year']}"
                      f"  {x['note']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
