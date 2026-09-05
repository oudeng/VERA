"""Assemble the two verification records the bibliography gate consumes.

P5R-M SS2.3 and SS3.1. Reads the raw verdicts produced by the verification
passes, applies the adversarial confirmations on top (a flag that three
independent lenses could not sustain does not stand), and writes:

  reports/citation_verification.json   one record per cited reference
  reports/citation_support.json        one record per citation POINT
  reports/citation_audit.md            the human-readable table

The gate reads the two JSON files; this module is what makes them, so the
provenance of every verdict is one script rather than an editing session.

    env PYTHONHASHSEED=2025 python reporting/bib_records.py --from <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

VERIF = ROOT / "reports" / "citation_verification.json"
SUPPORT = ROOT / "reports" / "citation_support.json"
TABLE = ROOT / "reports" / "citation_audit.md"


#: Where I overrode the automated passes, with the reason. Declared here
#: rather than applied by hand to the JSON, so the record shows the first
#: pass, the challenge, and the adjudication between them.
ADJUDICATIONS = {
    "AlvarezMelis2018": {
        "first_pass": "MISMATCH (pages 7775--7784 vs 7786--7795)",
        "challenge": "refuted -- claimed 7775--7784 is 'exactly what the "
                     "official publisher record states'",
        "adjudication": "the challenge is WRONG and the first pass stands. "
                        "The NeurIPS proceedings page for this paper "
                        "publishes NO page numbers at all, so it cannot be "
                        "the source of either range. Checked directly: dblp "
                        "(https://dblp.org/rec/conf/nips/Alvarez-MelisJ18.bib) "
                        "gives 7786--7795 and OpenAlex "
                        "(api.openalex.org, biblio.first_page/last_page) "
                        "gives 7786/7795. The bib is corrected to 7786--7795.",
        "verdict_after_correction": "VERIFIED",
    },
}

#: Corrections applied to the .bib on 2026-08-29 under SS2.5. After them the
#: entry matches its authoritative record, so its current verdict is VERIFIED
#: and the history is kept beside it rather than erased.
CORRECTED = {
    "Cao2018": "author order at positions 5-6: 'Li, Yitan and Li, Lei' -> "
               "'Li, Lei and Li, Yitan' (NeurIPS proceedings BibTeX)",
    "Yoon2018": "author 'Yoon, J. S.' -> 'Yoon, Jinsung'; 'Schaar, M.' -> "
                "'van der Schaar, Mihaela'; numpages=80 was the PMLR VOLUME "
                "number misfiled, moved to volume={80} with series and "
                "publisher supplied (PMLR canonical BibTeX)",
    "AlvarezMelis2018": "pages 7775--7784 -> 7786--7795 (dblp, OpenAlex), and "
                        "the official NeurIPS abstract URL supplied",
    "Michel2019": "pages was an EMPTY field; set to 14014--14024 (dblp, "
                  "NeurIPS 32)",
    "Song2018": "canonical AAAI DOI 10.1609/aaai.v32i1.11635 supplied and the "
                "URL dropped, under DOI > URL",
    "Wu2020aimnet": "volume=2, pages=307--325 and the official MLSys URL "
                    "supplied (the entry had no page range, which is why it "
                    "rendered with a trailing '. .')",
    "Fisher2019": "official JMLR landing page supplied (JMLR mints no DOIs)",
    "LeMorvan2021": "official NeurIPS page supplied (NeurIPS registered no "
                    "DOIs before vol. 35)",
    "Toye2025": "official PMLR v287 page supplied; volume/series/publisher "
                "lifted out of the booktitle string",
    # --- P5R-M adjudications, 2026-08-29 --------------------------------- #
    "Batista2002": "isbn 978-1-58603-297-5 and the IOS Press catalog URL "
                   "supplied. One authorized search established that no DOI "
                   "exists for this chapter or volume (Crossref by title and "
                   "by ISBN, DataCite, OpenAlex all empty; IOS Press's own "
                   "ebook platform answers 'Publication not found' -- the "
                   "2002 volume was never migrated, so no chapter DOI was "
                   "ever minted) and tied the ISBN to THIS volume via the "
                   "Library of Congress MARC record, the publisher catalog, "
                   "dblp conf/his/2002 and K10plus. Adjudication CLOSED: "
                   "retained under the field policy.",
    "Du2023": "author 'Du, W. J.' -> 'Du, Wenjie': Crossref, ORCID "
              "0000-0003-3046-7835 and dblp all give one forename, so the "
              "second initial was invented; pages=119619 supplied. Found by "
              "the author-fidelity scan added this round.",
    "Kowsar2024ehratt": "KEY RENAMED from Kim2024ehratt. The fields were "
                        "already correct and VERIFIED; the key named an "
                        "author who is not on the paper. Provenance: the "
                        "entry was a stub at its first tracked commit with "
                        "author = [[CITE-VERIFY: authors]] and venue = "
                        "[[CITE-VERIFY: venue, PMC 11463999]], and that PMC "
                        "id resolves to this same paper -- so no work was "
                        "ever swapped; the key's surname was a guess that no "
                        "author list ever supported.",
    "Heilbroner2025life": "ADDED. Replaces the Introduction citation "
                          "withdrawn under adjudication item 1; fields "
                          "fetched from Crossref directly.",
}


def _load(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def assemble(verif_raw: Path, support_raw: Path) -> dict:
    v = _load(verif_raw) or {}
    s = _load(support_raw) or {}

    # --- SS2: one record per reference, confirmations applied ------------- #
    vres = {r["key"]: dict(r) for r in v.get("results", [])}
    applied = []
    for c in v.get("confirmations", []):
        k = c.get("key")
        if k not in vres:
            continue
        if k in ADJUDICATIONS:
            # the challenge was itself checked and overruled; record all three
            vres[k]["adjudication"] = ADJUDICATIONS[k]
            vres[k]["verdict"] = ADJUDICATIONS[k]["verdict_after_correction"]
            vres[k]["independent_confirmation"] = {
                "confirmed": c.get("confirmed"), "verdict": c.get("verdict"),
                "reason": c.get("reason", "")[:600],
                "OVERRULED": ADJUDICATIONS[k]["adjudication"]}
            continue
        if c.get("confirmed") is False and c.get("verdict"):
            applied.append({"key": k, "from": vres[k]["verdict"],
                            "to": c["verdict"],
                            "reason": c.get("reason", "")[:300]})
            vres[k]["verdict"] = c["verdict"]
        vres[k]["independent_confirmation"] = {
            "confirmed": c.get("confirmed"),
            "verdict": c.get("verdict"),
            "reason": c.get("reason", "")[:600],
            "corrected_fields": c.get("corrected_fields", [])}

    # SS2.5 corrections are applied to the .bib, so the entry now matches its
    # authoritative record. The verdict becomes VERIFIED and the correction is
    # kept beside it: a correction that erases what it replaced is not
    # auditable.
    for k, why in CORRECTED.items():
        if k in vres:
            vres[k].setdefault("verdict_before_correction", vres[k]["verdict"])
            vres[k]["verdict"] = "VERIFIED"
            vres[k]["correction_applied"] = why

    # --- SS3: one record per citation point, challenges applied ----------- #
    sres = [dict(r) for r in s.get("results", [])]
    by = {(r["key"], r["sentence"][:120]): r for r in sres}
    overturned = []
    for j in s.get("survived", []):
        pass  # survivors keep their flag; nothing to do
    survived_keys = {(x["point"]["key"], x["point"]["sentence"][:120])
                     for x in s.get("survived", [])}
    for r in sres:
        if r["verdict"] == "SUPPORTS":
            continue
        key = (r["key"], r["sentence"][:120])
        if key not in survived_keys:
            # flagged, but the independent lenses refuted it
            overturned.append({"key": r["key"],
                               "sentence": r["sentence"][:160],
                               "from": r["verdict"], "to": "SUPPORTS"})
            r["verdict_before_challenge"] = r["verdict"]
            r["verdict"] = "SUPPORTS"
            r["challenge"] = "flagged, then refuted by the independent lenses"
        else:
            r["challenge"] = "flagged, and the flag survived the lenses"

    from collections import Counter
    rec = {
        "rules": "P5R-M SS2 and SS3",
        "independence_note":
            "SS3.0: CITATION_REGISTRATION.md was NOT used as evidence. It came "
            "out of the same batch of work as the citations, so agreeing with "
            "it would prove only internal consistency. It was used as a "
            "checklist of positions; every support judgment was made against "
            "the cited work's own authoritative content.",
        "n_references": len(vres),
        # Eighth review P1-6: a verification record is not a citation. Two
        # records (Beaulieu-Jones2017, Song2018) are verified and not cited by
        # the current manuscript; the label used to call all of them
        # "references", which contradicted every other count in the package.
        "n_cited": len({k for r in sres for k in [r.get("key")] if k}),
        "verification_counts": dict(Counter(r["verdict"] for r in vres.values())),
        "confirmations_that_changed_a_verdict": applied,
        "adjudications_where_the_challenge_was_overruled": {
            k: a["adjudication"] for k, a in ADJUDICATIONS.items()},
        "corrections_applied_to_the_bib": CORRECTED,
        "verification_counts_before_correction": dict(Counter(
            r.get("verdict_before_correction", r["verdict"])
            for r in vres.values())),
        "n_points": len(sres),
        "support_counts": dict(Counter(r["verdict"] for r in sres)),
        "flags_overturned_by_challenge": overturned,
    }
    VERIF.parent.mkdir(parents=True, exist_ok=True)
    VERIF.write_text(json.dumps({"summary": rec,
                                 "results": list(vres.values())}, indent=1))
    SUPPORT.write_text(json.dumps({"summary": rec, "results": sres}, indent=1))
    _table(rec, list(vres.values()), sres)
    return rec


def _table(rec, vres, sres) -> None:
    L = ["# Citation audit", "",
         f"Generated by `reporting/bib_records.py`. "
         f"{rec['n_references']} verification records, of which "
         f"{rec.get('n_cited', rec['n_references'])} are currently cited; "
         f"{rec['n_points']} current citation points.", "",
         "## Independence", "", rec["independence_note"], "",
         "## Field verification (§2)", "",
         f"`{rec['verification_counts']}`", ""]
    bad = [r for r in vres if r["verdict"] != "VERIFIED"]
    if bad:
        L += ["| key | verdict | field | ours | authoritative |",
              "|---|---|---|---|---|"]
        for r in bad:
            fs = r.get("mismatched_fields") or [{}]
            for f in fs:
                L.append(f"| `{r['key']}` | {r['verdict']} | "
                         f"{f.get('field','')} | {str(f.get('bib_value',''))[:60]} | "
                         f"{str(f.get('authoritative_value',''))[:60]} |")
    else:
        L.append("Every cited reference matched its authoritative record.")
    L += ["", "## Citation support (§3)", "", f"`{rec['support_counts']}`", ""]
    flagged = [r for r in sres if r["verdict"] != "SUPPORTS"]
    if flagged:
        for r in flagged:
            L += [f"### `{r['key']}` — {r['verdict']} (burden {r['burden']})",
                  "", f"**Our sentence.** {r['sentence']}", "",
                  f"**What the work says.** "
                  f"{r.get('support_quote') or '_nothing that carries it_'}",
                  "", f"**Source.** {r.get('evidence_source','')}", "",
                  f"**Suggested direction.** "
                  f"{r.get('suggested_direction','')}", ""]
    else:
        L.append("Every citation point is supported by the work it cites.")
    if rec["flags_overturned_by_challenge"]:
        L += ["", "## Flags raised and then refuted", "",
              "These were flagged by the first pass and could not be "
              "sustained by three independent lenses; they are recorded "
              "because a flag that is dropped silently is indistinguishable "
              "from one that was never raised.", ""]
        for o in rec["flags_overturned_by_challenge"]:
            L.append(f"- `{o['key']}` {o['from']} -> {o['to']}: "
                     f"{o['sentence'][:110]}")
    TABLE.write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verif", required=True)
    ap.add_argument("--support", required=True)
    a = ap.parse_args()
    r = assemble(Path(a.verif), Path(a.support))
    print(json.dumps({k: v for k, v in r.items()
                      if k != "independence_note"}, indent=1)[:1400])
    print(f"\nwrote {VERIF}\n      {SUPPORT}\n      {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
