"""Do our author lists match the authoritative records, name by name?

P5R-M follow-through (2026-08-29). The round-2 field verification compared
entries against authoritative records and passed 55 of 55 -- but it compared
them at the level a reader would: surnames, order, year, venue. Checking a
citation for SAITS afterwards turned up `Du, W. J.` where Crossref gives one
given name, Wenjie. An initial that does not exist is small, but it is the
same class of defect as the two the round did catch (an author order swapped,
a name abbreviated past recognition), and one instance means the rest have to
be looked at the same way.

This does that mechanically, from the record itself:

  * surname SEQUENCE -- exact, in order (this is what catches a swap);
  * author COUNT -- ours against theirs;
  * GIVEN names -- ours must be an abbreviation of theirs and nothing more.
    "Du, W." is a fine short form of Wenjie. "Du, W. J." is not: it asserts
    a middle initial the record does not have.

Records come from Crossref by DOI, DataCite for 10.48550 (arXiv) DOIs, and
OpenAlex by title for entries with no DOI. Responses are cached so the scan is
re-runnable without the network.

    env PYTHONHASHSEED=2025 python reporting/bib_author_fidelity.py \
        [--cache DIR] [--offline] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

OUT = ROOT / "reports" / "bib_author_fidelity.json"
UA = "bib-audit (dengou@toki.waseda.jp)"


def _fold(s: str) -> str:
    """Strip accents and LaTeX accent commands, lowercase, letters only."""
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)          # \v, \^ ... as commands
    s = re.sub(r"[\\{}'\"`^~]", "", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # NFKD leaves these alone -- a stroke is part of the letter, not a mark --
    # and without the mapping 'Lukasz' reads as a different initial from
    # 'Lukasz' with the stroke.
    s = s.translate(str.maketrans("\u0141\u0142\u00d8\u00f8\u0110\u0111"
                                  "\u00de\u00fe\u0126\u0127",
                                  "LlOoDdTtHh"))
    return re.sub(r"[^a-z]", "", s.lower())


def split_names(author_field: str) -> list:
    """BibTeX author field -> [(surname, given), ...] in order."""
    out = []
    for raw in re.split(r"\s+and\s+", author_field or ""):
        raw = raw.strip().strip("{}").strip()
        if not raw:
            continue
        if "," in raw:
            fam, giv = raw.split(",", 1)
        else:
            parts = raw.split()
            if len(parts) == 1:
                fam, giv = parts[0], ""
            else:
                i = len(parts) - 1
                while i > 0 and parts[i - 1][:1].islower():
                    i -= 1
                fam, giv = " ".join(parts[i:]), " ".join(parts[:i])
        out.append((fam.strip().strip("{}"), giv.strip().strip("{}")))
    return out


def _initials(given: str) -> list:
    """['Matthew','B','A'] -> ['m','b','a']; handles 'M.B.A.' too."""
    # A letter class, not an ASCII range: 'Lukasz' with a stroke has to
    # tokenize as one word or its initial comes out as the second letter.
    toks = re.findall(r"[^\W\d_]+", given or "", re.UNICODE)
    return [_fold(t)[:1] for t in toks if _fold(t)]


def given_compatible(ours: str, theirs: str) -> tuple:
    """Is OUR given-name string a faithful (possibly abbreviated) form?

    Returns (ok, why). Ours may abbreviate theirs; ours may NOT add tokens
    theirs does not have, and each of our tokens must match theirs in order.
    """
    o, t = _initials(ours), _initials(theirs)
    if not o:
        return (True, "we give no forename")
    if len(o) > len(t):
        # NOT a defect on its own. Crossref/OpenAlex deposits frequently drop a
        # middle initial the paper itself carries (AAAI's deposit for SAnD
        # gives 'Jayaraman Thiagarajan' where the paper says 'Jayaraman J.'),
        # so "we have more detail than the record" needs a second source, not
        # an automatic verdict.
        return ("REVIEW", f"we give {len(o)} forename token(s) ({ours!r}) "
                          f"where the record has {len(t)} ({theirs!r}); the "
                          f"record may be the abbreviated one")
    # ours must be a prefix-in-order subsequence match on initials
    for a, b in zip(o, t):
        if a != b:
            return (False, f"forename initials differ: ours {ours!r} vs "
                           f"record {theirs!r}")
    # a full token we spell out must match the record's spelling
    ot = [x for x in re.findall(r"[^\W\d_]+", ours or "", re.UNICODE)
          if len(x) > 1]
    tt = [x for x in re.findall(r"[^\W\d_]+", theirs or "", re.UNICODE)
          if len(x) > 1]
    for a in ot:
        if _fold(a) and not any(_fold(a) == _fold(b) for b in tt):
            return (False, f"we spell out {a!r}, which is not in the record's "
                           f"{theirs!r}")
    return (True, "")


def compare(ours: str, theirs: list) -> dict:
    """theirs = [(family, given), ...] from the authoritative record."""
    mine = split_names(ours)
    issues = []
    if len(mine) != len(theirs):
        # More authors in the record than in our entry means we dropped
        # someone. FEWER means the deposit is short -- government and society
        # deposits routinely truncate a long byline -- which needs a look but
        # is not our error to assert.
        issues.append({"kind": "author-count",
                       "class": "DEFECT" if len(theirs) > len(mine)
                                else "REVIEW",
                       "detail": f"we list {len(mine)}, the record has "
                                 f"{len(theirs)}"})
    for i, (m, t) in enumerate(zip(mine, theirs)):
        fm, ft = _fold(m[0]), _fold(t[0])
        if fm != ft:
            # A deposit that puts "A." in `given` and "Colin Cameron" in
            # `family`, or that swaps the two fields outright, disagrees with
            # us without either of us being wrong. Containment is the
            # signature of that, so it goes to REVIEW; a genuinely different
            # name is the defect.
            # Ours SHORTER than the record is only benign when the extra
            # material is a forename we already carry -- a deposit that put
            # "Colin" in `family` where we put it in `given`. If the extra
            # material is a particle ("van der"), we dropped it, and that is
            # the Yoon/van der Schaar defect this check exists to catch.
            benign_short = (fm and ft and fm in ft
                            and ft.replace(fm, "") in _fold(m[1]))
            benign_long = fm and ft and ft in fm      # the deposit dropped it
            contained = benign_short or benign_long
            swapped = _fold(m[1]) and _fold(m[1]) == ft
            issues.append({"kind": "surname", "position": i + 1,
                           "class": "REVIEW" if (contained or swapped)
                                    else "DEFECT",
                           "detail": f"ours {m[0]!r} vs record {t[0]!r}"
                                     + (" (the record's own fields look "
                                        "transposed)" if swapped else "")})
            continue
        ok, why = given_compatible(m[1], t[1])
        if ok is not True:
            issues.append({"kind": "forename", "position": i + 1,
                           "class": "REVIEW" if ok == "REVIEW" else "DEFECT",
                           "detail": f"{m[0]}: {why}"})
    # An order swap is the signature that matters, so name it as one. It does
    # NOT always show up as a surname mismatch: the swap this bibliography
    # actually had was 'Li, Yitan and Li, Lei' for 'Li, Lei and Li, Yitan',
    # where the surnames line up at every position and only the forenames
    # move. Compare the (surname, first initial) sequence instead -- if ours
    # is a permutation of theirs but not in their order, the authors were
    # reordered.
    if issues and len(mine) == len(theirs):
        sig = lambda xs: [(_fold(f), (_initials(g) or [""])[0])
                          for f, g in xs]
        ms, ts = sig(mine), sig(theirs)
        if ms != ts and sorted(ms) == sorted(ts):
            issues.append({"kind": "order", "class": "DEFECT",
                           "detail": "the same authors in a different order"})
    return {"n_ours": len(mine), "n_record": len(theirs), "issues": issues}


def _get(url: str, cache: Path, offline: bool) -> dict | None:
    cache.mkdir(parents=True, exist_ok=True)
    fn = cache / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-120:] + ".json")
    if fn.exists():
        try:
            return json.loads(fn.read_text())
        except Exception:
            pass
    if offline:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
    except Exception as e:
        return {"__error__": str(e)}
    fn.write_text(json.dumps(d))
    time.sleep(0.2)
    return d


def record_for(entry: dict, cache: Path, offline: bool) -> tuple:
    """(authors, source) for one entry, from the best record we can fetch."""
    f = entry["fields"]
    doi = (f.get("doi") or "").strip()
    if doi:
        if doi.lower().startswith("10.48550"):
            d = _get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}",
                     cache, offline)
            cr = (((d or {}).get("data") or {}).get("attributes") or {})
            au = [(a.get("familyName") or a.get("name", ""), a.get("givenName", ""))
                  for a in cr.get("creators", [])]
            if au:
                return au, f"DataCite {doi}"
        d = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}",
                 cache, offline)
        m = (d or {}).get("message") or {}
        au = [(a.get("family", ""), a.get("given", ""))
              for a in m.get("author", [])]
        # Crossref deposits sometimes carry a blank author slot (the PSB
        # volume does); an empty name is not a missing author of ours.
        au = [x for x in au if _fold(x[0])]
        if au:
            return au, f"Crossref {doi}"
    title = re.sub(r"[{}\\]", "", f.get("title", "")).strip()
    if title:
        q = urllib.parse.quote(title[:180])
        # Ask for more than a handful: OpenAlex's top hit for "Attention Is
        # All You Need" is a 2025 re-index, and the 2017 paper is further
        # down. The year gate below rejects the wrong one, so the list has to
        # be long enough to still contain the right one.
        d = _get(f"https://api.openalex.org/works?filter=title.search:{q}"
                 f"&per-page=10", cache, offline)
        for w in (d or {}).get("results", []) or []:
            # A prefix match is not a match. Matching on the first 60 folded
            # characters pulled a completely different work in for a UCI
            # dataset entry and produced a confident author "defect" against
            # a paper we do not cite. Require the whole title.
            wt, ot = _fold(w.get("title", "")), _fold(title)
            # Titles collide. "Communities and Crime" is both a UCI dataset we
            # cite and a 1986 Reiss/Tonry volume we do not, and matching on
            # title alone produced a confident author defect against the wrong
            # book. Without a DOI, identity needs the year as well.
            oy = re.search(r"\d{4}", str(f.get("year", "")))
            wy = w.get("publication_year")
            year_ok = (not oy or not wy or abs(int(oy.group(0)) - int(wy)) <= 1)
            if wt and ot and year_ok and (
                    wt == ot or SequenceMatcher(None, wt, ot).ratio() >= 0.97):
                au = []
                for a in w.get("authorships", []):
                    nm = (a.get("author") or {}).get("display_name", "")
                    # 'Mihaela van der Schaar' is one surname with a particle;
                    # splitting on the last space gives 'Schaar' and invents a
                    # mismatch. Use the same parser we use on our own field.
                    parsed = split_names(nm.replace("\u2010", "-"))
                    if parsed:
                        au.append(parsed[0])
                au = [x for x in au if _fold(x[0])]
                if au:
                    return au, f"OpenAlex {w.get('id')}"
    return [], "NO-RECORD"


def scan(cache: Path, offline: bool = False) -> dict:
    from reporting.bib_inventory import parse_bib, cited_keys
    bib = parse_bib()
    try:
        cited = cited_keys()
    except FileNotFoundError:
        cited = list(bib)
    rows = []
    for k in cited:
        e = bib.get(k)
        if not e:
            continue
        au, src = record_for(e, cache, offline)
        if not au:
            rows.append({"key": k, "source": src, "verdict": "NO-RECORD",
                         "issues": []})
            continue
        c = compare(e["fields"].get("author") or e["fields"].get("editor", ""),
                    au)
        defects = [i for i in c["issues"] if i.get("class") == "DEFECT"]
        rows.append({"key": k, "source": src,
                     "verdict": ("DEFECT" if defects else
                                 "REVIEW" if c["issues"] else "CLEAN"),
                     "n_defects": len(defects),
                     "record_authors": [f"{g} {f}".strip() for f, g in au],
                     "our_authors": e["fields"].get("author", "")[:300],
                     **c})
    from collections import Counter
    rec = {"n_checked": len(rows),
           "verdicts": dict(Counter(r["verdict"] for r in rows)),
           "issue_kinds": dict(Counter(f"{i['kind']}/{i.get('class','?')}"
                                       for r in rows for i in r["issues"])),
           "defects": [r for r in rows if r["verdict"] == "DEFECT"],
           "review": [r for r in rows if r["verdict"] == "REVIEW"],
           "with_issues": [r for r in rows if r["issues"]],
           "no_record": [r["key"] for r in rows if r["verdict"] == "NO-RECORD"],
           "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    return rec


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    # the defect that prompted this check
    r = compare("Du, W. J. and C\\^{o}t\\'{e}, D. and Liu, Y.",
                [("Du", "Wenjie"), ("Côté", "David"), ("Liu", "Yan")])
    chk(any(i["kind"] == "forename" and i["class"] == "REVIEW"
            for i in r["issues"]),
        "extra forename detail is raised for REVIEW, not called a defect")

    r = compare("Kaiser, L and Polosukhin, I.",
                [("Kaiser", "\u0141ukasz"), ("Polosukhin", "Illia")])
    chk(not r["issues"], "a stroked letter folds to its base ('L' for Lukasz)")

    r = compare("Yoon, J. and Schaar, M.",
                [("Yoon", "Jinsung"), ("van der Schaar", "Mihaela")])
    chk(any(i["kind"] == "surname" and i["class"] == "DEFECT"
            for i in r["issues"]),
        "a dropped particle IS a defect")

    r = compare("Du, W. and C\\^{o}t\\'{e}, D. and Liu, Y.",
                [("Du", "Wenjie"), ("Côté", "David"), ("Liu", "Yan")])
    chk(not r["issues"], "a plain initial for a single forename is fine")

    r = compare("McDermott, Matthew B. A. and Wang, Shirly",
                [("McDermott", "Matthew B. A."), ("Wang", "Shirly")])
    chk(not r["issues"], "an exact multi-initial name passes")

    # the order swap the round did catch, restated mechanically
    r = compare("Cao, W. and Wang, D. and Li, Yitan and Li, Lei",
                [("Cao", "Wei"), ("Wang", "Dong"), ("Li", "Lei"),
                 ("Li", "Yitan")])
    chk(any(i["kind"] == "order" for i in r["issues"]),
        "an author-order swap is named as an order defect")

    r = compare("Yoon, J. S. and Jordon, J. and Schaar, M.",
                [("Yoon", "Jinsung"), ("Jordon", "James"),
                 ("van der Schaar", "Mihaela")])
    chk(any(i["kind"] == "surname" for i in r["issues"])
        and any(i["kind"] == "forename" for i in r["issues"]),
        "a dropped particle and an extra initial are both raised")

    r = compare("Alvarez-Melis, David and Jaakkola, Tommi",
                [("Alvarez-Melis", "David"), ("Jaakkola", "Tommi S.")])
    chk(not r["issues"], "the record having MORE detail than us is not a defect")

    r = compare("Fisher, A. and Rudin, C.",
                [("Fisher", "Aaron"), ("Rudin", "Cynthia"),
                 ("Dominici", "Francesca")])
    chk(any(i["kind"] == "author-count" for i in r["issues"]),
        "a missing author is caught")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path("/tmp/claude-1002") /
                                           "bib_records_cache"))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = scan(Path(a.cache), a.offline)
    print(f"checked {r['n_checked']}  {r['verdicts']}")
    print(f"issue kinds {r['issue_kinds']}")
    if r["no_record"]:
        print(f"no authoritative record fetched: {r['no_record']}")
    for row in r["with_issues"]:
        print(f"\n  {row['key']}  [{row['source']}]")
        print(f"    ours   : {row['our_authors'][:160]}")
        print(f"    record : {', '.join(row['record_authors'])[:160]}")
        for i in row["issues"]:
            print(f"    -> {i['kind']}: {i['detail']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
