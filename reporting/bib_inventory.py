"""Inventory of the bibliography and of every citation POINT in the manuscript.

P5R-M SS1 and SS3.1. Two things a bibliography audit needs and that no
existing artifact provides:

1. the field state of every CITED entry -- which of doi / url / isbn it
   carries, and whether it carries a hand-written "Available from:" prefix
   inside a note (the journal's .bst generates that prefix itself, so a
   second one in a field is what makes the rendered list inconsistent);
2. every citation POINT, not every reference: one work cited in four places
   supports four different claims, and each has to be judged on its own. Each
   point carries the whole sentence it sits in, quoted from the source with
   comments stripped, so a reader can see what the citation is being asked to
   carry.

    env PYTHONHASHSEED=2025 python reporting/bib_inventory.py [--selftest]
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
TEXS = [paper_file("paperY_main.tex"), paper_file("paperY_ESM.tex")]
OUT = ROOT / "reports" / "bib_inventory.json"

FIELD = re.compile(r"(\w+)\s*=\s*[{\"]", re.M)


def _strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def parse_bib(path: Path = None) -> dict:
    # Resolved at CALL time: a default argument would freeze the
    # root at import, and package mode's rebinding never reaches it.
    path = path or BIB
    """Every live entry: type, key, fields. Commented-out lines are ignored."""
    live = "\n".join(l for l in path.read_text().splitlines()
                     if not l.lstrip().startswith("%"))
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", live):
        start = m.end()
        depth, i = 1, live.find("{", m.start())
        i += 1
        while i < len(live) and depth:
            if live[i] == "{":
                depth += 1
            elif live[i] == "}":
                depth -= 1
            i += 1
        body = live[start:i - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name = fm.group(1).lower()
            j = fm.end()
            if j >= len(body):
                continue
            if body[j] == "{":
                d, k = 1, j + 1
                while k < len(body) and d:
                    if body[k] == "{":
                        d += 1
                    elif body[k] == "}":
                        d -= 1
                    k += 1
                fields[name] = body[j + 1:k - 1].strip()
            elif body[j] == '"':
                k = body.find('"', j + 1)
                fields[name] = body[j + 1:k].strip()
            else:
                k = body.find(",", j)
                fields[name] = body[j:k if k > 0 else len(body)].strip()
        out[m.group(2).strip()] = {"type": m.group(1).lower(),
                                   "key": m.group(2).strip(), "fields": fields}
    return out


def cited_keys(path: Path = None) -> list:
    # Resolved at CALL time: a default argument would freeze the
    # root at import, and package mode's rebinding never reaches it.
    path = path or BBL
    return re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", path.read_text())


def classify(entry: dict) -> dict:
    """P5R-M SS1's four buckets, plus the prefix problem."""
    f = entry["fields"]
    doi = f.get("doi", "").strip()
    url = f.get("url", "").strip()
    note = f.get("note", "")
    howpub = f.get("howpublished", "")
    isbn = f.get("isbn", "").strip()
    url_in_note = bool(re.search(r"\\url\{|https?://", note + howpub))
    prefix_in_field = bool(re.search(r"Available from|Retrieved from",
                                     note + howpub + url))
    if doi:
        bucket = "1-has-doi"
    elif url or url_in_note:
        bucket = "2-url-only"
    elif isbn and entry["type"] in ("book", "inbook"):
        bucket = "3-isbn-book"
    else:
        bucket = "4-none"
    return {"bucket": bucket, "doi": doi, "url": url, "isbn": isbn,
            "url_in_note": url_in_note, "prefix_in_field": prefix_in_field,
            "doi_and_url": bool(doi and (url or url_in_note)),
            "note": note[:200], "type": entry["type"]}


BST = paper_file("sn-vancouver-num.bst")


def types_that_emit_url(bst: Path = None) -> set:
    # Resolved at CALL time, for the same reason as above.
    bst = bst or BST
    """Entry types whose style function actually prints a `url` field.

    Not every type does. In this journal's style `@article` never calls
    output.web.refs, so a url on an @article is silently dropped and the
    reference renders with no locator at all -- while every .bib-level check
    sees a perfectly good url field and passes it. Read the answer out of the
    .bst rather than hard-coding it, so the rule follows the journal's file if
    the journal changes it.
    """
    if not Path(bst).exists():
        return set()
    src = Path(bst).read_text(errors="replace")
    out, cur = set(), None
    for line in src.splitlines():
        m = re.match(r"FUNCTION \{([\w.]+)\}", line)
        if m:
            cur = m.group(1)
        if cur and "output.web.refs" in line and "FUNCTION" not in line:
            out.add(cur)
    return out


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\(])")


def citation_points(texs=None) -> list:
    """Every \\cite in the manuscript sources, with the sentence it sits in.

    A sentence is what a reader would take as the claim the citation carries,
    so the unit is the sentence, not the paragraph and not the reference.
    """
    pts = []
    for tex in (texs or TEXS):
        raw = _strip_comments(tex.read_text())
        # a flat sentence stream, with positions preserved well enough to
        # attribute each cite to the sentence containing it
        # A window must not run back across a structural boundary: the first
        # citation in the Introduction otherwise picks up the tail of the
        # abstract, and the recorded "sentence" then contains text the
        # citation has nothing to do with.
        bounds = [0] + [b.end() for b in re.finditer(
            r"\\(?:maketitle|section|subsection|paragraph)\*?\{[^}]*\}"
            r"|\\end\{abstract\}|\\abstract\{", raw)] + [len(raw)]
        for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", raw):
            prev = max([b for b in bounds if b <= m.start()], default=0)
            nxt = min([b for b in bounds if b > m.end()], default=len(raw))
            lo = max(prev, m.start() - 900)
            hi = min(nxt, m.end() + 900)
            window = raw[lo:hi]
            rel = m.start() - lo
            parts, pos = [], 0
            for s in _SENT_END.split(window):
                parts.append((pos, pos + len(s), s))
                pos += len(s) + 1
            sent = next((s for a, b, s in parts if a <= rel <= b), window)
            sent = " ".join(sent.split())
            for key in [k.strip() for k in m.group(1).split(",") if k.strip()]:
                pts.append({"key": key, "file": tex.name,
                            "char": m.start(), "sentence": sent})
    return pts


def build() -> dict:
    bib = parse_bib()
    cited = cited_keys()
    pts = citation_points()
    entries = {}
    for k in cited:
        e = bib.get(k)
        entries[k] = {"key": k, "present_in_bib": e is not None,
                      **(classify(e) if e else {"bucket": "MISSING"}),
                      "fields": (e or {}).get("fields", {})}
    from collections import Counter
    rec = {
        "n_bib_entries_live": len(bib),
        "n_cited": len(cited),
        "n_citation_points": len(pts),
        "buckets": dict(Counter(v["bucket"] for v in entries.values())),
        "doi_and_url_both": sorted(k for k, v in entries.items()
                                   if v.get("doi_and_url")),
        "prefix_written_into_a_field": sorted(
            k for k, v in entries.items() if v.get("prefix_in_field")),
        "points_per_key": dict(Counter(p["key"] for p in pts)),
        "keys_cited_in_text_but_not_in_bbl": sorted(
            {p["key"] for p in pts} - set(cited)),
        "cited_in_bbl_but_no_point_found": sorted(
            set(cited) - {p["key"] for p in pts}),
        "entries": entries,
        "citation_points": pts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    return rec


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    bib = parse_bib()
    check(len(bib) > 100, f"the bib parses ({len(bib)} live entries)")
    some = next(iter(bib.values()))
    check("fields" in some and some["fields"], "entries carry their fields")
    check(any("doi" in v["fields"] for v in bib.values()), "doi fields parsed")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.tex"
        p.write_text("First one \\cite{A}. Second says X \\cite{B,C}.\n"
                     "% a comment \\cite{NEVER}\n")
        pts = citation_points([p])
        check([x["key"] for x in pts] == ["A", "B", "C"],
              "one point per key, comments stripped")
        check("Second says X" in pts[1]["sentence"],
              "each point carries its own sentence, not the paragraph")
        check(pts[0]["sentence"] != pts[1]["sentence"],
              "two cites in different sentences get different sentences")

    r = build()
    check(r["n_citation_points"] > r["n_cited"],
          f"points ({r['n_citation_points']}) exceed references "
          f"({r['n_cited']}) -- works are cited more than once")
    check(not r["keys_cited_in_text_but_not_in_bbl"],
          f"every cited key resolves ({r['keys_cited_in_text_but_not_in_bbl']})")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    r = build()
    print(f"live bib entries      {r['n_bib_entries_live']}")
    print(f"cited references      {r['n_cited']}")
    print(f"citation POINTS       {r['n_citation_points']}")
    print(f"buckets               {r['buckets']}")
    print(f"doi AND url both      {len(r['doi_and_url_both'])} "
          f"{r['doi_and_url_both'][:6]}")
    print(f"prefix in a field     {len(r['prefix_written_into_a_field'])} "
          f"{r['prefix_written_into_a_field'][:6]}")
    if r["cited_in_bbl_but_no_point_found"]:
        print(f"cited but no point found: "
              f"{r['cited_in_bbl_but_no_point_found']}")
    top = sorted(r["points_per_key"].items(), key=lambda x: -x[1])[:6]
    print(f"most-cited            {top}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
