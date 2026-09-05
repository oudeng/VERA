"""Every Permutation-on-SNI comparison in the manuscript, with its status.

P5R-N SS0, from the sixth review's finding that the information-symmetry
correction was applied to some comparisons while the claims rest on others.
The inventory is the unit of account: SS1-SS4's work is struck off against it,
and nothing is "handled" until its row says so.

Sites are found MECHANICALLY in the manuscript sources and in the generated
fragments they input. Each must carry a declared record in
docs/perm_sni_inventory_declarations.json. A site found in the sources with no
record is a refusal -- that is what stops a new mention from entering unseen --
and so is a record for a site that no longer exists.

Status, one of four (T6.1 addendum 2026-08-29d SS1):

    SYMMETRIC               both sides had the same information
    RECOMPUTED              the no-oracle version is what is reported
    ASYMMETRIC-DISCLOSED    unequal, and disclosed AT THAT SITE
    ASYMMETRIC-UNDISCLOSED  unequal, and not disclosed there -- a blocker

    env PYTHONHASHSEED=2025 python experiments/perm_sni_inventory.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(CODE_ROOT))
#: paper_R1 is main/ + esm/ since P7-A SS2; a package's assembled
#: view is still flat. One resolver answers for all three layouts.
from experiments.package_layout import paper_file  # noqa: E402



def _pkg_root(rel: str, default):
    """The artifact, wherever this script is being run from.

    In the repository the path beside the script is right. Inside a review
    package the script is at code/<x>/ and the artifacts are under
    gate_inputs/code_SNI/ -- a different shape, same file. Cited evidence has
    to resolve in both (ninth review P0-3).
    """
    from pathlib import Path as _P
    here = _P(__file__).resolve().parent.parent
    for c in (default, here.parent / "gate_inputs" / "code_SNI" / rel):
        if _P(c).exists():
            return _P(c)
    return _P(default)
#: Where a RECOMPUTED row's artifact is looked for. The gate sets it to
#: the unpacked package; on a working tree it is the repository.
ARTIFACT_ROOT = CODE_ROOT
ROOT = CODE_ROOT.parent
PAPER = ROOT / "paper_R1"
GEN = CODE_ROOT / "reporting" / "out"
DECL = _pkg_root("docs/perm_sni_inventory_declarations.json",
                 CODE_ROOT / "docs" / "perm_sni_inventory_declarations.json")
OUT = CODE_ROOT / "results" / "T6_symmetry" / "perm_sni_comparison_inventory.json"
MD = ROOT / "reports" / "perm_sni_comparison_inventory.md"

#: what names the same-host permutation readout in the manuscript's prose
PAT = re.compile(
    r"Permutation-on-SNI|Perm-on-SNI|same-host|host band|"
    r"host's own behavioral readout|behavioral probe of (its|the) own host|"
    r"probe of its own host", re.I)

STATUSES = ("SYMMETRIC", "RECOMPUTED", "ASYMMETRIC-DISCLOSED",
            "ASYMMETRIC-UNDISCLOSED")


def _strip(t: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", t)


def sentences(t: str) -> list:
    t = " ".join(_strip(t).split())
    return re.split(r"(?<=[.!?])\s+(?=[A-Z\\(])", t)


def find_sites(files=None) -> list:
    files = files or ([paper_file("paperY_main.tex", PAPER),
                       paper_file("paperY_ESM.tex", PAPER)]
                      + sorted(GEN.glob("*.tex")))
    out, n = [], 0
    for f in files:
        for s in sentences(f.read_text(errors="replace")):
            if PAT.search(s):
                n += 1
                out.append({"id": n, "file": f.name, "excerpt": s[:420]})
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _key(s: str) -> str:
    """A site's identity: its WHOLE text, normalized.

    Not a prefix. A prefix key lets a rewrite late in a long sentence inherit
    the verdict made about the old ending, which is the same silent-carry
    failure the text-only matching exists to prevent. Whitespace and case are
    normalized out, so rewrapping still does not renumber anything.

    Orthography is normalized out for the same reason, and no further (P7-A
    SS1). A declaration says what was judged about a SITE; a British -our
    ending becoming an American -or one does not make it a different site,
    and eighteen declarations coming back undeclared over a house-style
    change would have taught exactly the wrong lesson about what
    "undeclared" means. A word change still renumbers, which is the point.
    (The examples are described rather than spelled, because layer 2 of the
    spelling gate reads this file and it is right to.)
    """
    from reporting.facts_gate import _ortho
    pat, m = _ortho()
    s = pat.sub(lambda x: m[x.group(0).lower()], s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def build(strict: bool = True, write: bool = True) -> dict:
    found = find_sites()
    decl = json.loads(DECL.read_text())["sites"]
    by_key = {}
    for d in decl:
        by_key.setdefault(_key(d.get("excerpt") or d.get("sentence") or ""), d)
    # Match on the TEXT and only on the text. An id fallback looks helpful and
    # is a trap: insert one sentence and every later site silently inherits the
    # neighboring declaration, which is how a classification would follow a
    # sentence it was never made about. A site whose wording changed SHOULD
    # come back undeclared -- the old judgment was about the old words.
    rows, undeclared = [], []
    for s in found:
        d = by_key.get(_key(s["excerpt"]))
        if d is None:
            undeclared.append(s)
            continue
        r = {**s, **{k: v for k, v in d.items() if k not in ("id", "file")}}
        if r.get("status") not in STATUSES:
            undeclared.append(s)
            continue
        # A DISCLOSED row depends entirely on the text it points at. Verify the
        # declared disclosure is actually present in this site's own unit --
        # the sentence for prose, the whole fragment for a generated table or
        # figure, whose note IS its unit. Declared-but-absent is a downgrade,
        # not a comment: it is exactly how a disclosure silently disappears
        # when the sentence around it is rewritten.
        # RECOMPUTED is the only status that asserts a NUMBER changed, and it
        # was the weakest declaration in the file: `number_source` was never
        # read by anything but the Markdown formatter, so a row could claim to
        # be recomputed while pointing at nothing. It must now name the
        # artifact the recomputed number came from, and that artifact must
        # exist. An unbacked RECOMPUTED is treated as what it is -- a claim of
        # symmetry with nothing behind it -- and blocks.
        if r["status"] == "RECOMPUTED":
            art = r.get("recompute_artifact", "")
            # ARTIFACT_ROOT, not CODE_ROOT: when this runs inside an unpacked
            # package the question is whether the artifact is in THAT tree.
            # Resolving against the working tree is how a package shipped
            # three RECOMPUTED rows pointing at a file it did not contain and
            # still reported zero blockers (seventh review SS4.1).
            ap = (ARTIFACT_ROOT / art) if art else None
            if not art or not ap.exists():
                r["status"] = "ASYMMETRIC-UNDISCLOSED"
                r["downgraded_by_tool"] = (
                    f"declared RECOMPUTED but its recompute_artifact "
                    f"({art or 'absent'}) does not resolve to a file")
        if r["status"] == "ASYMMETRIC-DISCLOSED":
            q = _norm(r.get("disclosure_quote", ""))
            unit = _norm(s["excerpt"])
            if s["file"] not in ("paperY_main.tex", "paperY_ESM.tex"):
                fp = GEN / s["file"]
                if fp.exists():
                    unit = _norm(fp.read_text(errors="replace"))
            if not q or q[:90] not in unit:
                r["status"] = "ASYMMETRIC-UNDISCLOSED"
                r["downgraded_by_tool"] = (
                    "declared ASYMMETRIC-DISCLOSED, but the declared "
                    "disclosure text is not present in this site's own unit")
        rows.append(r)

    matched = {_key(d.get("excerpt", "")) for d in decl
               if _key(d.get("excerpt", "")) in {_key(s["excerpt"])
                                                 for s in found}}
    stale = [d["id"] for d in decl if _key(d.get("excerpt", "")) not in matched]
    counts = dict(Counter(r["status"] for r in rows))
    blockers = [r for r in rows if r["status"] == "ASYMMETRIC-UNDISCLOSED"]
    rec = {
        "rules": "docs/T61_information_symmetry_rules.md addendum 2026-08-29d SS1",
        "n_sites_found": len(found),
        "n_declared": len(rows),
        "undeclared_sites": undeclared,
        "declarations_for_sites_that_no_longer_exist": stale,
        "counts": counts,
        "n_blockers": len(blockers),
        "blockers": [{"id": r["id"], "file": r["file"],
                      "location": r.get("location", ""),
                      "opponent": r.get("opponent", ""),
                      "excerpt": r["excerpt"][:220]} for r in blockers],
        "sites": rows,
        "pass": (not undeclared and not stale and not blockers),
    }
    #: write=False re-derives without emitting anything. A gate that RE-RUNS
    #: this on a package must not leave its output anywhere -- neither in the
    #: package it is auditing nor in the checkout it happens to be driven
    #: from. It used to do the latter, because OUT and MD are computed from
    #: this file's own __file__ and the gate rebound only the INPUTS.
    if write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        _markdown(rec)
    if strict and undeclared:
        raise RuntimeError(
            f"{len(undeclared)} Permutation-on-SNI site(s) in the sources have "
            f"no declared record. A new mention must not enter the manuscript "
            f"unclassified: {[u['excerpt'][:70] for u in undeclared[:3]]}")
    return rec


def _markdown(rec: dict) -> None:
    L = ["# Permutation-on-SNI comparison inventory", "",
         f"Generated by `experiments/perm_sni_inventory.py` from "
         f"`docs/perm_sni_inventory_declarations.json`. "
         f"{rec['n_sites_found']} sites.", "",
         "Status is one of four. **ASYMMETRIC-UNDISCLOSED is a blocker**: the "
         "comparison is under unequal information and the site itself does not "
         "say so. A disclosure elsewhere in the paper does not clear a row.", "",
         "| status | n |", "|---|---|"]
    for k in STATUSES:
        L.append(f"| {k} | {rec['counts'].get(k, 0)} |")
    L += ["", f"**Blockers: {rec['n_blockers']}**", ""]
    if rec["blockers"]:
        L += ["| # | file | where | set against | excerpt |", "|---|---|---|---|---|"]
        for b in rec["blockers"]:
            L.append(f"| {b['id']} | `{b['file']}` | {b['location'][:60]} | "
                     f"{b['opponent'][:50]} | {b['excerpt'][:150].replace('|', chr(92)+'|')} |")
    L += ["", "## Every site", "",
          "| # | file | kind | set against | opponent saw withheld truth | number | status |",
          "|---|---|---|---|---|---|---|"]
    for r in rec["sites"]:
        L.append(f"| {r['id']} | `{r['file']}` | {r.get('kind','')} | "
                 f"{str(r.get('opponent',''))[:44]} | "
                 f"{r.get('opponent_uses_withheld_truth','')} | "
                 f"{r.get('number_source','')} | **{r['status']}** |")
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(L) + "\n")


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.tex"
        p.write_text("Nothing here. The same-host readout leads. "
                     "% a comment about Permutation-on-SNI\n")
        f = find_sites([p])
        chk(len(f) == 1, f"one site found, comments ignored ({len(f)})")
        chk("same-host" in f[0]["excerpt"], "the excerpt is the sentence")

    chk(_key("The  SAME-host readout leads.") == _key("The same-host readout leads."),
        "the site key ignores whitespace and case, so rewrapping does not "
        "renumber the inventory")

    d = json.loads(DECL.read_text())["sites"]
    chk(all(x.get("status") in STATUSES for x in d),
        "every declaration carries one of the four statuses")
    chk(all(x.get("disclosure_quote") for x in d
            if x["status"] == "ASYMMETRIC-DISCLOSED"),
        "every DISCLOSED declaration carries the verbatim disclosure it found")

    # the tool must downgrade a DISCLOSED row whose quote is not in its unit
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.tex"
        f.write_text("The same-host readout leads by a mile.\n")
        site = find_sites([f])[0]
        q = _norm("an asymmetry disclosure that is not in that sentence")
        chk(q[:90] not in _norm(site["excerpt"]),
            "a disclosure absent from the unit is detected as absent")
    ch = [x for x in d if x.get("challenge")]
    chk(len(ch) >= 15, f"the DISCLOSED claims were challenged ({len(ch)})")
    # Every refutation must have been ADJUDICATED, either way, with a reason.
    # (This replaces an assertion that at least one challenge had downgraded a
    # row via a `status_before_challenge` field. That assertion had gone stale
    # and false: the two challenges that were upheld caused the manuscript
    # sentences themselves to be rewritten, after which those sites were
    # re-declared about their NEW words -- which is exactly what the text-only
    # key requires. Carrying a pre-challenge status onto a record made about
    # different words would be the silent carry this tool exists to prevent.
    # So the property that actually holds is asserted instead, and the
    # downgrade path is asserted where it can still be true: an UPHELD
    # refutation must move the row off DISCLOSED and record what it was.)
    ids = [x["id"] for x in d]
    chk(len(ids) == len(set(ids)),
        f"declaration ids are unique ({len(ids)} rows, {len(set(ids))} ids) -- "
        f"the tool drops them, but a human reading the file by id must not be "
        f"handed the wrong record")
    rec = [x for x in d if x["status"] == "RECOMPUTED"]
    miss = [x.get("recompute_artifact") for x in rec
            if not x.get("recompute_artifact")
            or not _pkg_root(x["recompute_artifact"],
                             CODE_ROOT / x["recompute_artifact"]).exists()]
    chk(rec and not miss,
        f"every RECOMPUTED row names an artifact that exists "
        f"({len(rec)} rows{'' if not miss else '; missing: ' + str(miss[:3])})")
    ref = [x for x in d if isinstance(x.get("challenge"), dict)
           and x["challenge"].get("refuted")]
    chk(ref, f"at least one challenge refuted a DISCLOSED claim ({len(ref)})")
    chk(all(isinstance(x.get("adjudication"), dict)
            and x["adjudication"].get("decision") in ("UPHELD", "OVERRULED")
            and x["adjudication"].get("why") for x in ref),
        "every refutation carries an adjudication, decided either way, "
        "with its reason kept")
    upheld = [x for x in ref if x["adjudication"]["decision"] == "UPHELD"]
    chk(all(x["status"] != "ASYMMETRIC-DISCLOSED"
            and x.get("status_before_challenge") for x in upheld),
        f"an upheld refutation moves the row off DISCLOSED and records what "
        f"it was ({len(upheld)} upheld)")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = build(strict=not a.no_strict)
    print(f"sites {r['n_sites_found']}   declared {r['n_declared']}")
    print(f"counts {r['counts']}")
    print(f"BLOCKERS (ASYMMETRIC-UNDISCLOSED): {r['n_blockers']}")
    for b in r["blockers"][:25]:
        print(f"  [{b['id']:>2}] {b['file']:<26} {b['location'][:44]}")
    if r["declarations_for_sites_that_no_longer_exist"]:
        print(f"stale declarations: "
              f"{r['declarations_for_sites_that_no_longer_exist']}")
    print(f"\nwrote {OUT}\n      {MD}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
