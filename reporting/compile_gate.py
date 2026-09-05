"""Compile gate for the paper package (P5 follow-up adjudication,
2026-08-27): two mechanical checks joined to the existing error/overfull
discipline -- (1) the log must contain no undefined references or
citations; (2) `pdftotext` output must contain no '??'. A broken
cross-reference reached a review package once; this gate exists so it
cannot happen silently again.

    python reporting/compile_gate.py --tex paperY_main.tex
    python reporting/compile_gate.py --selftest
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(CODE_ROOT))
#: paper_R1 is main/ + esm/ since P7-A SS2; a package's assembled
#: view is still flat. One resolver answers for all three layouts.
from experiments.package_layout import paper_file  # noqa: E402

PAPER = CODE_ROOT.parent / "paper_R1"
#: P7-A SS2: each document now compiles in its OWN directory, and that is
#: the point -- pdflatex resolves \input and \includegraphics relative to
#: the working directory, so the gate has to run where the document lives.
PAPER_DIRS = {"paperY_main.tex": "main", "paperY_ESM.tex": "esm"}


def _source_date_epoch(paper_dir: Path) -> str:
    """The moment the SOURCE was fixed, not the moment the PDF was built.

    pdfTeX stamps a wall clock into /CreationDate and derives /ID from it, so
    two compiles of identical sources produced different bytes and "the PDF
    did not change" could never be checked in bytes -- only in pixels. With
    SOURCE_DATE_EPOCH the stamp becomes the commit the manuscript was built
    from, which is deterministic AND is still a real date: the date the
    sources carry. Same commit, same sources, same bytes.
    """
    r = subprocess.run(["git", "log", "-1", "--format=%ct"],
                       cwd=str(Path(__file__).resolve().parent.parent),
                       capture_output=True, text=True)
    ts = r.stdout.strip()
    if ts.isdigit():
        return ts
    # No git (a package copy, an export): fall back to the source's own mtime,
    # which is the same kind of fact, measured a different way.
    return str(int(max(f.stat().st_mtime for f in paper_dir.glob("*.tex"))))


def _repro_env(paper_dir: Path) -> dict:
    import os
    return {**os.environ, "SOURCE_DATE_EPOCH": _source_date_epoch(paper_dir),
            "FORCE_SOURCE_DATE": "1"}


def gate(tex: str, paper_dir: Path = PAPER, *, runs: int = 3,
         max_errors: int = 0, check_overfull: bool = True,
         facts: bool = True) -> dict:
    #: WHERE the document is compiled, which is not the same as where the
    #: paper tree starts. Each document owns a directory it can build in on
    #: its own (P7-A SS2), and pdflatex resolves \input, \includegraphics
    #: and \bibliography from the working directory -- so running the gate
    #: one level up finds the fragments through the fallback path and proves
    #: nothing about the directory that will actually be submitted. A flat
    #: tree (a package's assembled view, the selftest's fixtures) has no
    #: subdirectory and answers unchanged.
    work = paper_dir
    if tex in PAPER_DIRS and (paper_dir / PAPER_DIRS[tex]).is_dir():
        work = paper_dir / PAPER_DIRS[tex]
    # P5R-H SS0.3: the single-facts gate is a PRE-COMPILE precondition for
    # the two manuscript documents -- banned-variant hits in the sources
    # stop the compile; the rendered-PDF layer and the numeric checks run
    # after. Diff builds and selftest fixtures are exempt (facts=False /
    # non-manuscript tex names).
    facts_pre = None
    if facts and tex in ("paperY_main.tex", "paperY_ESM.tex"):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from reporting import facts_gate
        pats = facts_gate._load_registry()
        src_hits = facts_gate.scan_sources(pats)
        if src_hits:
            return {"tex": tex, "facts_source_hits": len(src_hits),
                    "first_hits": [f"{h['layer']}::{h['variant']}"
                                   for h in src_hits[:8]],
                    "pass": False}
        # P7-A SS1.2: mixed British/American spelling is a standing gate, and
        # it is a PRE-COMPILE one for the same reason the terminology scan is
        # -- the cheapest moment to catch it is before a PDF exists to be
        # inspected page by page.
        sp_hits = facts_gate.scan_orthography(
            {f"tex:{f.name}": facts_gate._strip_tex_comments(f.read_text())
             for f in facts_gate.TEX_FILES if f.exists()})
        if sp_hits:
            return {"tex": tex, "spelling_source_hits": len(sp_hits),
                    "first_spellings": [f"{h['layer']}::{h['found']} -> "
                                        f"{h['canonical']}"
                                        for h in sp_hits[:8]],
                    "pass": False}
        facts_pre = pats
        # P5R-M SS4.1: the bibliography gate is a pre-compile precondition for
        # the main text too. Field policy and verification are enforced
        # always; the per-citation-point support requirement is enforced once
        # reports/citation_support.json exists, so the gate does not block a
        # compile before that record has been built for the first time.
        if tex == "paperY_main.tex":
            from reporting import bib_gate
            strict = (Path(bib_gate.SUPPORT).exists())
            # Items formally awaiting adjudication (P5R-M SS1.4 / SS3.4) are
            # tolerated HERE and printed: the paper has to build while they
            # wait. Packaging gate 13 does not tolerate them.
            b = bib_gate.check(strict_support=strict, tolerate_open=True)
            if not b["pass"]:
                return {"tex": tex, "bib_gate": b["failures"],
                        "bib_support_enforced": strict, "pass": False}
            bib_deferred = b["deferred_to_adjudication"]
    for _ in range(runs):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex],
                       cwd=work, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, env=_repro_env(work))
    log = (work / tex.replace(".tex", ".log")).read_text(
        errors="replace")
    pdf = work / tex.replace(".tex", ".pdf")
    errors = len(re.findall(r"^!", log, re.M))
    overfull = len(re.findall(r"^Overfull", log, re.M))
    undef = re.findall(r"(?:Reference|Citation) `([^']*)'.*undefined", log)
    txt = subprocess.run(["pdftotext", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    qq = txt.count("??")
    out = {"tex": tex, "errors": errors, "overfull": overfull,
           "undefined": undef, "qq_in_pdf": qq,
           "pages": int(re.search(r"Pages:\s*(\d+)", subprocess.run(
               ["pdfinfo", str(pdf)], capture_output=True,
               text=True).stdout).group(1))}
    out["pass"] = (errors <= max_errors and undef == [] and qq == 0
                   and (overfull == 0 or not check_overfull))
    if facts_pre is not None:
        if tex == "paperY_main.tex":
            out["bib_deferred_to_adjudication"] = locals().get(
                "bib_deferred", {})
        from reporting import facts_gate
        pdf_hits = facts_gate.scan_pdf([pdf], pats=facts_pre)
        num_errs = facts_gate.numeric_checks()
        out["facts_pdf_hits"] = len(pdf_hits)
        out["facts_numeric_errors"] = num_errs
        #: The third orthography layer: what the READER holds. The sources can
        #: be clean while the page is not, because a fragment arrives from a
        #: generator and a figure arrives from a designer.
        sp_pdf = facts_gate.scan_orthography()
        out["spelling_hits"] = len(sp_pdf)
        if sp_pdf:
            out["first_spellings"] = [f"{h['layer']}::{h['found']}"
                                      for h in sp_pdf[:8]]
        out["pass"] = out["pass"] and not sp_pdf
        if pdf_hits:
            out["first_hits"] = [f"{h['layer']}::{h['variant']}"
                                 for h in pdf_hits[:8]]
        out["pass"] = out["pass"] and not pdf_hits and not num_errs

        # The rendered page, in pixels. LaTeX raises no warning for a table
        # rule drawn through its caption's descenders, or for a caption that
        # overruns the block and lets the folio print inside it -- there is no
        # Overfull \vbox for either. Both shipped, and only the per-page human
        # inspection saw them. This is that inspection, mechanized.
        from reporting import render_checks
        rc = render_checks.run(pdf)
        out["render_rule_collisions"] = len(rc["rule_collisions"])
        out["render_block_overruns"] = len(rc["block_overruns"])
        out["render_figure_pages"] = rc["figure_pages"]
        if not rc["pass"]:
            out["render_first"] = (rc["rule_collisions"][:4]
                                   + rc["block_overruns"][:4])
        out["pass"] = out["pass"] and rc["pass"]
    return out


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "good.tex").write_text(
            "\\documentclass{article}\\begin{document}"
            "\\section{A}\\label{s}Sect~\\ref{s}.\\end{document}")
        g = gate("good.tex", d)
        check(g["pass"] and g["undefined"] == [] and g["qq_in_pdf"] == 0,
              "clean document passes the gate")
        (d / "bad.tex").write_text(
            "\\documentclass{article}\\begin{document}"
            "Table~\\ref{tab:nowhere}.\\end{document}")
        b = gate("bad.tex", d)
        check(not b["pass"] and b["undefined"] == ["tab:nowhere"]
              and b["qq_in_pdf"] == 1,
              "undefined ref caught by BOTH checks (log + pdftotext ??)")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex")
    ap.add_argument("--max-errors", type=int, default=0)
    ap.add_argument("--no-overfull-check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = gate(a.tex, max_errors=a.max_errors,
             check_overfull=not a.no_overfull_check)
    print(r)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
