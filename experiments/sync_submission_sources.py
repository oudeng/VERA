"""The two documents' working directories, and the flat views they submit as.

P5R-X, extended by P7-A SS2. Two layouts have to hold at once, and neither may
be maintained by hand:

  * paper_R1/main/ and paper_R1/esm/ -- one directory per document, each one
    COMPILABLE ON ITS OWN. `cd paper_R1/main && pdflatex paperY_main.tex` has
    to work with no environment set and no sibling directory present, which
    means the class, the style, the bibliography, the figures and the table
    fragments all have to be inside it.
  * VERA_paper_R1/main/ and VERA_paper_R1/esm/ -- the same two sets FLATTENED.
    Editorial Manager compiles from the files an author uploads and cannot
    process subfolders, so no directory path may survive in \\input,
    \\includegraphics or \\bibliography. The first author uploads the contents
    of VERA_paper_R1/main/.

Most of what a document needs is not written where the document lives: the
thirty-three generated fragments and two of the three figures are written by
the reporting generators into code_SNI/reporting/out/, and that stays their
one home. So both layouts are VIEWS, and three rules govern them:

  * ONE DIRECTION. The canonical home is the source. Nothing is ever edited
    in a view -- an edit made in one is lost at the next sync, silently, which
    is the worst way to lose an edit.
  * BYTE-IDENTICAL. Every copied file equals its source byte for byte, there
    is nothing extra, and there is nothing missing. --check asserts all three
    for both layouts and is wired into the packaging gates, so a stale or
    hand-edited view is a red gate rather than a surprise at submission.
  * ONE SET OF BYTES, THREE LAYOUTS. The .tex sources are layout-agnostic:
    \\genout tries Table/<name>, then the flat <name>, then the working-tree
    path, and \\graphicspath lists Fig/ and the flat directory. So the nested
    directory, the flat directory and the package's assembled view all compile
    from the same bytes -- which is what lets the restructure be proved in
    bytes rather than argued.

    PYTHONHASHSEED=2025 python experiments/sync_submission_sources.py
    PYTHONHASHSEED=2025 python experiments/sync_submission_sources.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
PAPER = ROOT / "paper_R1"
GEN = CODE_ROOT / "reporting" / "out"
VIEW = ROOT / "VERA_paper_R1"
MANIFEST = "MANIFEST.sha256"

DOCS = ("main", "esm")

#: The files each document owns outright -- they are written by hand and live
#: in the document's own directory, so they are not copied from anywhere.
#: Everything else a document needs is a copy, and every copy is checked.
OWNED = {
    "main": ["paperY_main.tex", "references_Y.bib", "paperY_main.bbl",
             "sn-jnl.cls", "sn-vancouver-num.bst", "Fig/Fig_vera.pdf"],
    "esm": ["paperY_ESM.tex", "esm_corrections.tex"],
}

#: The document each one is built from.
TEX = {"main": "paperY_main.tex", "esm": "paperY_ESM.tex"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pulled(doc: str) -> tuple:
    """(fragment names, figure names) the document's own source asks for.

    Derived from the source, never typed: a hand-kept list of thirty-three
    fragments goes stale the first time a section is added, and it goes stale
    SILENTLY -- the document still builds, because \\genout falls through to
    the working tree and finds the file there. It is the standalone directory
    and the submission that break, later, somewhere else.
    """
    s = (PAPER / doc / TEX[doc]).read_text()
    frags = {m if m.endswith(".tex") else m + ".tex"
             for m in re.findall(r"\\genout\{([^}]+)\}", s)}
    figs = {Path(m).name for m in
            re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)}
    return sorted(frags), sorted(figs)


def copies(doc: str) -> list:
    """(canonical source, path relative to the document directory)."""
    frags, figs = _pulled(doc)
    owned = set(OWNED[doc])
    out = []
    for name in frags:
        src = GEN / name
        if not src.exists():
            raise FileNotFoundError(
                f"{TEX[doc]} pulls in {name}, which is not in {GEN}. Run "
                f"reporting/regenerate_all.sh first.")
        out.append((src, f"Table/{name}"))
    for name in figs:
        if f"Fig/{name}" in owned:
            continue          # a delivered asset: the document IS its home
        src = GEN / name
        if not src.exists():
            raise FileNotFoundError(f"{TEX[doc]} includes {name}, absent from "
                                    f"{GEN} and not a registered asset")
        out.append((src, f"Fig/{name}"))
    return out


def files(doc: str) -> list:
    """Every file the document directory must hold, owned or copied."""
    return ([(PAPER / doc / rel, rel) for rel in OWNED[doc]]
            + [(src, rel) for src, rel in copies(doc)])


# --------------------------------------------------------------------- #
# the working directories: paper_R1/main and paper_R1/esm
# --------------------------------------------------------------------- #
def sync_working() -> dict:
    made = {}
    for doc in DOCS:
        n = 0
        for src, rel in copies(doc):
            dst = PAPER / doc / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or _sha(dst) != _sha(src):
                shutil.copy2(src, dst)
            n += 1
        #: A fragment that stops being pulled leaves a copy behind, and a
        #: stale copy in a directory that compiles on its own is exactly the
        #: figure-freshness defect one level down.
        keep = {rel for _, rel in files(doc)}
        for sub in ("Table", "Fig"):
            d = PAPER / doc / sub
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.is_file() and f"{sub}/{f.name}" not in keep:
                    if f.suffix in (".tex", ".pdf") and \
                            not f.name.endswith(".pptx"):
                        f.unlink()
        made[doc] = n
    return made


def check_working() -> dict:
    bad = []
    for doc in DOCS:
        d = PAPER / doc
        if not d.is_dir():
            bad.append(f"{doc}/ does not exist")
            continue
        for src, rel in files(doc):
            p = d / rel
            if not p.exists():
                bad.append(f"{doc}/{rel} missing")
            elif rel not in OWNED[doc] and _sha(p) != _sha(src):
                bad.append(f"{doc}/{rel} differs from "
                           f"{src.relative_to(ROOT)}")
    n = sum(len(files(d)) for d in DOCS)
    return {"pass": not bad, "n_files": n, "problems": bad,
            "detail": (f"paper_R1/main and paper_R1/esm hold {n} files, every "
                       f"copied one byte-identical to its canonical source"
                       if not bad else "; ".join(bad[:6]))}


# --------------------------------------------------------------------- #
# the submission views: VERA_paper_R1/main and VERA_paper_R1/esm, flat
# --------------------------------------------------------------------- #
def flat(doc: str) -> list:
    """(canonical source, FLAT name) -- Fig/ and Table/ collapse to the top."""
    out = [(src, Path(rel).name) for src, rel in files(doc)]
    names = [n for _, n in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{doc}: two sources flatten onto {dup}. Editorial "
                         f"Manager has one namespace; so must this view.")
    return out


def _head() -> str:
    r = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "(no git)"


def build() -> dict:
    made = sync_working()
    VIEW.mkdir(parents=True, exist_ok=True)
    total, removed = 0, []
    for doc in DOCS:
        d = VIEW / doc
        d.mkdir(parents=True, exist_ok=True)
        deps = flat(doc)
        keep = {name for _, name in deps} | {MANIFEST}
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name not in keep:
                f.unlink()
                removed.append(f"{doc}/{f.name}")
        for src, name in deps:
            shutil.copy2(src, d / name)
        head = _head()
        lines = [
            f"# VERA_paper_R1/{doc} -- the HISC R1 submission set for the "
            f"{'main article' if doc == 'main' else 'supplementary material'}.",
            "#",
            "# A FLAT VIEW. Editorial Manager cannot process subfolders, so",
            "# upload the files in THIS directory, not the directory itself.",
            f"# Sources live in paper_R1/{doc}/ and code_SNI/reporting/out/.",
            "# Never edit a file here: the next sync overwrites it. Edit the",
            "# source, re-run experiments/sync_submission_sources.py, and the",
            "# change arrives.",
            "#",
            f"# generated from code_SNI {head}",
            "# columns: sha256  flat-name  <-  source path (repo-relative)",
            "#",
            "# Check a local copy against this set with:",
            "#     sha256sum -c MANIFEST.sha256",
            "#",
        ]
        for src, name in deps:
            lines.append(f"{_sha(src)}  {name}")
        lines.append("#")
        lines.append("# provenance:")
        for src, name in deps:
            lines.append(f"#   {name}  <-  {src.relative_to(ROOT)}")
        (d / MANIFEST).write_text("\n".join(lines) + "\n")
        total += len(deps)
    #: The old flat-at-the-top layout is superseded by main/ + esm/. Files
    #: left over from it would be uploaded by anyone who kept the habit.
    for f in sorted(VIEW.iterdir()):
        if f.is_file():
            f.unlink()
            removed.append(f.name)
    return {"files": total, "per_doc": {d: len(flat(d)) for d in DOCS},
            "working_copies": made, "removed": removed, "head": _head()}


def check() -> dict:
    """Both layouts: every file identical to its source, nothing extra, nothing
    missing."""
    w = check_working()
    bad = list(w["problems"])
    if not VIEW.is_dir():
        bad.append("VERA_paper_R1 does not exist")
    else:
        stray = sorted(f.name for f in VIEW.iterdir() if f.is_file())
        if stray:
            bad.append(f"VERA_paper_R1 holds files outside main/ and esm/: "
                       f"{stray[:4]} -- the submission set is per document")
        for doc in DOCS:
            d = VIEW / doc
            if not d.is_dir():
                bad.append(f"VERA_paper_R1/{doc} does not exist")
                continue
            expected = {name: src for src, name in flat(doc)}
            present = {f.name for f in d.iterdir() if f.is_file()}
            for name in sorted(set(expected) - present):
                bad.append(f"VERA_paper_R1/{doc}/{name} missing")
            for name in sorted(present - set(expected) - {MANIFEST}):
                bad.append(f"VERA_paper_R1/{doc}/{name} is not in the set")
            for name in sorted(set(expected) & present):
                if _sha(expected[name]) != _sha(d / name):
                    bad.append(f"VERA_paper_R1/{doc}/{name} differs from "
                               f"{expected[name].relative_to(ROOT)}")
    n = sum(len(flat(d)) for d in DOCS) if VIEW.is_dir() else 0
    return {"pass": not bad, "n_files": n, "n_working": w["n_files"],
            "problems": bad,
            "detail": (f"paper_R1/{{main,esm}} and VERA_paper_R1/{{main,esm}}: "
                       f"{w['n_files']} + {n} files, each byte-identical to "
                       f"its canonical source; nothing extra, nothing missing"
                       if not bad else "; ".join(bad[:6]))}


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    for doc in DOCS:
        f = files(doc)
        c(len(f) > 15, f"{doc}: dependency list is derived, not typed "
                       f"({len(f)})")
        names = [n for _, n in flat(doc)]
        c(all("/" not in n for n in names),
          f"{doc}: every submitted name is flat -- Editorial Manager cannot "
          f"process subfolders")
        c(len(names) == len(set(names)),
          f"{doc}: no two sources flatten onto the same name")
    m = [n for _, n in flat("main")]
    c({"paperY_main.tex", "sn-jnl.cls", "sn-vancouver-num.bst",
       "paperY_main.bbl", "references_Y.bib"} <= set(m),
      "main: class, style, pre-compiled bbl and .bib are in the submission set")
    c("Fig_vera.pdf" in m, "main: the delivered Fig. 1 asset is in the set")
    #: The property the whole restructure rests on: the two layouts differ in
    #: where files sit, not in what they contain.
    for doc in DOCS:
        nested = {Path(rel).name: src for src, rel in files(doc)}
        c(nested == {n: s for s, n in flat(doc)},
          f"{doc}: the nested and flat layouts carry the same bytes under the "
          f"same names")
    r = check()
    c(r["pass"], f"both layouts match their sources: {r['detail']}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--working-only", action="store_true",
                    help="populate paper_R1/{main,esm} and stop")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.check:
        r = check()
        print(json.dumps(r, ensure_ascii=False))
        print(("[OK] " if r["pass"] else "[RED] ") + r["detail"])
        return 0 if r["pass"] else 1
    if a.working_only:
        print(json.dumps(sync_working(), ensure_ascii=False))
        return 0
    r = build()
    print(json.dumps(r, ensure_ascii=False))
    print(f"[ok] {VIEW}/{{main,esm}} : {r['files']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
