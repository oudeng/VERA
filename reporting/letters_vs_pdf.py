"""P6 SS5: every pointer and number in the letters, against the frozen PDFs.

The letters go to the editor beside the manuscript. Three documents stating
the same facts is two chances to disagree, and the letters are the pair nobody
recompiles. So this reads them back and checks, mechanically:

  * every "Sect. N (p. P)" names a label the frozen paperY_main.aux defines,
    at that number and that page;
  * every value the generators supplied appears in the manuscript or the
    Online Resource, as printed;
  * every path the letters cite as evidence EXISTS in the published
    repository. The first version of these letters pointed a reviewer at
    eleven paths, and essentially none of them resolved: they named the
    internal review package's layout and this workspace's results/ tree,
    neither of which a reviewer has. A pointer that cannot be followed is
    worse than no pointer -- it reads as evidence and delivers none.

Two equivalences are DECLARED rather than assumed, because a checker that
silently accepted them would accept anything:

  * "+0.158627 unrounded" is the full-precision form of a figure the paper
    prints rounded. The letter states both on purpose -- it tells the reviewer
    the rounded number hides nothing -- so the unrounded form is matched
    against the rounded one it is declared to expand.
  * "4/5" and "4 of 5" are the same count in two notations.

    PYTHONHASHSEED=2025 python reporting/letters_vs_pdf.py
    PYTHONHASHSEED=2025 python reporting/letters_vs_pdf.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from reporting.response_letter import locations, values     # noqa: E402
from experiments.package_layout import paper_file           # noqa: E402

#: The markup-slot copy is a SUBMITTED file, so it is checked like the
#: others. It should be identical to the response letter apart from its
#: first-page note -- and if it ever is not, this is where that shows.
LETTERS = ("RESPONSE_TO_REVIEWERS_R1.md", "COVER_LETTER_R1.md",
           "RESPONSE_TO_REVIEWERS_R1_markup_slot.md")

#: key -> the key whose PRINTED form it expands. Declared, with the reason.
UNROUNDED = {"fair_T_full": "fair_T"}

#: Paths in the letters are relative to the published repository. Anything
#: else -- and in particular anything under results/, reports/ or code/, the
#: three trees that are ours and not the reviewer's -- is a broken pointer.
PUBLIC = ROOT / "VERA_GitHub"
NEVER = ("results/", "reports/", "code/", "ir_staging/", "paper_R1/",
         "internal_review/")


def pointers(text: str) -> list[str]:
    """The path-shaped literals in a letter: `dir/name.ext`."""
    return sorted({m for m in re.findall(r"`([\w./-]+/[\w./-]+\.\w+)`", text)})


def reachable() -> dict:
    """Every cited path, against the tree the reviewer can actually clone."""
    bad, n = [], 0
    for name in LETTERS:
        f = ROOT / "VERA_response" / name
        if not f.exists():
            continue
        for rel in pointers(f.read_text()):
            n += 1
            if rel.startswith(NEVER):
                bad.append(f"{name}: `{rel}` names one of our own working "
                           f"trees, which the reviewer does not have")
            elif not (PUBLIC / rel).exists():
                bad.append(f"{name}: `{rel}` is not in the published "
                           f"repository")
    return {"pointers": n, "problems": bad, "pass": not bad}


def _text(pdf: Path) -> str:
    t = subprocess.run(["pdftotext", str(pdf), "-"],
                       capture_output=True, text=True).stdout
    #: pdftotext writes the math minus as U+2212 and TeX's thin space as a
    #: narrow no-break space. The letters are ASCII. Normalizing both sides is
    #: the difference between a check and a source of spurious disagreements -- the
    #: first version of this reported every signed effect as a disagreement.
    for a, b in (("−", "-"), ("–", "-"), (" ", ""),
                 (" ", " "), (" ", "")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t)


def run() -> dict:
    main = _text(paper_file("paperY_main.pdf"))
    esm = _text(paper_file("paperY_ESM.pdf"))
    loc, vals = locations(), values()
    printed = {(v[0], v[1]) for v in loc.values()}
    bad, n_ptr, n_num, seen = [], 0, 0, {"main": 0, "esm": 0, "expanded": 0}

    for name in LETTERS:
        f = ROOT / "VERA_response" / name
        if not f.exists():
            bad.append(f"{name}: not delivered")
            continue
        t = f.read_text()
        for m in re.finditer(r"(?:Sect\.|Table|Fig\.) ([\d.]+) \(p\. (\d+)\)", t):
            n_ptr += 1
            if (m.group(1), m.group(2)) not in printed:
                bad.append(f"{name}: '{m.group(0)}' names no label the "
                           f"manuscript defines at that number and page")
        for k, v in vals.items():
            if v not in t:
                continue
            n_num += 1
            probe = v.lstrip("+")
            alt = probe.replace("/", " of ")
            if probe in main or alt in main:
                seen["main"] += 1
            elif probe in esm or alt in esm:
                seen["esm"] += 1
            elif k in UNROUNDED and vals[UNROUNDED[k]].lstrip("+") in (main + esm):
                seen["expanded"] += 1
            else:
                bad.append(f"{name}: states {k} = {v}, which appears in "
                           f"neither the manuscript nor the Online Resource")
    r = reachable()
    bad += r["problems"]
    return {"pointers": n_ptr, "numbers": n_num, "paths": r["pointers"],
            "located": seen, "problems": bad, "pass": not bad}


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    r = run()
    c(r["pointers"] >= 10, f"the letters carry pointers to check ({r['pointers']})")
    c(r["numbers"] >= 25, f"the letters carry numbers to check ({r['numbers']})")
    c(r["pass"], f"every pointer and number agrees with the frozen PDFs: "
                 f"{r['problems'][:3]}")
    c(r["paths"] >= 10, f"the letters cite evidence paths ({r['paths']})")
    #: the reachability check must be able to fail
    c(not (PUBLIC / "results/T5_stats/t_final.json").exists(),
      "a workspace-only path is genuinely absent from the published tree")
    #: The check must be able to FAIL, or it is decoration.
    m = _text(paper_file("paperY_main.pdf"))
    c("0.159" in m, "the fixture value is genuinely in the PDF")
    c("0.15900001" not in m, "a value that is NOT in the PDF is not found")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "problems"},
                     ensure_ascii=False))
    for p in r["problems"]:
        print(f"  [RED] {p}")
    print(("[OK] " if r["pass"] else "[RED] ")
          + f"{r['pointers']} pointers, {r['numbers']} numbers and "
            f"{r['paths']} evidence paths checked against the frozen "
            f"manuscript, the Online Resource and the published repository")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
