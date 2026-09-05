"""The R1 cover letter, generated -- not typed.

P6 SS4. The cover letter is the fourth document that states the same facts
(manuscript, response letter, repository, letter), and the only one nobody
re-derives before sending. So it takes its numbers from the same artifacts the
response letter does, and its section pointers from the frozen paperY_main.aux.

    PYTHONHASHSEED=2025 python reporting/cover_letter.py
    PYTHONHASHSEED=2025 python reporting/cover_letter.py --check <file>
    PYTHONHASHSEED=2025 python reporting/cover_letter.py --selftest
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from reporting.response_letter import (values, _resolve_labels,  # noqa: E402
                                       locations)

OUT = ROOT / "VERA_response" / "COVER_LETTER_R1.md"
BODY = Path(__file__).parent / "cover_letter_body.md"

#: Imported, not repeated. They are stated in the manuscript too, and the
#: check below asserts the two agree -- a cover letter naming a different
#: commit than the paper does is the exact failure this project already spent
#: a round fixing.
from reporting.response_letter import (REPO_URL, REPO_TAG,     # noqa: E402
                                       REPO_COMMIT)


def _pages(name: str) -> str:
    """Page count, read from the frozen PDF.

    The cover letter said "(29 pages)" and "(26 pages)" as typed literals.
    They were right, and they were the only numbers in either letter that
    nothing checked -- exactly the class this project derives rather than
    types, and exactly the class that goes stale when a float moves.
    """
    import subprocess
    from experiments.package_layout import paper_file
    out = subprocess.run(["pdfinfo", str(paper_file(name))],
                         capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"cannot read the page count of {name}; the letter "
                           f"will not state one it has not measured")
    return m.group(1)


def _manuscript() -> str:
    from experiments.package_layout import paper_file
    return paper_file("paperY_main.tex").read_text()


def render() -> str:
    body = BODY.read_text()
    vals = dict(values())
    vals.update({"repo_url": REPO_URL, "repo_tag": REPO_TAG,
                 "repo_commit": REPO_COMMIT,
                 "main_pages": _pages("paperY_main.pdf"),
                 "esm_pages": _pages("paperY_ESM.pdf")})
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", body)) - set(vals))
    if missing:
        raise KeyError(f"the cover letter references values with no source: "
                       f"{missing}")
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], body)
    return _resolve_labels(text)


def check(delivered: Path) -> dict:
    text = render()
    have = delivered.read_text() if delivered.exists() else None
    ok = have == text
    return {"pass": ok, "chars": len(text),
            "detail": (f"{delivered.name}: character-identical to what this "
                       f"script renders now ({len(text)} chars)" if ok else
                       f"{delivered.name}: DIFFERS from the rendered letter")}


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    text = render()
    c("{{" not in text, "every reference was substituted")
    tex = _manuscript()
    #: The letter and the paper must name the SAME repository state. They are
    #: written at different times by different processes, which is exactly how
    #: they come to disagree.
    c(REPO_COMMIT in tex,
      f"the manuscript cites the same commit this letter does ({REPO_COMMIT})")
    c("oudeng/VERA" in tex, "the manuscript cites the same repository")
    c(REPO_TAG in tex, f"the manuscript cites the same release ({REPO_TAG})")
    #: No typed decimals outside quotations, same rule as the response letter.
    stray = re.findall(r"(?<![\w.=/#§v-])\d+\.\d+(?![\w.])",
                       "\n".join(l for l in BODY.read_text().splitlines()
                                 if not l.lstrip().startswith(">")))
    c(not stray, f"no decimal is typed into the source: {stray[:6]}")
    loc = locations()
    c(len(loc) > 20, f"the frozen manuscript's labels are readable ({len(loc)})")
    c("no competing interests" in text.lower(),
      "the competing-interests restatement is present")
    #: The generative-AI section was removed on the first author's
    #: instruction (2026-09-05): the manuscript's own declaration carries it,
    #: and the cover letter was repeating it. Asserted so it cannot creep back.
    c("generative AI" not in text,
      "the generative-AI section is not repeated in the cover letter")
    c("\\bmhead{Declaration of generative AI" in _manuscript(),
      "the manuscript still carries the declaration itself")
    c("Qun Jin" in text, "the editorial-role disclosure names the co-author")
    c(f"({_pages('paperY_main.pdf')} pages)" in text
      and f"({_pages('paperY_ESM.pdf')} pages)" in text,
      "the page counts are the frozen PDFs' own, not typed")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.check:
        r = check(Path(a.check))
        print(("[OK] " if r["pass"] else "[RED] ") + r["detail"])
        return 0 if r["pass"] else 1
    text = render()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"[ok] wrote {OUT} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
