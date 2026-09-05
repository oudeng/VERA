"""Inventory of AI-assisted commits, by category.

P5R-L SS4.2: the AI declaration in the manuscript (file C-prime) points at the
code repository for "a fuller account of the AI-assisted workflow". This makes
that account something that exists and can be checked, rather than a promise.
It is NOT for the manuscript: it is the material for the public repository's
AI section, and the thing to answer with if a reader asks what "assist with
code development, execution of the specified analyses, and manuscript
drafting" covered.

Commits are categorised by what they touched and what their message says, and
every category prints a representative example so the categorisation can be
argued with rather than believed.

    env PYTHONHASHSEED=2025 python experiments/ai_usage_inventory.py [--selftest]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

REPOS = ["code_SNI", "paper_R1", "VERA_GitHub"]
MARKER = "Co-Authored-By: Claude"
OUT = ROOT / "reports" / "ai_usage_inventory.md"

#: (category, path patterns, message patterns). A commit falls in every
#: category it matches, because one commit routinely does two things; the
#: totals therefore exceed the commit count and that is stated, not hidden.
CATEGORIES = [
    ("statistical rule drafting",
     [r"^docs/T\d.*rules?\.md$", r"^docs/T\d.*decision_rule\.md$",
      r"^docs/T\d.*precheck\.md$"],
     [r"\brule[s]?\b.*commit", r"zero-artifact", r"prospectively specified"]),
    ("experiment design",
     [r"^experiments/.*\.py$", r"^configs/.*\.ya?ml$"],
     [r"\bdesign\b", r"\bprotocol\b", r"\bpre-?check\b"]),
    ("analysis code",
     [r"^stats/", r"^evaluation/", r"^common/", r"^sni/", r"^baselines/",
      r"^missingness/", r"^data_layer/", r"^experiments/t\d"],
     [r"\bimputer\b", r"\bmetric", r"\bestimand\b"]),
    ("artifact generation",
     [r"^reporting/.*\.py$"],
     [r"\bgenerat", r"\btable\b", r"\bfigure\b", r"\bFig\."]),
    ("interpretation supplement",
     [r"^docs/.*interpretation.*\.md$", r"^docs/adjudications\.md$"],
     [r"\binterpret", r"\badjudicat"]),
    ("discussion blueprint",
     [r"^docs/D\d.*blueprint.*\.md$"],
     [r"\bblueprint\b", r"\bdiscussion\b"]),
    ("manuscript drafting",
     [r"\.tex$", r"^paper", r"references_Y\.bib$"],
     [r"\babstract\b", r"\bmanuscript\b", r"\bsection\b"]),
    ("text editing",
     [],
     [r"\bwording\b", r"\bnarrow", r"\breword", r"\bneutralis", r"\btypo\b",
      r"\bphrasing\b", r"\bterminolog"]),
    ("audit and verification",
     [r"^experiments/.*audit.*\.py$", r"^experiments/.*census.*\.py$",
      r"^reporting/(facts_gate|package_gates|status_table)\.py$"],
     [r"\baudit\b", r"\bgate\b", r"\bverif", r"\bcensus\b"]),
]


def _log(repo: str) -> list:
    """One record per commit: sha, date, subject, body, files.

    Commit bodies contain blank lines and the file list follows the body, so
    a naive split loses commits. Records are delimited by a control character
    git will never emit, and the fields inside each record by another.
    """
    d = ROOT / repo
    if not (d / ".git").exists():
        return []
    REC, FLD = "\x1e", "\x1f"
    raw = subprocess.run(
        ["git", "log", f"--format={REC}%H{FLD}%ad{FLD}%s{FLD}%b{FLD}",
         "--date=short", "--name-only"],
        cwd=d, capture_output=True, text=True).stdout
    out = []
    for chunk in raw.split(REC):
        if not chunk.strip():
            continue
        parts = chunk.split(FLD)
        if len(parts) < 4:
            continue
        h, date, subj, body = parts[0], parts[1], parts[2], parts[3]
        files = [f.strip() for f in (parts[4] if len(parts) > 4 else "")
                 .split("\n") if f.strip()]
        out.append({"repo": repo, "sha": h[:8], "date": date,
                    "subject": subj, "body": body,
                    "ai": MARKER in (body + subj), "files": files})
    return out


def categorise(c: dict) -> list:
    hits = []
    text = (c["subject"] + " " + c["body"]).lower()
    for name, paths, msgs in CATEGORIES:
        if any(re.search(p, f) for p in paths for f in c["files"]) or \
           any(re.search(m, text) for m in msgs):
            hits.append(name)
    return hits or ["uncategorised"]


def build() -> dict:
    commits = [c for r in REPOS for c in _log(r)]
    ai = [c for c in commits if c["ai"]]
    per = {}
    for c in ai:
        for cat in categorise(c):
            per.setdefault(cat, []).append(c)
    lines = [
        "# AI usage inventory",
        "",
        f"**Generated** by `experiments/ai_usage_inventory.py` from the "
        f"version histories of {', '.join(REPOS)}. Not part of the "
        f"manuscript: this is the material behind the AI declaration's "
        f"pointer to the code repository, and the answer to give if a reader "
        f"asks what the declaration covers.",
        "",
        "## What is counted",
        "",
        f"A commit counts as AI-assisted when its message carries the "
        f"`{MARKER}` trailer. That trailer was written by the tooling on "
        f"every commit made with model assistance, so the count is a record "
        f"rather than a reconstruction. It says nothing about how much of any "
        f"one commit was model-written: the authors directed and reviewed all "
        f"of it and are responsible for all of it.",
        "",
        "| repository | commits | AI-assisted | share |",
        "|---|---|---|---|",
    ]
    for r in REPOS:
        n = sum(1 for c in commits if c["repo"] == r)
        k = sum(1 for c in ai if c["repo"] == r)
        lines.append(f"| `{r}` | {n} | {k} | "
                     f"{(k / n * 100 if n else 0):.0f}% |")
    lines += [
        f"| **total** | **{len(commits)}** | **{len(ai)}** | "
        f"**{(len(ai) / len(commits) * 100 if commits else 0):.0f}%** |",
        "",
        "## By category",
        "",
        "A commit routinely does two things, so it is counted in every "
        "category it matches and the category totals exceed the commit "
        "count. The categorisation is by the files a commit touched and the "
        "words its message used; each row carries an example so the "
        "categorisation can be argued with.",
        "",
        "| category | AI-assisted commits | example |",
        "|---|---|---|",
    ]
    for name, _, _ in CATEGORIES + [("uncategorised", [], [])]:
        rows = per.get(name, [])
        if not rows:
            continue
        ex = rows[0]
        lines.append(f"| {name} | {len(rows)} | `{ex['sha']}` "
                     f"{ex['subject'][:78]} |")
    lines += [
        "",
        "## What the model did not do",
        "",
        "No number in this paper was produced by a language model. Every "
        "reported quantity comes from a generator that reads a stored "
        "artifact, and the manuscript contains no hand-typed result: the "
        "single-facts gate refuses a compile whose sources disagree with "
        "`results/T5_stats/t_final.json`. The model wrote and revised the "
        "code that computes those quantities, and the authors reviewed it.",
        "",
        f"*Regenerate with* `python experiments/ai_usage_inventory.py`.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    return {"n_commits": len(commits), "n_ai": len(ai),
            "categories": {k: len(v) for k, v in per.items()},
            "path": str(OUT)}


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    commits = [c for r in REPOS for c in _log(r)]
    check(len(commits) > 100, f"the histories were read ({len(commits)} commits)")
    check(any(c["files"] for c in commits), "commits carry their file lists")
    n_ai = sum(1 for c in commits if c["ai"])
    check(0 < n_ai < len(commits),
          f"the marker discriminates ({n_ai} of {len(commits)} carry it)")
    fake = {"subject": "reword the abstract", "body": "", "files": []}
    check("text editing" in categorise(fake),
          "a wording-only commit lands in text editing")
    fake2 = {"subject": "x", "body": "", "files": ["reporting/fig_vera.py"]}
    check("artifact generation" in categorise(fake2),
          "a figure-generator commit lands in artifact generation")
    fake3 = {"subject": "x", "body": "", "files": ["zzz/nothing.txt"]}
    check(categorise(fake3) == ["uncategorised"],
          "an unmatched commit is called uncategorised, not forced into a bin")
    r = build()
    t = OUT.read_text()
    check(str(r["n_ai"]) in t, "the totals appear in the document")
    check("No number in this paper was produced by a language model" in t,
          "the document states the boundary the declaration implies")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    r = build()
    print(f"commits {r['n_commits']}, AI-assisted {r['n_ai']}")
    for k, v in sorted(r["categories"].items(), key=lambda x: -x[1]):
        print(f"  {k:<28} {v}")
    print(f"\nwrote {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
