"""Which inspection record covers each page, and the pixel proof that it does.

Ninth-round adjudication (2026-08-31), condition on accepting an incremental
visual pass: "no page went unseen" must be a traceable statement, not a verbal
guarantee. So for every page of the delivered build this records

  * the SHA-256 of that page rendered at 150 dpi from the DELIVERED PDF, and
  * the inspection pass that read it, together with the SHA-256 of the same
    page in the build that pass was reading.

When the two hashes are equal, the page a human read and the page being
delivered are the same pixels, and the covering record carries forward. When
they differ the page needs re-reading, and this refuses to emit a table that
would claim otherwise.

The passes are given oldest-first; a page is attributed to the LAST pass whose
build differs from the one before it -- that is the pass that had to look at
it, because that is when it changed.

    PYTHONHASHSEED=2025 python reporting/inspection_coverage.py \
        --render-root DIR --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

ROOT = CODE_ROOT.parent
PAPER = ROOT / "paper_R1"

#: Oldest first. Each entry: the render directory holding that pass's build,
#: and what to call the pass in the table.
PASSES = [
    ("final",  "round-8 freeze, full 55-page pass"),
    ("s3",     "P5R-S incremental (18 pages)"),
    ("final9", "P5R-T incremental (13 pages)"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def render(pdf: Path, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-r", "150", "-png", str(pdf),
                    str(out / stem)], check=True)


def coverage(render_root: Path, delivered: Path) -> list:
    rows = []
    for page in sorted(delivered.glob("*.png")):
        name = page.name
        cur = _sha(page)
        # the last pass at which this page's pixels changed
        covering, prev = PASSES[0], None
        for d, label in PASSES:
            f = render_root / d / name
            if not f.exists():
                continue
            h = _sha(f)
            if prev is not None and h != prev:
                covering = (d, label)
            prev = h
        d, label = covering
        f = render_root / d / name
        read_sha = _sha(f) if f.exists() else None
        rows.append({"page": name.replace(".png", ""),
                     "covered_by": label,
                     "read_build_sha256": read_sha,
                     "delivered_sha256": cur,
                     "carries_forward": read_sha == cur})
    return rows


def table(rows: list) -> str:
    bad = [r for r in rows if not r["carries_forward"]]
    if bad:
        raise SystemExit(
            "REFUSING to emit a coverage table: these pages differ from the "
            f"build their inspection record was made on: {[r['page'] for r in bad]}. "
            "Re-read them, or re-run after the covering pass.")
    head = (
        "| page | covered by | page SHA-256 (150 dpi, delivered build) |\n"
        "|---|---|---|\n")
    body = "".join(f"| {r['page'].replace('main-', 'main p.').replace('esm-', 'ESM p.').replace('p.0', 'p.')} "
                   f"| {r['covered_by']} | `{r['delivered_sha256'][:32]}…` |\n"
                   for r in rows)
    return head + body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    root = Path(a.render_root)
    delivered = root / "delivered"
    render(paper_file("paperY_main.pdf", PAPER), delivered, "main")
    render(paper_file("paperY_ESM.pdf", PAPER), delivered, "esm")
    rows = coverage(root, delivered)
    Path(a.out).write_text(table(rows))
    n = {}
    for r in rows:
        n[r["covered_by"]] = n.get(r["covered_by"], 0) + 1
    print(json.dumps({"pages": len(rows), "by_pass": n,
                      "all_carry_forward": all(r["carries_forward"] for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
