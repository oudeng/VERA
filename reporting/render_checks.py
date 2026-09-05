"""Pixel-level checks on the RENDERED page, for the defects no text layer has.

Every gate in this repository until now looked at strings or numbers. The
per-page visual inspection kept finding a class neither can see: ink drawn on
top of other ink, and ink outside the block it belongs in. A table rule drawn
through its caption's descenders is a legal LaTeX box; a caption that overruns
the text block and lands under the folio compiles without a warning; a
figure's cell border crossing its own label is a correct matplotlib call. The
text layer holds every one of those characters, in the right order.

So this module renders the page and looks at it.

    python reporting/render_checks.py --pdf ../paper_R1/paperY_main.pdf
    python reporting/render_checks.py --selftest

Two checks, both calibrated against defects found by human inspection on
2026-08-30 and required to reproduce them:

  rules      every horizontal rule has clear air above and below it, so a
             rule can never be drawn onto a glyph
  block      no page's ink extends below the text block the rest of the
             document establishes, so a caption cannot overrun into the folio

Neither check knows anything about LaTeX. That is the point: they see what a
reader sees.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DPI = 300
#: A rule is a near-solid horizontal run this wide, as a fraction of the ink
#: width of the page. Table rules span a column or the full text width; a long
#: underline or a fraction bar does not come close.
RULE_MIN_FRAC = 0.28
#: Clear air demanded above and below a rule, in points. 2 pt is well under
#: booktabs' own \abovetopsep/\belowrulesep, so a passing table is not made to
#: look different -- it only fails when a glyph is actually touching.
CLEAR_PT = 2.0
#: Ink this far below the document's own text-block bottom is an overrun.
#: Generous, because the block bottom legitimately varies by a line's depth.
#: Air the body must leave above the folio. Pages 2-28 of the main text
#: clear it by ~23 pt; the page whose caption overran cleared 5.5.
FOLIO_CLEAR_PT = 12.0
#: The blank band that separates the folio from the text block. Wider than
#: any inter-line gap inside running text, narrower than the real margin.
FOLIO_GAP_PT = 5.0
PT_PER_IN = 72.0


def _px(pt: float) -> int:
    return max(1, int(round(pt * DPI / PT_PER_IN)))


def render(pdf: Path, out_dir: Path) -> list:
    subprocess.run(["pdftoppm", "-r", str(DPI), "-gray", "-png",
                    str(pdf), str(out_dir / "p")], check=True)
    return sorted(out_dir.glob("p-*.png"))


def _ink(png: Path) -> np.ndarray:
    from PIL import Image
    a = np.asarray(Image.open(png).convert("L"))
    return a < 128


def find_rules(ink: np.ndarray) -> list:
    """(y0, y1, x0, x1) for every horizontal rule, merged over its thickness."""
    h, w = ink.shape
    cols = ink.any(axis=0)
    if not cols.any():
        return []
    span = int(cols.nonzero()[0][-1] - cols.nonzero()[0][0] + 1)
    need = int(RULE_MIN_FRAC * span)
    rows = []
    for y in range(h):
        r = ink[y]
        if r.sum() < need:
            continue
        xs = r.nonzero()[0]
        # one near-solid run, not scattered ink that happens to be dense
        if (xs[-1] - xs[0] + 1) and r[xs[0]:xs[-1] + 1].mean() > 0.9:
            rows.append((y, int(xs[0]), int(xs[-1])))
    out, i = [], 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[j][0] + 1:
            j += 1
        out.append((rows[i][0], rows[j][0],
                    min(r[1] for r in rows[i:j + 1]),
                    max(r[2] for r in rows[i:j + 1])))
        i = j + 1
    return out


def check_rules(ink: np.ndarray) -> list:
    """No rule may have ink in the clear band above or below it."""
    clear = _px(CLEAR_PT)
    bad = []
    rules = find_rules(ink)
    ys = {y for y0, y1, _, _ in rules for y in range(y0, y1 + 1)}
    for y0, y1, x0, x1 in rules:
        for side, band in (("above", range(max(0, y0 - clear), y0)),
                           ("below", range(y1 + 1, min(ink.shape[0], y1 + 1 + clear)))):
            n = 0
            for y in band:
                if y in ys:          # an adjacent rule is not a collision
                    continue
                n += int(ink[y, x0:x1 + 1].sum())
            if n > 0:
                bad.append({"rule_y": int(y0), "side": side, "ink_px": n,
                            "x0": int(x0), "x1": int(x1)})
    return bad


def _row_runs(ink: np.ndarray) -> list:
    """Contiguous bands of rows that carry ink, as (start, end) inclusive."""
    rows = ink.any(axis=1)
    runs, y = [], 0
    h = len(rows)
    while y < h:
        if not rows[y]:
            y += 1
            continue
        j = y
        while j + 1 < h and rows[j + 1]:
            j += 1
        runs.append((y, j))
        y = j + 1
    return runs


def _block_bottom_of(ink: np.ndarray) -> int:
    """The last row of BODY ink on this page, i.e. above the folio.

    The folio is the final band of ink, separated from the block by a gap no
    line of running text ever leaves. If there is no such gap, the page has no
    folio band and its own last ink row is the bottom.
    """
    runs = _row_runs(ink)
    if not runs:
        return 0
    if len(runs) >= 2 and runs[-1][0] - runs[-2][1] > _px(FOLIO_GAP_PT):
        return int(runs[-2][1])
    return int(runs[-1][1])


def block_bottom(inks: list) -> int:
    """The text block the DOCUMENT establishes, not any one page's."""
    b = [_block_bottom_of(ink) for ink in inks if ink.any()]
    return int(np.median(b)) if b else 0


def check_block(ink: np.ndarray, _bottom: int) -> list:
    """Body ink must not reach the folio.

    Not "below the median page bottom": in a single-column supplement with
    tables set in place, the block legitimately ends at different heights on
    different pages, and a median turns that into noise. What is never legal
    is body text crowding or overprinting the page number, which is exactly
    the defect this was built for (the Fig. 3 caption overran its block and
    the folio printed on top of the word "cell").
    """
    runs = _row_runs(ink)
    if len(runs) < 2:
        return []
    folio_top, body_bottom = runs[-1][0], runs[-2][1]
    gap = folio_top - body_bottom
    if gap <= _px(FOLIO_GAP_PT):
        # no folio band at all: the last run IS body text running to the foot
        return [{"body_bottom_px": int(runs[-1][1]), "folio": "absent or merged",
                 "clearance_pt": 0.0}]
    if gap < _px(FOLIO_CLEAR_PT):
        return [{"body_bottom_px": int(body_bottom), "folio_top_px": int(folio_top),
                 "clearance_pt": round(gap * PT_PER_IN / DPI, 1)}]
    return []


def figure_pages(pdf: Path) -> list:
    """Pages carrying a figure, found by the caption the document itself sets.

    A figure's interior is full of short rules -- cell borders, an axis, a
    connector -- and a bar standing on its own axis is not a defect. Those
    interiors are asserted by the generators that draw them (fig_vera.py,
    fig_scoreboard.py measure every string against the box it goes in), so the
    rule check does not second-guess them here. The page list is DERIVED, from
    a caption line "Fig. N", not written down: a figure that moves, moves with
    its caption.
    """
    n = int(re.search(r"Pages:\s*(\d+)",
                      subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                     text=True).stdout).group(1))
    out = []
    for i in range(1, n + 1):
        txt = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i),
                              str(pdf), "-"], capture_output=True,
                             text=True).stdout
        if re.search(r"^Fig\.\s*\d", txt, re.M):
            out.append(i)
    return out


def run(pdf: Path) -> dict:
    figs = figure_pages(pdf)
    with tempfile.TemporaryDirectory() as td:
        pages = render(pdf, Path(td))
        inks = [_ink(p) for p in pages]
        bottom = block_bottom(inks)
        rules, block, absorbed = [], [], 0
        for i, ink in enumerate(inks, 1):
            hits = check_rules(ink)
            if i in figs:
                absorbed += len(hits)          # printed, never silent
            else:
                rules += [{"page": i, **b} for b in hits]
            block += [{"page": i, **b} for b in check_block(ink, bottom)]
    return {"pdf": pdf.name, "pages": len(inks), "document_bottom_px": bottom,
            "figure_pages": figs, "rule_hits_absorbed_on_figure_pages": absorbed,
            "rule_collisions": rules, "block_overruns": block,
            "pass": not rules and not block}


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    h, w = 200, 400
    ink = np.zeros((h, w), bool)
    ink[100:102, 40:360] = True                 # a clean rule
    chk(len(find_rules(ink)) == 1, "a horizontal rule is found")
    chk(check_rules(ink) == [], "a rule with clear air passes")
    ink2 = ink.copy()
    ink2[99, 100:110] = True                    # a descender touching it
    hits = check_rules(ink2)
    chk(len(hits) == 1 and hits[0]["side"] == "above",
        "a glyph touching the rule from above is caught")
    ink3 = ink.copy()
    ink3[102:104, 40:360] = True                # a second rule right below
    chk(check_rules(ink3) == [],
        "an adjacent rule is not mistaken for a collision")
    ink4 = np.zeros((400, w), bool)
    ink4[10:200, 20:380] = True                 # a text block
    ink4[300:310, 190:210] = True               # the folio, a real gap below
    chk(_block_bottom_of(ink4) == 199,
        f"the block bottom is found above the folio ({_block_bottom_of(ink4)})")
    chk(check_block(ink4, 0) == [], "a page that clears its folio passes")
    ink5 = ink4.copy()
    ink5[201:290, 20:380] = True                # a caption overrunning the block
    chk(check_block(ink5, 0) != [], "body ink crowding the folio is caught")
    ink6 = np.zeros((400, w), bool)
    ink6[10:281, 20:380] = True                 # body running down to the folio
    ink6[285:295, 190:210] = True               # folio, all but touching it
    chk(check_block(ink6, 0) != [],
        "a folio the body has run down onto is caught")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max", type=int, default=25)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = run(Path(a.pdf))
    for b in r["rule_collisions"][:a.max]:
        print(f"[RULE ] p{b['page']:>3} y={b['rule_y']} {b['side']}: "
              f"{b['ink_px']} ink px touching the rule")
    for b in r["block_overruns"][:a.max]:
        print(f"[BLOCK] p{b['page']:>3} body ink clears the folio by only "
              f"{b['clearance_pt']} pt")
    print(f"{r['pdf']}: {len(r['rule_collisions'])} rule collisions, "
          f"{len(r['block_overruns'])} block overruns, pass={r['pass']}; "
          f"figure pages {r['figure_pages']} absorbed "
          f"{r['rule_hits_absorbed_on_figure_pages']} in-figure rule hits "
          f"(their generators assert their own geometry)")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
