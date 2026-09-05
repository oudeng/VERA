"""SUPERSEDED-BY-ASSET (2026-09-01).

Fig. 1 is now a delivered design asset -- paper_R1/Fig/Fig_vera.pdf, exported
from Fig_vera.pptx -- registered in docs/figure_assets.json. This generator no
longer produces the figure the manuscript prints.

It stays in the repository on purpose, and not as a courtesy: it is the
HISTORICAL CONTENT TRUTH. The asset was accepted by comparing its text against
the filled brief, and the brief was filled from what this script drew. If a
future question asks what Fig. 1 was supposed to say, the answer is here, at
the commit registered as content_source_commit -- not in a PowerPoint file
whose edit history nobody keeps.

Do not run it to refresh the manuscript figure; it would write a file the
packaging gates now reject on identity. Run it to see what the asset is
supposed to contain.

Fig. 1 -- the VERA protocol schematic, five-band simplification of the
review-SS9 structure (adjudication carried in P5R-F; the date this
file once carried, 2026-08-31, was a copied instruction-file date and
is not a date any clock in this project showed): the four core revisions are
kept and now ASSERTED by the selftest -- (i) the declared-claim-and-scope
band, (ii) the four-way readout split, (iii) zero banned variants from
docs/terminology_registry.json (plus the legacy "one ruler" / "before
data" wordings), and
(iv) the circularity-exclusion sentence living in the CAPTION (main tex),
not duplicated in the figure. Eligibility is a thin arrow label; axis
boxes carry name + a 3--5-word phrase (full operationalization lives in
Table 1); the estimates and rules boxes are merged. All hyphens are plain
text (no mathtext minus glyphs).

    PYTHONHASHSEED=2025 python reporting/fig_vera.py [--selftest]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(CODE_ROOT))
#: paper_R1 is main/ + esm/ since P7-A SS2; a package's assembled
#: view is still flat. One resolver answers for all three layouts.
from experiments.package_layout import paper_file  # noqa: E402

sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402

OUT = CODE_ROOT / "reporting" / "out" / "Fig_vera.pdf"
MAIN_TEX = paper_file("paperY_main.tex")

# Final axis names and order (review SS5; atomic with the paper-wide sync)
AXES = [
    ("1. Structural recovery", "known generating adjacency",
     "Synthetic generating graph"),
    ("2. Stability alignment", "repeatability vs host variation",
     "Repeated runs + host behavior"),
    ("3. Behavioral faithfulness", "same-host ablation reference",
     "Same-host intervention matrix"),
    ("4. Leakage-risk discrimination",
     "controlled proxy challenges and discrepancy controls; "
     "empirical null rates reported",
     "Proxy/null challenge sets"),
    ("5. Resource cost", "marginal & total, separate",
     "Runtime/resource logs"),
]

READOUTS = [
    ("Target artifact", "SNI reliance matrix $\\mathbf{D}$", "#fddbc7"),
    ("Prespecified comparator", "TAP$_0$: training-free, marginal",
     "#d1e5f0"),
    ("Same-host positive control", "permutation readout", "#d9f0d3"),
    ("Alternative-host readouts", "MF importance / SHAP / permutation",
     "#e6e6fa"),
]

ELIGIBILITY = "common representation & row set; circularity exclusions"


#: geometry of the last rendered axis rows, for the selftest to check
_MEASURED: list = []
_READOUT_FIT: list = []


#: The stroke a rounded box spends outside the rectangle it is given.
BOX_PAD = 0.06
#: Every box's outer extent, in data coordinates, recorded as it is
#: drawn so the selftest can prove none of them is clipped.
_BOXES: list = []
#: The axes limits the last build used, for the same selftest.
_LIMS: dict = {"x": (0.0, 0.0), "y": (0.0, 0.0)}


def _extent(fig, ax, t):
    """Text bbox in data coordinates."""
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    return bb.transformed(ax.transData.inverted())


#: Clearance a glyph must keep from the stroke that bounds it, in data units
#: (~44.6 pt to the unit here). matplotlib's text extent under-reports
#: descenders by about a point, so a fit measured to the hairline renders as a
#: border drawn through the tail of a "p". Row 4 cleared by 1.65 pt and did
#: exactly that.
STROKE_PAD = 0.075


def _fit_axis_rows(fig, ax, rows, min_size=5.2):
    """Make each axis row's description fit its box without touching the
    axis name above it, and record the measured result.

    A terminology gate reads strings, not pixels: it will happily pass a
    figure whose corrected wording is drawn on top of its own title, which
    is exactly what happened when the leakage row took the review's given
    (long) wording. So the description is first anchored strictly below the
    name, then shrunk while it is too wide, and finally wrapped to two
    lines if a single line cannot fit at a legible size.
    """
    import textwrap
    for r in rows:
        avail_w = (r["x1"] - r["x0"]) - 0.34
        name_bb = _extent(fig, ax, r["name"])
        r["sub"].set_y(name_bb.y0 - 0.015)
        # 1) shrink a single line while it overflows the box width
        while r["sub"].get_fontsize() > min_size:
            if _extent(fig, ax, r["sub"]).width <= avail_w:
                break
            r["sub"].set_fontsize(r["sub"].get_fontsize() - 0.15)
        # 2) still too wide at the legibility floor -> wrap to two lines
        bb = _extent(fig, ax, r["sub"])
        if bb.width > avail_w:
            text = " ".join(r["sub"].get_text().split())
            per_unit = len(text) / max(bb.width, 1e-6)
            budget = max(12, int(avail_w * per_unit * 0.97))
            r["sub"].set_text(textwrap.fill(text, budget))
            r["sub"].set_linespacing(0.92)
            # a wrapped row must also fit VERTICALLY inside its box
            for _ in range(20):
                bb2 = _extent(fig, ax, r["sub"])
                if (bb2.width <= avail_w
                        and bb2.y0 >= r["y"] - 0.335 + STROKE_PAD
                        or r["sub"].get_fontsize() <= 4.6):
                    break
                r["sub"].set_fontsize(r["sub"].get_fontsize() - 0.15)
                r["sub"].set_y(_extent(fig, ax, r["name"]).y0 - 0.010)
        final = _extent(fig, ax, r["sub"])
        name_bb = _extent(fig, ax, r["name"])
        _MEASURED.append({
            "row": r["y"],
            "overlaps_title": bool(final.y1 > name_bb.y0 + 0.005),
            "fits_box": bool(final.width <= avail_w + 1e-9
                             and final.y0 >= r["y"] - 0.335 + STROKE_PAD),
            "width": round(float(final.width), 3),
            "y0": round(float(final.y0), 3),
            "floor": round(float(r["y"] - 0.335 + STROKE_PAD), 3),
            "avail": round(float(avail_w), 3),
            "n_lines": r["sub"].get_text().count("\n") + 1,
            "fontsize": round(float(r["sub"].get_fontsize()), 2)})


def build(out_path: Path = OUT) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _BOXES.clear()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(-BOX_PAD, 10 + BOX_PAD)
    # The bottom band sits at y = 0 and its rounded box adds BOX_PAD of stroke
    # BELOW that, which the axes clipped: the last box shipped with no bottom
    # edge (eighth-round freeze inspection, main p.6). The limits now carry the
    # pad the boxstyle spends, and _BOXES + the selftest below make the clip a
    # measured condition rather than something only a reader can see.
    ax.set_ylim(-BOX_PAD, 9.6)
    _LIMS["x"], _LIMS["y"] = ax.get_xlim(), ax.get_ylim()
    ax.axis("off")

    def box(x, y, w, h, fc, ec="#333333"):
        _BOXES.append((x - BOX_PAD, y - BOX_PAD, x + w + BOX_PAD, y + h + BOX_PAD))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad={BOX_PAD}",
                                    fc=fc, ec=ec, lw=0.8))

    def arrow(x, y0, y1):
        """A connector whose HEAD is sized to its own run.

        At a fixed mutation_scale the head is about 0.15 data units tall, so
        the two short connectors at the foot of the figure were all head and
        no shaft -- they read as a stray triangle sitting in a box rule. The
        head now takes at most a third of the run, so every connector shows
        a shaft.
        """
        run = abs(y0 - y1)
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                     mutation_scale=min(10, 20 * run), lw=0.9,
                                     color="#333333"))

    # Band 1: declared claim and scope (title + one small line)
    box(1.1, 8.55, 7.8, 0.82, "#fff7e6")
    ax.text(5.0, 9.14, "Declared audit claim and evaluation scope",
            ha="center", va="center", fontsize=8.6, fontweight="bold")
    ax.text(5.0, 8.78, "host, task, target-source unit, intended "
                       "semantics", ha="center", va="center",
            fontsize=7.2, color="#333333")
    arrow(5.0, 8.50, 8.13)

    # Band 2: four readout kinds, side by side (three-way split kept)
    box(0.30, 6.55, 9.4, 1.42, "#fbfbfb", ec="#999999")
    ax.text(5.0, 7.78, "Audit readouts under evaluation", ha="center",
            fontsize=8.2, fontweight="bold")
    # Band 2's four labels were drawn at a fixed size and never measured, so
    # the longest of them ("MF importance / SHAP / permutation") was printed
    # WIDER than its box: the border stroke ran through the first M and the
    # last n. Nothing could see it -- the string is correct, the box is
    # correct, and only the two together are wrong. Measure, shrink, assert.
    _READOUT_FIT.clear()
    xw = [(0.5, 2.1), (2.75, 2.2), (5.1, 2.15), (7.4, 2.15)]
    for (x, w), (t1, t2, fc) in zip(xw, READOUTS):
        box(x, 6.72, w, 0.86, fc)
        avail = w - 2 * STROKE_PAD
        for txt, yy, sz, bold in ((t1, 7.34, 6.6, "bold"),
                                  (t2, 6.98, 6.2, "normal")):
            t = ax.text(x + w / 2, yy, txt, ha="center", va="center",
                        fontsize=sz, fontweight=bold)
            # shrink only as far as a reader can still follow, then WRAP:
            # a label driven to 4.5 pt to fit one line is not a fix.
            while (t.get_fontsize() > 5.6
                   and _extent(fig, ax, t).width > avail):
                t.set_fontsize(t.get_fontsize() - 0.1)
            e = _extent(fig, ax, t)
            if e.width > avail and " " in txt:
                import textwrap
                per_unit = len(txt) / max(e.width, 1e-6)
                t.set_text(textwrap.fill(txt, max(8, int(avail * per_unit))))
                t.set_linespacing(1.15)
                while (t.get_fontsize() > 4.8
                       and _extent(fig, ax, t).width > avail):
                    t.set_fontsize(t.get_fontsize() - 0.1)
            e = _extent(fig, ax, t)
            _READOUT_FIT.append({"text": txt, "width": round(float(e.width), 3),
                                 "avail": round(float(avail), 3),
                                 "fits": bool(e.width <= avail),
                                 "fontsize": round(float(t.get_fontsize()), 2)})

    # Eligibility: thin arrow label between bands 2 and 3
    arrow(5.0, 6.50, 5.83)
    ax.text(5.18, 6.16, ELIGIBILITY, ha="left", va="center",
            fontsize=6.6, style="italic", color="#555555")

    # Band 3: five axes (name + short phrase) with side reference labels
    # 2.20/3.48, not 2.38/3.30: axis row 4 wraps to two lines and at the
    # old pitch its second line sat on the box's bottom stroke even at the
    # font floor. The remedy is height, never a smaller glyph.
    box(0.30, 2.00, 9.4, 3.68, "#fbfbfb", ec="#999999")
    ax.text(5.0, 5.44,
            "Five complementary evaluation axes with prespecified "
            "applicability", ha="center", fontsize=8.2,
            fontweight="bold")
    # The axis rows carry text of very different lengths (the leakage row
    # takes the third review's given wording verbatim, which is long). A
    # fixed font size silently overstrikes the title when a line wraps, and
    # a text-matching gate cannot see that -- so the size is chosen from the
    # MEASURED width and the sub is anchored below the title, never centered
    # through it. _selftest asserts the result geometrically.
    _MEASURED.clear()
    axis_rows = []
    for i, (name, sub, ref) in enumerate(AXES):
        y = 4.98 - i * 0.655
        box(0.55, y - 0.335, 4.9, 0.65, "#f2f2f2")
        t_name = ax.text(0.75, y + 0.13, name, ha="left", va="center",
                         fontsize=6.9, fontweight="bold")
        t_sub = ax.text(0.75, y - 0.02, sub, ha="left", va="top",
                        fontsize=6.1, color="#333333")
        axis_rows.append({"y": y, "name": t_name, "sub": t_sub,
                          "x0": 0.55, "x1": 0.55 + 4.9})
        # A DRAWN arrow, not the two characters "<" and "-". Set as text it
        # landed in the PDF's text layer as literal ASCII beside four properly
        # drawn arrows elsewhere in the same figure, and read as markup that
        # had failed to convert.
        ax.annotate("", xy=(5.68, y - 0.03), xytext=(6.06, y - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color="#888888",
                                    lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(6.15, y - 0.03, ref, fontsize=6.3, ha="left", va="center",
                color="#555555", style="italic")
    _fit_axis_rows(fig, ax, axis_rows)
    ax.text(6.15, 5.12, "axis-specific reference evidence", fontsize=6.4,
            ha="left", color="#555555", style="italic")
    arrow(5.0, 1.98, 1.72)

    # Band 4: merged evidence-and-rules box (two lines)
    box(1.1, 0.94, 7.8, 0.76, "#f0f6ef")
    ax.text(5.0, 1.50, "Axis-wise evidence: effect sizes, "
                       "axis-appropriate uncertainty", ha="center", va="center",
            fontsize=7.4, fontweight="bold")
    ax.text(5.0, 1.18, "prespecified non-compensatory rules where "
                       "defined; no composite score", ha="center",
            va="center", fontsize=7.0)
    arrow(5.0, 0.92, 0.72)

    # Band 5: output (circularity note lives in the caption, not here)
    box(1.1, 0.00, 7.8, 0.70, "#f4e8f7")
    ax.text(5.0, 0.51, "Axis-wise evidence profile (the formal output)",
            ha="center", va="center", fontsize=8.0, fontweight="bold")
    ax.text(5.0, 0.20, "no cross-axis verdict rule is committed in advance; "
                       "any overall reading is a post-hoc synthesis",
            ha="center", va="center", fontsize=6.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight",
                metadata={"CreationDate": None,
                          "Subject": f"generator: reporting/fig_vera.py "
                                     f"(five-band schematic; no data); "
                                     f"commit: "
                                     f"{runconfig.git_commit()}"})
    plt.close(fig)
    return out_path


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    src = Path(__file__).read_text()
    body = src[src.index('"""', src.index('"""') + 3) + 3:
               src.index("def _selftest")]
    # (iii) banned variants (registry-driven), zero
    from reporting.termmap import banned_variants
    flat = body.replace("-", " ").lower()
    for bad in banned_variants() + ["one ruler", "before data"]:
        check(bad.replace("-", " ").lower() not in flat,
              f"banned variant absent from diagram source: {bad!r}")
    # (i) claim-and-scope band present
    check("Declared audit claim and evaluation scope" in body,
          "core revision 1: claim-and-scope band present")
    # (ii) four-way readout split present
    check(len(READOUTS) == 4 and
          [r[0] for r in READOUTS] == ["Target artifact",
                                       "Prespecified comparator",
                                       "Same-host positive control",
                                       "Alternative-host readouts"],
          "core revision 2: four readout kinds, split kept")
    check("circularity exclusions" in ELIGIBILITY,
          "eligibility strip carries circularity exclusions")
    # (iv) circularity sentence in the CAPTION, not duplicated in-figure
    cap = MAIN_TEX.read_text()
    cap = cap[cap.index("\\caption{The \\VERA{} evaluation protocol"):]
    cap = " ".join(cap[:cap.index("\\label{fig:vera}")].split())
    check("defines the behavioral reference" in cap,
          "core revision 4: circularity sentence verbatim in the caption")
    check("association or model reliance" in cap
          and "causal effect" not in cap,
          "semantics enumeration moved to the caption")
    check("defines the behavioral reference" not in body,
          "no in-figure duplication of the circularity sentence")
    # glyphs: no mathtext-minus constructions, no unicode minus
    check("\\!-\\!" not in body and "−" not in body,
          "hyphens are plain text (no mathtext/unicode minus)")
    # RENDERED geometry: a terminology gate reads strings, not pixels, and
    # will pass a figure whose corrected wording is drawn over its own
    # title. This measures what was actually produced.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        build(Path(td) / "probe.pdf")
    check(len(_MEASURED) == len(AXES),
          "every axis row was measured after rendering")
    for i, g in enumerate(_MEASURED):
        check(not g["overlaps_title"],
              f"axis row {i + 1} description clears its title "
              f"(fontsize {g['fontsize']})")
        check(g["fits_box"],
              f"axis row {i + 1} description fits its box "
              f"({g['width']} <= {g['avail']})")
    check(len(_READOUT_FIT) == 2 * len(READOUTS),
          f"every band-2 label was measured ({len(_READOUT_FIT)})")
    for g in _READOUT_FIT:
        check(g["fits"], f"band-2 label fits its box: {g['text'][:34]!r} "
                         f"({g['width']} <= {g['avail']}, {g['fontsize']} pt)")
    names = [a[0] for a in AXES]
    check(names == ["1. Structural recovery", "2. Stability alignment",
                    "3. Behavioral faithfulness",
                    "4. Leakage-risk discrimination", "5. Resource cost"],
          "axis names and order = review SS5 final")
    out = build(OUT.parent / "Fig_vera_SELFTEST.pdf")
    xlo, xhi = _LIMS["x"]
    ylo, yhi = _LIMS["y"]
    clipped = [b for b in _BOXES
               if b[0] < xlo or b[1] < ylo or b[2] > xhi or b[3] > yhi]
    check(not clipped,
          f"every box lies inside the axes, stroke included ({clipped})")

    check(out.exists() and out.stat().st_size > 8000,
          "figure renders, non-trivial size")
    out.unlink()
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build()
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
