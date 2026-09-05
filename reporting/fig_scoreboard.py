"""Fig_scoreboard (P5R-C SS3.6; P5R-H SS3 rework): object x axis verdict
matrix. Every cell = the axis's committed decision-rule category (color)
+ headline numbers (text). No radar, no composite index -- the caption
states that the axes are complementary properties and are not aggregated.

Single source (P5R-H P1-5): every number and branch in this figure is
read from results/T5_stats/t_final.json -- nothing else.

Category vocabulary (one set, figure == caption): WIN / LOSS (a
prospectively specified rule applies and its statistical criterion is
met) / INDET (direction present, not statistically established) / DESC
(descriptive readout, incl. comparisons whose enumeration floor
precludes significance) / REF (the comparator itself) / EXCL (excluded
by construction) / NA.

Rule archaeology (P5R-H SS1.1, state b): the original committed rules
fixed per-class reporting and the interaction win-threshold but no
single primary inferential contrast, so the two leakage cells carry
INDET with their counts -- WIN/LOSS stay reserved for committed rules
whose statistical criterion is met.

Branch-change tripwire: --selftest asserts the REGISTERED snapshot of
every verdict-bearing field against the live t_final; any re-run that
changes a branch or count explodes here, forcing review before the
figure can be regenerated.

    PYTHONHASHSEED=2025 python reporting/fig_scoreboard.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402

T_FINAL = CODE_ROOT / "results" / "T5_stats" / "t_final.json"
OUT = CODE_ROOT / "reporting" / "out"

COL = {"WIN": "#2e7d32", "LOSS": "#c62828",
       "INDET": "#ef6c00", "DESC": "#757575", "REF": "#455a64",
       "NA": "#cfd8dc", "EXCL": "#8d6e63"}


def _retired_label() -> str:
    """The retired verdict's label, from the one module that defines it.

    Writing the literal here made this tripwire itself a hit against the
    quarantine gate the moment the file started shipping -- the same shape as
    the archive declaration that restated the phrases it forbade. A tripwire
    watching for a string it also contains is not watching, it is echoing.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "experiments"))
    from audit_history import RETIRED_LABEL
    return RETIRED_LABEL


def gather() -> dict:
    """The single evidence source."""
    return json.loads(T_FINAL.read_text())


def snapshot(tf: dict) -> dict:
    """The registered verdict snapshot (the tripwire's expectation)."""
    leak = tf["leakage"]
    return {
        "faith_MIMIC_branch": tf["faithfulness"]["MIMIC"]["branch"],
        "faith_eICU_branch": tf["faithfulness"]["eICU"]["branch"],
        "noprior_eICU_branch": tf["noprior_faithfulness"]["eICU"]["branch"],
        "noprior_MIMIC_branch": tf["noprior_faithfulness"]["MIMIC"]["branch"],
        "recovery_branch": tf["recovery"]["branch"],
        "interaction_counts": dict(leak["batches"]["original"]["interaction"]),
        "conf_D": leak["batches"]["confirmatory"]["interaction"]["SNI-D"],
        "conf_pooled_D": int(leak["pooled_gen3_secondary"]["SNI-D"
                                                          ].split("/")[0]),
        "discrepancy_D": leak["batches"]["original"]["discrepancy_control"
                                                     ]["SNI-D"],
        "probe_null": leak["probe_null_by_batch"]["original"]["detected"],
        # Both of these now read the SYMMETRIC pair, because that is what the
        # recovery cell prints. Watching the superseded pair would leave the
        # printed number unguarded -- the tripwire has to sit on the figure's
        # own source, not on the number it replaced.
        "probe_recovery_p_exact":
            tf["recovery"]["probe_vs_D_same_host_symmetric"]["p_exact"],
        "probe_vs_D_symmetric_T":
            tf["recovery"]["probe_vs_D_same_host_symmetric"]["T"],
        # The tripwire used to watch that the retired label was PRESENT under
        # its renamed key. Tenth review P0-1 removed the label from the
        # canonical store altogether, so the thing worth watching inverted:
        # the store must stay free of it. A tripwire that fires when the dead
        # rule comes back is worth more than one that fires when its epitaph
        # is edited.
        "retired_label_absent_from_fact_store": (
            _retired_label() not in json.dumps(tf))}


REGISTERED = {
    "faith_MIMIC_branch": "N", "faith_eICU_branch": "N",
    "noprior_eICU_branch": "S", "noprior_MIMIC_branch": "S",
    "recovery_branch": "N",
    "interaction_counts": {"SNI-D": 0, "P": 3,
                           "MissForest-importance": 6,
                           "SHAP-on-MissForest": 6,
                           "Permutation-on-MissForest": 6,
                           "Permutation-on-SNI": 6},
    "conf_D": 0, "conf_pooled_D": 0, "discrepancy_D": 6, "probe_null": 6,
    "probe_recovery_p_exact": 0.0625,
    # Registered 2026-08-29 (P5R-P SS2.3), the within-host pair under
    # information symmetry. Registered BY HAND on purpose: a tripwire whose
    # expectation is derived from the thing it watches guards nothing.
    "probe_vs_D_symmetric_T": 0.158627,
    "retired_label_absent_from_fact_store": True}


def cells(tf: dict) -> tuple:
    """(rows, axes, cell dict {(row, axis): (category, text)})."""
    leak = tf["leakage"]
    sd = tf["scoreboard_desc"]
    ic = leak["batches"]["original"]["interaction"]
    dc = leak["batches"]["original"]["discrepancy_control"]
    confI = leak["batches"]["confirmatory"]["interaction"]
    tfrec_n = tf["recovery"]["D_vs_TAP"]["n_seeds"]
    rows = ["SNI-D", "TAP", "Perm-on-SNI", "MF importance",
            "SHAP-on-MF", "Perm-on-MF"]
    axes = ["1. Structural recovery\n(validity, vs G)",
            "2. Stability alignment\n(robustness)",
            "3. Behavioral faithfulness\n(validity, vs $R_{\\mathrm{host}}$)",
            "4. Leakage-risk discrimination\n(vs L / host-use)",
            "5. Resource cost\n(operational)"]
    # Third review SS9.3: the five axes differ in inference unit, evidence
    # strength and applicability; the figure must say so per axis rather
    # than letting one category vocabulary imply one evidential currency.
    # P5R-O SS2 item 6: the inference UNIT stays in the figure, in its
    # minimal form; what each unit implies for inference -- the exact floor,
    # the Holm family, the post-hoc status of the paired tests -- moves to the
    # caption, which is where a reader has room to read it.
    units = [f"n = {tfrec_n} seeds",
             "n = seed pairs",
             f"n = {tf['faithfulness']['MIMIC']['n_seeds']} seeds / table",
             "n = 6 injections / class",
             "n = 1 probe / object"]
    stab = {o: (f"{sd['stability_rows12_mean'][o]['MIMIC']:.2f} / "
                f"{sd['stability_rows12_mean'][o]['eICU']:.2f}")
            for o in sd["stability_rows12_mean"]}
    rho = {o: (f"{sd['faith_rho_mean'][o]['MIMIC']:.2f} / "
               f"{sd['faith_rho_mean'][o]['eICU']:.2f}")
           for o in sd["faith_rho_mean"]}
    # T6.1: the band the Discussion compares D against is the one measured
    # under information symmetry, not the archived one. The archived reading
    # stays in Table 5's note; a figure cell has no room for both, and the
    # cell that contradicts the text is the worse of the two options.
    host = (f"{sd['host_band_mean_symmetric']['MIMIC']:.2f} / "
            f"{sd['host_band_mean_symmetric']['eICU']:.2f}")
    cost = (f"total {sd['cost_display']['grid_range']} (grid) /\n"
            f"{sd['cost_display']['single_thread_range']} "
            f"(1-thread, algorithm clock); "
            f"marginal ~0").replace("$\\times$", "\u00d7").replace("--", "\u2013")
    c = {}
    tfrec = tf["recovery"]["D_vs_TAP"]
    c[("SNI-D", 0)] = ("INDET",
                       f"T vs TAP {tfrec['T']:+.3f} "
                       f"[{tfrec['ci95_T'][0]:+.3f},{tfrec['ci95_T'][1]:+.3f}]\n"
                       f"exact p={tfrec['p_exact']:.3f} (floor 0.0625)")
    tff = tf["faithfulness"]
    c[("SNI-D", 2)] = ("INDET",
                       f"T {tff['MIMIC']['T']:+.3f} / {tff['eICU']['T']:+.3f}\n"
                       f"Holm p={tff['MIMIC']['p_holm']:.3f} / "
                       f"{tff['eICU']['p_holm']:.2f}")
    c[("SNI-D", 1)] = ("DESC", f"{stab['SNI-D']}\nhost band {host}")
    c[("SNI-D", 4)] = ("DESC", cost)
    c[("SNI-D", 3)] = ("INDET",
                       f"interaction {ic['SNI-D']}/6 + "
                       f"{confI['SNI-D']}/6 conf\n"
                       f"discr. ctrl {dc['SNI-D']}/6")
    c[("TAP", 0)] = ("REF", "comparison baseline\n(all deltas vs TAP)")
    c[("TAP", 2)] = ("REF", "comparison baseline")
    c[("TAP", 1)] = ("DESC", "1.00 (seed-invariant\nby construction)")
    c[("TAP", 4)] = ("DESC", "~0.1 s total")
    c[("TAP", 3)] = ("DESC", f"interaction {ic['P']}/6 + "
                             f"{confI['P']}/6 conf\n"
                             f"discr. ctrl {dc['P']}/6 by construction")
    # P5R-P SS1: the WITHIN-HOST pair under information symmetry. The category
    # does not move -- the enumeration floor is unchanged, so significance is
    # unreachable for this comparison either way; only the caliber of the
    # number changes, and the cell says which caliber it is.
    tfp = tf["recovery"]["probe_vs_D_same_host_symmetric"]
    c[("Perm-on-SNI", 0)] = ("INDET",
                             f"same host, symmetric signal:\n"
                             f"T {tfp['T']:+.3f} "
                             f"[{tfp['ci95_T'][0]:+.2f},{tfp['ci95_T'][1]:+.2f}]\n"
                             f"exact p={tfp['p_exact']:.4f} = floor; "
                             f"no sig claim at 5 seeds")
    c[("Perm-on-SNI", 2)] = ("EXCL",
                             "excluded by construction:\ndefines the "
                             "behavioral reference")
    c[("Perm-on-SNI", 1)] = ("DESC", f"host band {host}")
    c[("Perm-on-SNI", 4)] = ("DESC", "audit-only on\nexisting host")
    nb = leak["probe_null_by_batch"]["original"]
    c[("Perm-on-SNI", 3)] = ("INDET",
                             f"interaction {ic['Permutation-on-SNI']}/6 "
                             f"+ 6/6 conf; "
                             f"discr. ctrl {dc['Permutation-on-SNI']}/6\n"
                             f"null rate {nb['detected']}/{nb['n']} "
                             f"(calibration diagnostic)")
    for row, obj in (("MF importance", "MissForest-importance"),
                     ("SHAP-on-MF", "SHAP-on-MissForest"),
                     ("Perm-on-MF", "Permutation-on-MissForest")):
        c[(row, 0)] = ("DESC", f"AUROC {sd['pilot_auroc'][obj]:.3f}\n"
                               f"(stronger imputer host)")
        # Eighth review P1-9: the cell has to say which statistic it is,
        # in the name the main text already uses for it.
        c[(row, 2)] = ("DESC", f"median target-wise rho\n{rho[obj]}")
        c[(row, 1)] = ("DESC", stab[obj])
        c[(row, 4)] = ("DESC", "family = the cost\nreference (1×)")
        c[(row, 3)] = ("DESC", f"interaction {ic[obj]}/6\n"
                               f"discr. ctrl {dc[obj]}/6")
    return rows, axes, units, c


FOOTER = ("Axes are complementary properties and are NOT aggregated into "
          "a composite score. One category vocabulary: WIN / LOSS appear "
          "ONLY where a prospectively specified rule applies and its "
          "statistical criterion is met; INDET = direction present but "
          "not statistically established under the exact seed-block "
          "test (this includes the five-seed families, where the exact "
          "floor of 0.0625 puts significance out of reach at any effect "
          "size); DESC = no prespecified per-object decision rule applies; "
          "REF = the "
          "comparison baseline itself; EXCL = excluded by construction; "
          "NA = not applicable. Resource cost is an operational "
          "dimension, reported separately and never compensatory "
          "against the validity axes.")


#: measured geometry of the last render's axis headers, for the selftest
_HEADERS: list = []
_HEADER_GEOM: list = []


def _measure_headers(fig, ax, headers, cell_w=1.0):
    """Record where each axis header landed. MEASURE ONLY.

    Its predecessor shrank the font until the header fitted its column, which
    is how a 6 pt label reached the page. P5R-O forbids that: the wrapping is
    decided before the text is placed, at a size that is never reduced, and
    this function only records the result so _selftest can prove the columns
    do not collide and nothing renders below the floor.
    """
    fig.canvas.draw()
    inv = ax.transData.inverted()

    def bb(t):
        return t.get_window_extent(fig.canvas.get_renderer()).transformed(inv)

    _HEADER_GEOM.clear()
    for j, t1, t2 in headers:
        b1, b2 = bb(t1), bb(t2)
        _HEADER_GEOM.append({
            "axis": j + 1,
            "title": [round(b1.x0, 3), round(b1.x1, 3)],
            "unit": [round(b2.x0, 3), round(b2.x1, 3)],
            "title_size": round(t1.get_fontsize(), 2),
            "unit_size": round(t2.get_fontsize(), 2)})


#: measured geometry of the last render's cell texts, for the selftest
_CELL_GEOM: list = []


def _measure_cells(fig, ax, cells_drawn, cell_w=1.0):
    """Record where each cell's text landed. MEASURE ONLY, for the same
    reason as the headers: containment is achieved by wrapping at a fixed
    size, and this proves it rather than producing it."""
    fig.canvas.draw()
    inv = ax.transData.inverted()

    def bb(t):
        return t.get_window_extent(fig.canvas.get_renderer()).transformed(inv)

    _CELL_GEOM.clear()
    for row, j, t_txt, t_cat, y0, y1 in cells_drawn:
        bt, bc = bb(t_txt), bb(t_cat)
        _CELL_GEOM.append({
            "row": row, "axis": j + 1,
            "box": [round(j + 0.02, 3), round(j + 0.98, 3)],
            # the box's own vertical extent: without it nothing could tell
            # that a cell's first line had been sliced off by the box's top
            # edge and printed white on white paper
            "box_y": [round(y0, 3), round(y1, 3)],
            "size_pt": round(t_txt.get_fontsize(), 2),
            "text": [round(bt.x0, 3), round(bt.x1, 3)],
            "text_y": [round(bt.y0, 3), round(bt.y1, 3)],
            "cat_y": [round(bc.y0, 3), round(bc.y1, 3)],
            "size": round(t_txt.get_fontsize(), 2)})


PRINT_W_PT = 0.98 * 455.24408
#: matplotlib writes PDF at 72 pt per inch, so this is the divisor that makes
#: the emitted page exactly PRINT_W_PT wide and the LaTeX scale exactly 1.0
PT_PER_IN = 72.0
#: SS1.3: nothing in the figure may render below this ON PAPER. Because the
#: figure is now drawn at print size, the number in the source IS the number
#: on the page, and the selftest asserts it.
MIN_PT = 8.0

#: The legend the categories get, in the order they are read. Two to four
#: words each: the caption carries the definitions, this carries recognition.
GLOSS = {"WIN": "rule met, advantage",
         "LOSS": "rule met, deficit",
         "INDET": "not established",
         "DESC": "descriptive; not inferentially classified",
         "REF": "baseline",
         "EXCL": "defines the reference",
         "NA": "not applicable"}
LEGEND_ORDER = ["WIN", "LOSS", "INDET", "DESC", "REF", "EXCL", "NA"]

#: measured geometry of the last render's legend, for the selftest
_LEGEND_GEOM: list = []


def _wrap(fig, text, max_w_pt, size, hard_min=1):
    """Wrap to the column, never shrink to it (SS1: font sizes only increase).

    Measures the rendered width of each candidate wrapping instead of
    guessing from character counts, because the answer depends on the font.
    """
    import textwrap
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=size)

    def width_pt(s):
        probe.set_text(s)
        return max((probe.get_window_extent(renderer=r).width
                    for _ in [0]), default=0.0) * 72.0 / fig.dpi

    lines = str(text).split("\n")
    out = []
    for ln in lines:
        if not ln:
            out.append(ln)
            continue
        n = max(hard_min, len(ln))
        while n > 4 and width_pt(ln if n >= len(ln) else
                                 max(textwrap.wrap(ln, n) or [ln], key=len)) \
                > max_w_pt:
            n -= 2
        out.extend(textwrap.wrap(ln, n) or [ln])
    probe.remove()
    return "\n".join(out)


def _legend_rows(rows, axes, c) -> int:
    """How many rows the legend will need, decided before the height is set."""
    keys = [k for k in ("WIN", "LOSS", "INDET", "DESC", "REF", "EXCL", "NA")
            if any(c.get((r_, j), ("NA", ""))[0] == k
                   for r_ in rows for j in range(len(axes)))]
    # a cheap proxy for the measured width: the glosses are what overflow
    chars = sum(len(f"{k} - {GLOSS[k]}") for k in keys)
    return 1 if chars <= 88 else 2


def _draw_legend(fig, keys, W_IN, H_IN, LEFT_IN, RIGHT_IN, legend_in, size):
    """One compact row: swatch, token, and a two-to-four word gloss.

    Only the categories the profile actually uses are shown -- a legend that
    lists states no cell is in would be a fifth thing to read for nothing --
    and the row is measured afterwards so the selftest can prove the entries
    do not run into each other or below the size floor.
    """
    from matplotlib.patches import Rectangle
    # the legend belongs to the whole figure, so it spans the whole width --
    # confining it to the matrix left it 9% too wide for one line
    lax = fig.add_axes([0.0, 0.02 / H_IN, 1.0 - RIGHT_IN / W_IN,
                        legend_in / H_IN])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    # Entries are laid out by their MEASURED width with equal gaps, not on an
    # equal pitch: equal slots give "REF -- baseline" the same room as
    # "EXCL -- defines the reference" and the long ones run into their
    # neighbors, which is what the first render did.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", fontsize=size)
    labels = [f"{k} \u2013 {GLOSS[k]}" for k in keys]
    widths = []
    for s in labels:
        probe.set_text(s)
        widths.append(probe.get_window_extent(renderer=r).width
                      / (fig.get_size_inches()[0] * fig.dpi))
    probe.remove()
    sw = 0.014                       # swatch width, axes fraction
    span = 1.0 - RIGHT_IN / W_IN
    used_w = sum(widths) / span + len(keys) * (sw + 0.008)
    # Two rows when one will not hold the glosses. Buying the space with a
    # smaller font is forbidden (P5R-O SS1) and shortening the gloss would
    # undo the seventh review's point about DESC, so the legend grows a row.
    n_rows = 1 if used_w <= 1.0 else 2
    per = (len(keys) + n_rows - 1) // n_rows
    ys = [0.50] if n_rows == 1 else [0.72, 0.24]
    _LEGEND_GEOM.clear()
    for row in range(n_rows):
        ks = keys[row * per:(row + 1) * per]
        ws = widths[row * per:(row + 1) * per]
        ls = labels[row * per:(row + 1) * per]
        row_used = sum(ws) / span + len(ks) * (sw + 0.008)
        gap = max(0.004, (1.0 - row_used) / max(1, len(ks) - 1))
        x, y = 0.0, ys[row]
        for k, s_, w in zip(ks, ls, ws):
            h = 0.34 if n_rows == 1 else 0.22
            lax.add_patch(Rectangle((x, y - h / 2), sw, h, facecolor=COL[k],
                                    edgecolor="none", transform=lax.transAxes))
            tt = lax.text(x + sw + 0.008, y, s_, ha="left", va="center",
                          fontsize=size, color="#263238")
            x_end = x + sw + 0.008 + w / span
            _LEGEND_GEOM.append({"key": k, "row": row,
                                 "x": [round(x, 4), round(x_end, 4)],
                                 "text": tt,
                                 "size_pt": round(tt.get_fontsize(), 2)})
            x = x_end + gap
        _LEGEND_GEOM[-1]["row_end"] = round(x - gap, 4)
    # replace the PREDICTED extents with the rendered ones: the prediction is
    # what decides the layout, the measurement is what proves it, and the two
    # must not be the same number or the assertion proves nothing
    fig.canvas.draw()
    inv = lax.transAxes.inverted()
    for g in _LEGEND_GEOM:
        b = g.pop("text").get_window_extent(
            renderer=fig.canvas.get_renderer()).transformed(inv)
        g["x"] = [round(b.x0, 4), round(b.x1, 4)]
    for g in _LEGEND_GEOM:
        if "row_end" in g:
            g["row_end"] = g["x"][1]
    return lax


def build(out_pdf: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tf = gather()
    got = snapshot(tf)
    assert got == REGISTERED, (
        f"verdict snapshot changed -- review before regenerating: {got}")
    rows, axes, units, c = cells(tf)

    W_IN = PRINT_W_PT / PT_PER_IN
    RIGHT_IN = 0.02
    SZ = {"header": 8.2, "unit": 8.0, "row": 8.4, "cell": 8.0, "cat": 8.0}
    assert min(SZ.values()) >= MIN_PT, SZ

    # The row-label gutter is MEASURED, not guessed. Guessing it clipped
    # "MF importance" to "F importance" on the first render of this layout --
    # a defect that is invisible to every check that reads strings.
    import matplotlib.pyplot as _plt
    g = _plt.figure(figsize=(W_IN, 1.0))
    g.canvas.draw()
    _pr = g.text(0, 0, "", fontsize=SZ["row"], fontweight="bold")
    _w = []
    for r_ in rows:
        _pr.set_text(r_)
        _w.append(_pr.get_window_extent(renderer=g.canvas.get_renderer()
                                        ).width * 72.0 / g.dpi)
    _plt.close(g)
    LEFT_IN = (max(_w) + 8.0) / PT_PER_IN
    col_in = (W_IN - LEFT_IN - RIGHT_IN) / len(axes)
    col_pt = col_in * PT_PER_IN
    pad_pt = 5.0              # inside a cell, each side


    # a first pass on a scratch canvas, to learn how many lines each text
    # needs at the target size; the real canvas is then sized to hold them
    scratch = plt.figure(figsize=(W_IN, 4.0))
    scratch.canvas.draw()
    wrapped = {(r_, j): _wrap(scratch, c.get((r_, j), ("NA", ""))[1],
                              col_pt - 2 * pad_pt, SZ["cell"])
               for r_ in rows for j in range(len(axes))}
    heads = [_wrap(scratch, a, col_pt - 2 * pad_pt, SZ["header"]) for a in axes]
    unitw = [_wrap(scratch, u, col_pt - 2 * pad_pt, SZ["unit"]) for u in units]
    plt.close(scratch)

    head_lines = max(len(h.split("\n")) for h in heads)
    unit_lines = max(len(u.split("\n")) for u in unitw)

    # SS1.2 remedy (1): buy space with height, never with font size. Each row
    # gets the height ITS OWN tallest cell needs; a single global maximum gave
    # every two-line row the height of the six-line one and made the figure a
    # third taller than it has to be.
    # A row is as tall as its own text needs, derived from the geometry
    # rather than tuned: the text block occupies the band between the top of
    # the box and the category label below it, which is 0.66 of the row.
    line_in = 1.32 * SZ["cell"] / PT_PER_IN
    text_h = [max(len(wrapped[(r_, j)].split("\n"))
                  for j in range(len(axes))) * line_in for r_ in rows]
    # 0.72, not 0.66: the band the text sits in was given a third of the row
    # as slack, which was more than the descenders need and made the figure
    # taller than the float page can hold once the caption is counted. The
    # overprint assertions below are what makes tightening it safe -- they
    # measure, so a row that is now too tight fails loudly instead of
    # silently printing one line on top of another.
    row_h = [max(0.30, th / 0.72 + 0.04) for th in text_h]
    unit_block_in = unit_lines * 1.45 * SZ["unit"] / PT_PER_IN
    head_in = unit_block_in + head_lines * 1.45 * SZ["header"] / PT_PER_IN + 0.07
    # two rows when the glosses do not fit one (seventh review SS11.2
    # widened DESC); measured, not guessed -- the row band pays for it
    legend_in = (25.0 if _legend_rows(rows, axes, c) == 1
                 else 38.0) / PT_PER_IN
    H_IN = head_in + sum(row_h) + legend_in + 0.05

    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([LEFT_IN / W_IN, (legend_in + 0.02) / H_IN,
                       (W_IN - LEFT_IN - RIGHT_IN) / W_IN,
                       (H_IN - head_in - legend_in - 0.02) / H_IN])
    ax.set_xlim(0, len(axes))
    total = sum(row_h)
    ax.set_ylim(0, total)
    ax.axis("off")
    tops = []
    acc = total
    for h in row_h:
        tops.append(acc)
        acc -= h
    _HEADERS.clear()
    # Both header lines grow UPWARD from just above the boxes, so a unit
    # marker that wraps to two lines cannot descend into the first row --
    # which is exactly what it did when it was anchored by its top.
    gap = 0.035
    for j, (a, u) in enumerate(zip(heads, unitw)):
        t2 = ax.text(j + 0.5, total + gap, u, ha="center", va="bottom",
                     fontsize=SZ["unit"], style="italic", color="#37474f")
        t1 = ax.text(j + 0.5, total + gap + unit_block_in, a, ha="center",
                     va="bottom", fontsize=SZ["header"], fontweight="bold",
                     linespacing=1.35)
        _HEADERS.append((j, t1, t2))
    _measure_headers(fig, ax, _HEADERS)
    drawn = []
    for i, r_ in enumerate(rows):
        y = tops[i] - row_h[i]
        h = row_h[i]
        ax.text(-0.04, y + h / 2, r_, ha="right", va="center",
                fontsize=SZ["row"], fontweight="bold")
        for j in range(len(axes)):
            cat = c.get((r_, j), ("NA", ""))[0]
            txt = wrapped[(r_, j)]
            ax.add_patch(Rectangle((j + 0.02, y + 0.02 * h), 0.96, 0.96 * h,
                                   facecolor=COL[cat], alpha=0.9,
                                   edgecolor="white", lw=1.6))
            tcol = "white" if cat != "NA" else "#37474f"
            t_txt = ax.text(j + 0.5, y + 0.64 * h, txt, ha="center",
                            va="center", fontsize=SZ["cell"], color=tcol,
                            linespacing=1.30)
            t_cat = ax.text(j + 0.5, y + 0.13 * h, cat, ha="center",
                            va="center", fontsize=SZ["cat"], color=tcol,
                            fontweight="bold")
            drawn.append((r_, j, t_txt, t_cat,
                          y + 0.02 * h, y + 0.98 * h))
    _measure_cells(fig, ax, drawn)
    _draw_legend(fig, [k for k in LEGEND_ORDER
                       if any(v[0] == k for v in c.values())],
                 W_IN, H_IN, LEFT_IN, RIGHT_IN, legend_in, SZ["cell"])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # no bbox_inches="tight": the canvas IS the printed size, and trimming it
    # would silently rescale everything the size floor was computed against
    fig.savefig(out_pdf, metadata={"CreationDate": None})
    plt.close(fig)
    meta = {"generator": "reporting/fig_scoreboard.py",
            "input": str(T_FINAL),
            "snapshot": got,
            "code_SNI commit": runconfig.git_commit()}
    out_pdf.with_suffix(".provenance.json").write_text(
        json.dumps(meta, indent=1))
    plt.close(fig)
    return out_pdf


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    tf = gather()
    got = snapshot(tf)
    for k, v in REGISTERED.items():
        check(got[k] == v, f"snapshot[{k}] == {v!r} (got {got[k]!r})")
    rows, axes, units, c = cells(tf)
    check(len(units) == len(axes),
          "SS9.3: every axis carries its inference unit")
    check(len(rows) == 6 and len(axes) == 5, "6 objects x 5 axes")
    check(axes[3].startswith("4. Leakage-risk")
          and axes[4].startswith("5. Resource cost"),
          "axis order: leakage fourth, cost fifth (review SS5)")
    check("R_{\\mathrm{host}}" in axes[2],
          "faithfulness axis references R_host, typeset not raw")
    check(c[("Perm-on-SNI", 2)][0] == "EXCL",
          "probe faithfulness cell = excluded by construction")
    check(c[("SNI-D", 3)][0] == "INDET"
          and c[("Perm-on-SNI", 3)][0] == "INDET",
          "SS1.1 state (b): no committed primary contrast -> both "
          "leakage cells INDET (direction and counts reported)")
    check(c[("SNI-D", 0)][0] == "INDET"
          and c[("SNI-D", 2)][0] == "INDET",
          "SS2: D recovery/faithfulness are INDETERMINATE, not LOSS")
    check(c[("Perm-on-SNI", 0)][0] == "INDET"
          and c[("SNI-D", 0)][0] == "INDET"
          and "floor" in c[("Perm-on-SNI", 0)][1],
          "P5R-J SS2: both five-seed recovery cells are INDET (the exact "
          "floor makes significance unreachable for both), effect sizes "
          "printed")
    check("null rate" in c[("Perm-on-SNI", 3)][1]
          and "p=" not in c[("Perm-on-SNI", 3)][1],
          "P5R-K SS2: probe cell carries its observed null rate as a "
          "calibration diagnostic, with no fixed-alpha binomial p")
    from reporting.termmap import banned_variants
    blob = " ".join(t for (_cat, t) in c.values()) + " " + " ".join(axes) \
        + " " + " ".join(units) + " " + FOOTER
    for b in banned_variants():
        check(b.lower() not in blob.lower(), f"figure text free of: {b}")
    # RENDERED geometry: five headers on a fixed pitch will run into each
    # other at a fixed font size, which is what "unreadable at normal
    # zoom" meant. Measure the produced figure rather than its strings.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        build(Path(td) / "probe.pdf")
    # Cell containment: text inside its own box, clear of the category
    # label below it (P5R-K SS4.2).
    # A MARGIN, not a hairline. matplotlib's text extent under-reports the
    # ascender zone by about a point, so a fit measured to the box edge still
    # printed with the tops of "h", "t" and the digits shaved flat against the
    # box border -- eleven of thirty cells, found by reading the page after
    # the hairline version of this check had already passed.
    PAD = 0.035
    spill = [f"{g['row']}/axis{g['axis']}"
             for g in _CELL_GEOM
             if g["text_y"][1] > g["box_y"][1] - PAD
             or g["text_y"][0] < g["box_y"][0] + PAD]
    check(not spill,
          f"every cell's text clears its own colored box by {PAD} ({spill})")
    check(len(_CELL_GEOM) == len(rows) * len(axes),
          "every cell text was measured")
    spill = [g for g in _CELL_GEOM
             if g["text"][0] < g["box"][0] or g["text"][1] > g["box"][1]]
    check(not spill, f"no cell text crosses its box edge ({spill[:2]})")
    stack = [g for g in _CELL_GEOM if g["text_y"][0] < g["cat_y"][1]]
    check(not stack, f"no cell text overlaps its category label ({stack[:2]})")
    # P5R-O SS1.3: the figure is drawn at its printed size, so the size in
    # the source IS the size on the page. Assert the floor on what was
    # actually rendered, not on what the constants say.
    sizes = ([g["size_pt"] for g in _CELL_GEOM]
             + [g["title_size"] for g in _HEADER_GEOM]
             + [g["unit_size"] for g in _HEADER_GEOM]
             + [g["size_pt"] for g in _LEGEND_GEOM])
    check(sizes and min(sizes) >= MIN_PT,
          f"nothing renders below {MIN_PT} pt on paper "
          f"(smallest {min(sizes) if sizes else 0:.2f} pt)")
    import subprocess as _sp
    with tempfile.TemporaryDirectory() as td2:
        f2 = Path(td2) / "p.pdf"
        build(f2)
        info = _sp.run(["pdfinfo", str(f2)], capture_output=True, text=True).stdout
        import re as _re
        m = _re.search(r"Page size:\s*([0-9.]+) x ([0-9.]+)", info)
        w = float(m.group(1)) if m else 0.0
        check(abs(w - PRINT_W_PT) < 1.0,
              f"the PDF is drawn at its printed width, so LaTeX scales it by "
              f"1.0 ({w:.1f} pt vs {PRINT_W_PT:.1f} pt)")
    check(FOOTER not in "".join(t_ for (_c, t_) in c.values()),
          "the moved caption text is not also inside the figure")
    # the legend: one entry per category the profile actually uses, each
    # inside its own slot, none running into the next
    used = sorted({v[0] for v in c.values()})
    check(sorted(g["key"] for g in _LEGEND_GEOM) == used,
          f"the legend lists exactly the categories in use ({used})")
    # row-aware: the last entry of one row sits to the RIGHT of the first
    # entry of the next, which is not an overlap
    over = [a["key"] for a, b in zip(_LEGEND_GEOM, _LEGEND_GEOM[1:])
            if a.get("row", 0) == b.get("row", 0)
            and a["x"][1] > b["x"][0] + 1e-6]
    check(not over, f"no legend entry runs into the next ({over})")
    ends = [g["row_end"] for g in _LEGEND_GEOM if "row_end" in g]
    check(ends and max(ends) <= 1.0 + 1e-6,
          f"every legend row fits the figure width "
          f"({max(ends):.3f} <= 1.0, {len(ends)} row(s))")
    # the caption says the profile reads in grayscale without loss. That is
    # true because every cell PRINTS its category token, not because the hues
    # separate under a luminance transform -- DESC and EXCL do not. Assert the
    # thing the claim actually rests on.
    check(len(_CELL_GEOM) == len(rows) * len(axes)
          and all(g["cat_y"][1] > g["cat_y"][0] for g in _CELL_GEOM),
          "every cell prints its category token, which is what makes the "
          "grayscale claim in the caption true")
    check(not any(k in used for k in ("WIN", "LOSS")),
          "no cell carries WIN or LOSS -- the caption states this as a fact "
          "about the profile, so the figure asserts it")
    check(len(_HEADER_GEOM) == len(axes), "every axis header was measured")
    prev_t = prev_u = None
    for g in _HEADER_GEOM:
        check(g["title"][1] - g["title"][0] <= 1.0,
              f"axis {g['axis']} title fits its column "
              f"({g['title'][1] - g['title'][0]:.2f} <= 1.0)")
        check(prev_t is None or g["title"][0] >= prev_t - 1e-6,
              f"axis {g['axis']} title clears the one to its left")
        check(prev_u is None or g["unit"][0] >= prev_u - 1e-6,
              f"axis {g['axis']} inference unit clears the one to its left")
        check(g["unit"][1] - g["unit"][0] <= 1.0,
              f"axis {g['axis']} inference unit fits its column "
              f"({g['unit'][1] - g['unit'][0]:.2f} <= 1.0)")
        prev_t, prev_u = g["title"][1], g["unit"][1]
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build(OUT / "Fig_scoreboard.pdf")
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
