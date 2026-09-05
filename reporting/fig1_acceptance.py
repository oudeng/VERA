"""Fig. 1 asset acceptance, steps (1) to (3), run as a check rather than read.

P5R-O SS4 / commission SS6, made repeatable (P7-A closing round). The first
acceptance was done by hand on 2026-09-01. Then the asset was re-exported --
axis 4's fine note raised 8 pt -> 9 pt, and two British words respelled -- and
every one of those checks had to happen again. A procedure that has to be
re-derived from a receipt each time it is needed is not a procedure.

What it checks, and why each one is here:

  (1) FONTS      every face embedded with a Unicode map. An outlined figure
                 has no text layer, so none of the other checks can run and
                 no reader can search the page.
      SLOTS      every string the figure is supposed to draw, verbatim, from
                 the generator that is the declared content truth. Checked
                 LINE BY LINE: a whole-file comparison misreports five of them
                 as missing, because pdftotext interleaves the lines of
                 adjacent text boxes and a multi-line slot is several text
                 objects, not one.
      SUBSCRIPT  TAP's zero is a real subscript, not a small character on the
                 baseline. PowerPoint marks it baseline="-40000" in the run
                 properties and the PDF draws it at a reduced trm; both are
                 asserted, because either alone can be faked by the other.

  (2) WORDING    the terminology registry across the asset's text layer, and
                 the orthography layer with it -- a figure is part of the
                 document, and the reason this rerun exists is that it was
                 the last place carrying a British spelling.

  (3) SIZE       what the type measures ON PAPER, not in the asset. The two
                 are different by the ratio of the canvas to \\textwidth times
                 the inclusion width, and that gap is the entire subject: the
                 asset has always been legal against the brief and the
                 question has always been what survives the scaling.
      GRAY       the accent fills' luminance spread, recorded because four of
                 them collapse to one gray and the figure must not depend on
                 hue to be read.

    PYTHONHASHSEED=2025 python reporting/fig1_acceptance.py
    PYTHONHASHSEED=2025 python reporting/fig1_acceptance.py --width 0.91
    PYTHONHASHSEED=2025 python reporting/fig1_acceptance.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from experiments.package_layout import paper_file       # noqa: E402

REGISTRY = CODE_ROOT / "docs" / "figure_assets.json"
#: \textwidth of sn-jnl in this configuration, from \the\textwidth -- so it
#: is in TeX points. A PDF's page box is in BIG points, and 1 bp = 1/72 in
#: while 1 pt = 1/72.27 in. Treating the asset's 496 bp canvas as 496 TeX pt
#: overstates every on-paper size by 0.375%, which is small enough to look
#: like rounding and large enough to move a number the floor is judged on: it
#: predicted the axis-4 note at 7.52 pt where the page actually sets it at
#: 7.49. Everything below is computed in bp, the unit both the asset and the
#: measurement already use.
TEXTWIDTH_PT = 455.24408
BP_PER_PT = 72.0 / 72.27
TEXTWIDTH_BP = TEXTWIDTH_PT * BP_PER_PT
#: The brief's target and its hard floor (commission SS4.2).
TARGET_PT, FLOOR_PT = 8.0, 7.0


def _sh(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def asset_paths(figure: str = "Fig_vera") -> tuple:
    reg = json.loads(REGISTRY.read_text())
    a = next(x for x in reg["assets"] if x["figure"] == figure)
    pdf = ROOT / a["path"]
    pptx = ROOT / a["source_pptx"]
    if not pdf.exists():                       # a re-export may pre-date the
        pdf = paper_file(Path(a["path"]).name)  # registry being repointed
    if not pptx.exists():
        pptx = paper_file(Path(a["source_pptx"]).name)
    return pdf, pptx, a


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------- #
# (1) fonts, slots, subscript
# --------------------------------------------------------------------- #
def fonts(pdf: Path) -> dict:
    #: Counted FROM THE END: the type column is sometimes two words ("Type
    #: 1C", "CID TrueType"), so a fixed index from the left is wrong for some
    #: files and right for others -- the worst kind of wrong. Trailing layout
    #: is name .. type .. encoding emb sub uni objnum gen, so emb is [-5] and
    #: uni is [-3]. Getting this off by one made the gate read the SUBSET
    #: column as embedding (accidentally right) and the object number as the
    #: Unicode map (always wrong), which is how a fully-mapped figure was
    #: reported as having no Unicode map at all.
    rows = [l.split() for l in _sh(["pdffonts", str(pdf)]).splitlines()[2:]
            if l.strip()]
    emb = [r for r in rows if len(r) >= 7 and r[-5] == "yes"]
    uni = [r for r in rows if len(r) >= 7 and r[-3] == "yes"]
    return {"n": len(rows), "embedded": len(emb), "unicode_mapped": len(uni),
            "pass": bool(rows) and len(emb) == len(rows) == len(uni),
            "faces": [r[0] for r in rows]}


def slots() -> dict:
    """Every string the figure must draw, from the declared content truth.

    reporting/fig_vera.py is SUPERSEDED-BY-ASSET but kept precisely for this:
    it is what the commission was filled from, and the asset was accepted
    against the commission. P7-A SS1 respelled two of its strings, so the
    baseline moved with the manuscript -- which is the point. The supersession
    is recorded in the registry's orthography block, not implied here.
    """
    sys.path.insert(0, str(CODE_ROOT / "reporting"))
    import ast
    from reporting import fig_vera as F
    tree = ast.parse((CODE_ROOT / "reporting" / "fig_vera.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build")
    lits = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "text" and len(node.args) > 2
                and isinstance(node.args[2], ast.Constant)
                and isinstance(node.args[2].value, str)):
            lits[node.lineno] = node.args[2].value
    ordered = [lits[k] for k in sorted(lits) if lits[k] != "<-"]
    (b1t, b1s, b2t, b3t, refh, b4a, b4b, b5a, b5b) = ordered
    out = {"band1_title": b1t, "band1_subtitle": b1s, "band2_title": b2t}
    for i, (t1, t2, _c) in enumerate(F.READOUTS, 1):
        out[f"band2_box{i}"] = f"{t1}\n{t2}"
    out["eligibility_note"] = F.ELIGIBILITY
    out["band3_title"] = b3t
    for i, (n, s, r) in enumerate(F.AXES, 1):
        out[f"axis{i}"] = f"{n}\n{s}"
        out[f"ref_evidence_{i}"] = r
    out["ref_evidence_header"] = refh
    out["band4_line1"], out["band4_line2"] = b4a, b4b
    out["band5"] = f"{b5a}\n{b5b}"
    return out


def _flat(pdf: Path) -> str:
    """Three readings of the same page, because no single one suffices.

    -layout preserves columns but splits a line that crosses one. The
    reading-order dump keeps lines whole but interleaves adjacent boxes. And
    BOTH break a slot whose text box WRAPS: "no cross-axis verdict rule is
    committed in / advance; any overall reading is a post-hoc synthesis" is
    one string in the commission and two visual lines on the page, with a
    different box's text drawn between them.

    The third reading is mutool's draw order, where a wrapped box's lines are
    consecutive runs. Concatenating those puts the slot back together without
    loosening anything: it is still the exact character sequence that has to
    appear, only reassembled the way the renderer laid it down.
    """
    sep = " " * 3
    a = " ".join(_sh(["pdftotext", "-layout", str(pdf), "-"]).split())
    b = " ".join(_sh(["pdftotext", str(pdf), "-"]).split())
    #: Joined with NOTHING between runs. This exporter breaks a span at
    #: every hyphen, so "cross-axis" arrives as three runs; separating
    #: them by a space would manufacture "cross - axis" and report a
    #: string the page does contain as missing.
    c = " ".join("".join(r["text"] for r in trace_runs(pdf)).split())
    return sep.join((a, b, c))


def verbatim(pdf: Path, fill: dict = None) -> dict:
    """Each slot's each LINE, present verbatim on the page."""
    fill = fill or slots()
    flat = _flat(pdf)
    lines, missing = 0, []
    for k, v in fill.items():
        for ln in v.split("\n"):
            lines += 1
            probe = " ".join(ln.replace("TAP$_0$", "TAP0")
                               .replace("TAP₀", "TAP0")
                               .replace("$\\mathbf{D}$", "D")
                               .replace("**D**", "D").split())
            if probe not in flat:
                missing.append(f"{k}: {probe[:60]}")
    return {"n_slots": len(fill), "n_lines": lines, "missing": missing,
            "pass": not missing}


def subscript(pdf: Path, pptx: Path) -> dict:
    """TAP's zero is a true subscript in the source AND on the page."""
    src = None
    if pptx.exists():
        with zipfile.ZipFile(pptx) as z:
            xml = " ".join(z.read(n).decode("utf8", "replace")
                           for n in z.namelist()
                           if n.startswith("ppt/slides/slide")
                           and n.endswith(".xml"))
        #: the run carrying the zero must declare a lowered baseline
        src = bool(re.search(r'baseline="-\d+"[^>]*/?>(?:(?!</a:r>).)*?'
                             r'<a:t>\s*0\s*</a:t>', xml, re.S))
    runs = trace_runs(pdf)
    tap = [r for r in runs if r["text"].strip() in ("TAP", "TAP0")]
    zeros = [r for r in runs if r["text"].strip() == "0"]
    drawn = None
    if tap and zeros:
        parent = max(r["size"] for r in tap)
        drawn = min(r["size"] for r in zeros) < parent
    return {"pptx_baseline_marked": src, "pdf_drawn_smaller": drawn,
            "pass": bool(src) and bool(drawn)}


# --------------------------------------------------------------------- #
# (3) what the type measures on paper
# --------------------------------------------------------------------- #
def trace_runs(pdf: Path) -> list:
    """Every text run with its effective point size in the asset's own space.

    mutool reports each span's text matrix; the size is the matrix scale times
    the declared size, which is the only number that survives a designer
    scaling a text box rather than changing its font size.
    """
    #: There is no `size` attribute to read. A span's size is its text matrix
    #: scaled by the enclosing fill_text transform, and BOTH matter: this
    #: exporter writes every span at a nominal trm and puts the real scale in
    #: the transform (50 x 0.24 = 12 pt). Reading the trm alone would report
    #: every face as four times its printed size and call the figure legal.
    xml = _sh(["mutool", "draw", "-F", "trace", str(pdf)])
    runs, cur, outer = [], None, 1.0
    for m in re.finditer(
            r'<fill_text[^>]*transform="([-\d.eE ]+)"'
            r'|<span font="([^"]*)"[^>]*trm="([-\d.eE ]+)"[^>]*>'
            r'|<g unicode="([^"]*)"'
            r'|</span>', xml):
        if m.group(1) is not None:
            t = [float(x) for x in m.group(1).split()]
            outer = (abs(t[0]) + abs(t[3])) / 2 if len(t) >= 4 else 1.0
        elif m.group(2) is not None:
            trm = [float(x) for x in m.group(3).split()]
            inner = (abs(trm[0]) + abs(trm[3])) / 2 if len(trm) >= 4 else 1.0
            cur = {"font": m.group(2), "trm": inner, "transform": outer,
                   "size": inner * outer, "text": ""}
        elif m.group(4) is not None and cur is not None:
            cur["text"] += m.group(4)
        elif cur is not None:
            runs.append(cur)
            cur = None
    return runs


def baseline_offsets(pptx: Path) -> set:
    """The literal texts PowerPoint marks as raised or lowered runs.

    P5R-Z: the 7 pt floor binds BASELINE-LEVEL runs. A subscript is smaller
    than its body by definition and its legibility is carried by the run it
    belongs to, so it is measured through that parent. Detection is mechanical
    -- baseline="-40000" in the run properties -- not a judgment call.
    """
    if not pptx.exists():
        return set()
    with zipfile.ZipFile(pptx) as z:
        xml = " ".join(z.read(n).decode("utf8", "replace") for n in z.namelist()
                       if n.startswith("ppt/slides/slide") and n.endswith(".xml"))
    out = set()
    for m in re.finditer(r'baseline="(-?\d+)"(?:(?!</a:r>).)*?<a:t>(.*?)</a:t>',
                         xml, re.S):
        if m.group(1) != "0":
            out.add(m.group(2).strip())
    return out


def on_paper(pdf: Path, pptx: Path, width: float) -> dict:
    """Point sizes as printed, at `width` x \\textwidth."""
    page = _sh(["pdfinfo", str(pdf)])
    m = re.search(r"Page size:\s*([\d.]+) x ([\d.]+)", page)
    canvas_w = float(m.group(1))
    scale = width * TEXTWIDTH_BP / canvas_w
    offset = baseline_offsets(pptx)
    rows, tiny = {}, []
    for r in trace_runs(pdf):
        if not r["text"].strip():
            continue
        key = round(r["size"], 2)
        rows.setdefault(key, []).append(r["text"].strip())
    table = []
    for k in sorted(rows, reverse=True):
        paper = k * scale
        sample = sorted(rows[k], key=len, reverse=True)[0][:52]
        is_offset = any(t.strip() and t.strip() in offset for t in rows[k]) \
            or (k < 7 and all(len(t.strip()) <= 2 for t in rows[k]))
        table.append({"asset_pt": k, "paper_pt": round(paper, 2),
                      "n_runs": len(rows[k]), "baseline_offset": is_offset,
                      "sample": sample})
        if paper < FLOOR_PT and not is_offset:
            tiny.append(f"{k} pt -> {paper:.2f} pt: {sample}")
    base = [t for t in table if not t["baseline_offset"]]
    return {"canvas_bp": canvas_w, "textwidth_bp": round(TEXTWIDTH_BP, 4),
            "width": width, "scale": round(scale, 6),
            "table": table,
            "min_baseline_paper_pt": round(min(t["paper_pt"] for t in base), 2),
            "below_floor": tiny, "pass": not tiny}


def grayscale(pdf: Path, collapse_at: float = 0.02) -> dict:
    """Which distinct colors become the same gray, and how far apart they are.

    Not "the spread of every chromatic fill on the page" -- that number is
    large and means nothing, because black text and a pale background are
    supposed to differ. The question the deviation was filed about is narrower
    and answerable: are there DISTINCT colors a grayscale reader cannot tell
    apart? Two colors collapse when their relative luminance differs by less
    than `collapse_at`, so the groups are computed and reported rather than
    summarized into one figure.
    """
    xml = _sh(["mutool", "draw", "-F", "trace", str(pdf)])
    cols = set()
    for m in re.finditer(r'colorspace="DeviceRGB" color="([\d. ]+)"', xml):
        v = tuple(round(float(x), 4) for x in m.group(1).split())
        if len(v) == 3 and len(set(v)) > 1:       # a gray is not a hue
            cols.add(v)
    lum = sorted(((0.2126 * r + 0.7152 * g + 0.0722 * b), (r, g, b))
                 for r, g, b in cols)
    groups, cur = [], []
    for L, c in lum:
        if cur and L - cur[0][0] < collapse_at:
            cur.append((L, c))
        else:
            if len(cur) > 1:
                groups.append(cur)
            cur = [(L, c)]
    if len(cur) > 1:
        groups.append(cur)
    return {
        "n_distinct_hues": len(cols),
        "collapse_threshold": collapse_at,
        "collapsing_groups": [
            {"n": len(g), "luminance": [round(L, 4) for L, _ in g],
             "spread": round(g[-1][0] - g[0][0], 4),
             "rgb": [list(c) for _, c in g]} for g in groups],
        "n_colours_that_collapse": sum(len(g) for g in groups),
        "note": "recorded, non-blocking: the five axes are distinguished by "
                "ordinal and name, not by hue, so a reader who cannot "
                "separate these still reads the figure correctly",
    }


# --------------------------------------------------------------------- #
def run(width: float = 0.91) -> dict:
    pdf, pptx, reg = asset_paths()
    from reporting import facts_gate as fg
    txt = _sh(["pdftotext", str(pdf), "-"])
    pats = fg._load_registry()
    term: list = []
    fg._scan(f"asset:{pdf.name}", txt, pats, term)
    #: The asset is scanned with NO exemption. The rendered-page exemption
    #: exists for the manuscript PDF, which contains the figure; here we are
    #: looking at the figure itself, and it is the thing being accepted.
    spell = fg.scan_orthography({f"asset:{pdf.name}": txt})
    r = {
        "pdf": str(pdf.relative_to(ROOT)), "pdf_sha256": sha(pdf),
        "pdf_bytes": pdf.stat().st_size,
        "pptx": str(pptx.relative_to(ROOT)) if pptx.exists() else None,
        "pptx_sha256": sha(pptx) if pptx.exists() else None,
        "pptx_bytes": pptx.stat().st_size if pptx.exists() else None,
        "registered_pdf_sha256": reg.get("pdf_sha256"),
        "is_registered": sha(pdf) == reg.get("pdf_sha256"),
        "fonts": fonts(pdf),
        "verbatim": verbatim(pdf),
        "subscript": subscript(pdf, pptx),
        "terminology_hits": [h["variant"] for h in term],
        "spelling_hits": [f"{h['found']} -> {h['canonical']}" for h in spell],
        "on_paper": on_paper(pdf, pptx, width),
        "grayscale": grayscale(pdf),
    }
    r["pass"] = (r["fonts"]["pass"] and r["verbatim"]["pass"]
                 and r["subscript"]["pass"] and not term and not spell
                 and r["on_paper"]["pass"])
    return r


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    s = slots()
    c(len(s) == 23, f"23 slots, derived from the generator ({len(s)})")
    c(sum(len(v.split(chr(10))) for v in s.values()) == 33,
      "23 slots resolve to 33 literal lines -- the number the commission "
      "and the first acceptance both used")
    #: The British forms are NOT written out here. An earlier version spelled
    #: them inline, the P7-A converter respelled this file along with every
    #: other, and the assertion silently inverted -- it began demanding that
    #: the baseline contain no AMERICAN spelling, and failed on a correct
    #: baseline. Ask the registry, which is where the forms are declared.
    from reporting.facts_gate import _ortho
    _pat, _map = _ortho()
    c(not any(_pat.search(v) for v in s.values()),
      "the comparison baseline itself is American: P7-A SS1 respelled "
      "fig_vera.py, so the baseline moved with the manuscript rather than "
      "being patched to match the new asset")
    pdf, pptx, _ = asset_paths()
    c(pdf.exists(), f"the registered asset resolves: {pdf}")
    r = verbatim(pdf)
    c(r["pass"], f"33/33 verbatim on the delivered asset ({r['missing'][:2]})")
    o = on_paper(pdf, pptx, 0.91)
    c(o["scale"] < 1, "the asset is scaled DOWN on paper -- the whole subject")
    #: The unit fixture. 496 bp is 497.85 TeX pt, and a prediction that
    #: forgets it lands 0.375% high -- 7.52 where the page sets 7.49. The
    #: check is against what pdftex actually drew on page 6.
    c(abs(o["scale"] - 0.83212) < 5e-4,
      f"the predicted scale matches what the page draws ({o['scale']} vs "
      f"0.83212) -- bp and TeX pt are not the same unit")
    c(o["pass"], f"no baseline-level run below {FLOOR_PT} pt: {o['below_floor']}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=0.91)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = run(a.width)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(("[OK] " if r["pass"] else "[RED] ")
          + f"Fig. 1 acceptance (1)-(3) at width={a.width}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
