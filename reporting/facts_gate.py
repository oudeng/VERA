"""P5R-H SS0: the single-facts gate.

Terminology registry (docs/terminology_registry.json) enforced across five
layers -- (1) tex sources with comments stripped, (2) every generator's
string constants (ast-extracted, figure generators included), (3) the
rendered text layer of both PDFs, (4) the package README, (5) the package
CHANGE_SUMMARY -- plus numeric consistency of the abstract macros, the
faithfulness table and the Fig-3 snapshot against results/T5_stats/
t_final.json. Any banned-variant hit or numeric drift is red.

This gate is a PRE-COMPILE precondition: reporting/compile_gate.py calls
scan_sources() before compiling and scan_pdf()+numeric_checks() after.

    env PYTHONHASHSEED=2025 python reporting/facts_gate.py            # full run
    env PYTHONHASHSEED=2025 python reporting/facts_gate.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import json
import math
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
REGISTRY = CODE_ROOT / "docs" / "terminology_registry.json"
T_FINAL = CODE_ROOT / "results" / "T5_stats" / "t_final.json"
STAGING = CODE_ROOT.parent / "internal_review" / "ir_staging"

TEX_FILES = [paper_file("paperY_main.tex"), paper_file("paperY_ESM.tex")]
PDF_FILES = [paper_file("paperY_main.pdf"), paper_file("paperY_ESM.pdf")]
#: Scanners carry the strings they hunt. Both exclusions are declared in
#: docs/terminology_registry.json scan_exemptions.files_by_role as well,
#: so an exemption cannot grow quietly.
EXCLUDE_SOURCES = {"facts_gate.py", "package_gates.py",
                   "fill_designer_brief.py"}


def _load_registry(path: Path = None) -> list:
    reg = json.loads((path or REGISTRY).read_text())
    pats = []
    for term in reg["terms"]:
        for b in term.get("banned", []):
            toks = [re.escape(t) for t in b.split()]
            pats.append((b, re.compile(r"[\s-]*".join(toks), re.I)))
        for b in term.get("banned_word", []):
            pats.append((b + " (word)", re.compile(r"\b" + re.escape(b) + r"\b", re.I)))
    return pats


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


#: Comment markers per language. The marker is SYNTAX, not a character of the
#: prose it introduces -- and forgetting that is how a scanner reads across a
#: comment boundary. `_norm` collapses newlines, so a line ending in "a
#: reviewer" joined the next line's "#" and produced "a reviewer #", which
#: matched a banned phrase that appears nowhere in the file. Same family as the
#: TeX case below, where a "%" swallowed a word: both are the parser confusing
#: comment punctuation with content.
_LINE_COMMENT = {".py": r"#:?", ".sh": r"#", ".rb": r"#", ".yml": r"#",
                 ".yaml": r"#", ".toml": r"#", ".cfg": r"#"}


def strip_comment_markers(text: str, suffix: str) -> str:
    """Drop the leading comment marker from each line, keep the prose.

    The comment TEXT is still scanned -- withdrawn wording in a comment is a
    real hit. What is removed is the marker, so that joining wrapped lines
    cannot manufacture a phrase that spans the boundary.
    """
    mark = _LINE_COMMENT.get(suffix.lower())
    if not mark:
        return text
    return re.sub(rf"^[ \t]*{mark}[ \t]?", "", text, flags=re.M)


def _strip_tex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def _scan(label: str, text: str, pats: list, hits: list) -> None:
    t = _norm(text)
    for name, rx in pats:
        for m in rx.finditer(t):
            ctx = t[max(0, m.start() - 40):m.end() + 40]
            hits.append({"layer": label, "variant": name, "context": ctx})


#: How many generator files layer 2 actually read on the last
#: scan_sources() call. A layer that reads nothing must say so.
_LAYER2 = {"scanned": 0, "identifiers_skipped": 0}


def scan_sources(pats=None) -> list:
    """Layers 1 (tex) + 2 (generator string constants) + 4/5 (staging docs)."""
    pats = pats or _load_registry()
    hits: list = []
    for f in TEX_FILES:
        _scan(f"tex:{f.name}", _strip_tex_comments(f.read_text()), pats, hits)
    # ...and the comments themselves, for the one thing stripping them hides:
    # a trailing comment that ate the rest of its line. This must sit in
    # scan_sources, not in run(), because compile_gate calls scan_sources as a
    # PRE-COMPILE precondition -- which is the only moment the loss is cheap.
    hits += scan_swallowed_prose()
    #: In the repository the generators live under CODE_ROOT/reporting. In a
    #: review package CODE_ROOT is rebound to gate_inputs/code_SNI, which
    #: carries docs, results and rendered output but NO Python at all -- so
    #: this layer silently scanned nothing while the gate went on reporting
    #: "five layers". The shipped scripts are under <pkg>/code/ instead; look
    #: there too, and count what was actually read so an empty layer is
    #: visible in the output rather than implied away. (Eleventh round.)
    gens = sorted((CODE_ROOT / "reporting").glob("*.py"))
    if not gens:
        # A package binds CODE_ROOT to <pkg>/gate_inputs/code_SNI, which holds
        # no Python; the scripts that ship live at <pkg>/code. Walk up until a
        # sibling "code" tree with generators in it turns up.
        for base in (CODE_ROOT, *CODE_ROOT.parents):
            alt = base / "code"
            if alt.is_dir():
                found = sorted(alt.rglob("*.py"))
                if found:
                    gens = found
                    break
    _LAYER2["scanned"] = 0
    _LAYER2["identifiers_skipped"] = 0
    for py in gens:
        if py.name in EXCLUDE_SOURCES:
            continue
        _LAYER2["scanned"] += 1
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError as e:
            hits.append({"layer": f"gen:{py.name}", "variant": "SYNTAX ERROR",
                         "context": str(e)})
            continue
        # The registry bans reader-facing WORDING. A string constant with no
        # whitespace is a name -- a JSON key, a field, a path, a filename --
        # and a machine-readable artifact is allowed to keep the key it has
        # always had. Widening this layer to the shipped experiments/ scripts
        # surfaced exactly one hit, "decoy_false_positives", which is such a
        # key. Names are counted, not silently dropped: a narrowing nobody can
        # see is how a scope stops meaning what it says.
        consts = [n.value for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        prose = [c for c in consts if any(ch.isspace() for ch in c.strip())]
        _LAYER2["identifiers_skipped"] += len(consts) - len(prose)
        _scan(f"gen:{py.name}", " \n ".join(prose), pats, hits)
    for name in ("README.md", "CHANGE_SUMMARY.md"):
        p = STAGING / name
        if p.exists():
            _scan(f"staging:{name}", p.read_text(), pats, hits)
    return hits


def scan_swallowed_prose(files=None) -> list:
    """A `%` comment eats the rest of its line, including any prose after it.

    Found the hard way: a provenance comment was appended after a sentence that
    continued on the same line, and four words of the manuscript vanished from
    the PDF while every other layer stayed green -- the tex still held them,
    the terminology scan strips comments before looking, and no number moved.
    Only a reader would have caught it.

    The rule is lexical and narrow: after a `%`, prose is text that looks like
    running English -- a capitalized word followed by two more lowercase words.
    A bare source pointer, a TODO, a bibkey or a measurement never matches it.
    """
    bad: list = []
    for f in (files or TEX_FILES):
        lines = f.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            m = re.search(r"(?<!\\)%", line)
            if not m:
                continue
            tail = line[m.end():]
            if not line[:m.start()].strip():
                # A WHOLE-line comment usually swallows nothing -- multi-line
                # prose comments put % on every line. But it does swallow when
                # a sentence was joined onto its end: the tail then breaks off
                # mid-clause and the NEXT line carries on in lowercase, with no
                # % of its own. That is the exact shape this guard missed twice
                # (SS15.4 and SS15.6 in the seventh review's wording batch), so
                # it is the shape it now looks for.
                nxt = lines[i] if i < len(lines) else ""     # i is 1-indexed
                if not (len(tail.split()) >= 4
                        and not tail.rstrip().endswith((".", ":", ";", ")", "?",
                                                        "!", "-"))
                        and nxt.lstrip()[:1].islower()
                        and not nxt.lstrip().startswith("%")):
                    continue
            # [A-Z][a-z]* not [A-Z][a-z]+: the sentence this guard failed to
            # catch the second time began with the single letter "A".
            if re.search(r"(?:^|\s)[A-Z][a-z]*(?:\s+[a-z]+){2}", tail):
                bad.append({"layer": f"swallowed:{f.name}", "variant": f"line {i}",
                            "context": line.strip()[:160]})
                continue
            # ...and the third shape, found in P5R-S: only ONE word was
            # swallowed. "Allowing" was pulled onto the end of a comment and
            # the sentence resumed, lowercase, on the next line -- too short
            # for the three-word test above, and invisible in every other
            # layer. A capitalized word ending a comment, with a lowercase
            # continuation under it, is a sentence that lost its opening.
            nxt = lines[i] if i < len(lines) else ""
            last = tail.split()[-1] if tail.split() else ""
            if (re.fullmatch(r"[A-Z][a-z]+", last)
                    and nxt.lstrip()[:1].islower()
                    and not nxt.lstrip().startswith("%")):
                bad.append({"layer": f"swallowed:{f.name}",
                            "variant": f"line {i} (stranded sentence opener)",
                            "context": line.strip()[-90:]})
    return bad


def scan_pdf(pdfs=None, pats=None) -> list:
    """Layer 3: rendered text of both PDFs, whitespace-normalized."""
    pats = pats or _load_registry()
    hits: list = []
    for pdf in (pdfs or PDF_FILES):
        pdf = Path(pdf)
        if not pdf.exists():
            hits.append({"layer": f"pdf:{pdf.name}", "variant": "MISSING PDF",
                         "context": str(pdf)})
            continue
        txt = subprocess.run(["pdftotext", str(pdf), "-"],
                             capture_output=True, text=True).stdout
        _scan(f"pdf:{pdf.name}", txt, pats, hits)
    return hits


def _close(printed: str, actual: float) -> bool:
    """Printed decimal equals actual within half an ulp of its precision."""
    m = re.fullmatch(r"[+-]?\d+\.(\d+)", printed.strip())
    if not m:
        return False
    return abs(float(printed) - actual) <= 0.5 * 10 ** -len(m.group(1)) + 1e-12


def numeric_checks() -> list:
    """Abstract macros / faithfulness table / Fig-3 snapshot vs t_final.json."""
    errs: list = []
    tf = json.loads(T_FINAL.read_text())
    fai = tf["faithfulness"]

    macros = (CODE_ROOT / "reporting" / "out" / "faith_macros.tex").read_text()

    def macro(name):
        m = re.search(r"\\newcommand\{\\" + name +
                      r"\}\{((?:[^{}]|\{[^{}]*\})*)\}", macros)
        return m.group(1) if m else None

    for name, val in [("faithTMimic", fai["MIMIC"]["T"]),
                      ("faithTEicu", fai["eICU"]["T"]),
                      ("faithHolmMimic", fai["MIMIC"]["p_holm"]),
                      ("faithHolmEicu", fai["eICU"]["p_holm"])]:
        got = macro(name)
        if got is None or not _close(got, val):
            errs.append(f"macro {name}={got!r} vs t_final {val}")
    npe = macro("nopriorPExact")
    m = re.fullmatch(r"([\d.]+)\\times10\^\{(-?\d+)\}", (npe or "").strip())
    floor = tf["noprior_faithfulness"]["MIMIC"]["p_exact"]
    if not m or abs(float(m.group(1)) * 10 ** int(m.group(2)) - floor) > 0.05 * floor:
        errs.append(f"macro nopriorPExact={npe!r} vs t_final {floor}")

    tab = (CODE_ROOT / "reporting" / "out" / "tab_faithfulness.tex").read_text()
    for ds in ("MIMIC", "eICU"):
        want = f"{fai[ds]['T']:+.3f}"
        if want not in tab:
            errs.append(f"tab_faithfulness missing T {ds} {want}")

    prov = json.loads((CODE_ROOT / "reporting" / "out" /
                       "Fig_scoreboard.provenance.json").read_text())
    snap = prov["snapshot"]
    rec_sym = tf["recovery"]["probe_vs_D_same_host_symmetric"]
    if not math.isclose(snap.get("probe_recovery_p_exact", -1),
                        rec_sym["p_exact"]):
        errs.append("snapshot probe_recovery_p_exact vs t_final")
    if not math.isclose(snap.get("probe_vs_D_symmetric_T", -99),
                        rec_sym["T"]):
        errs.append("snapshot probe_vs_D_symmetric_T vs t_final")
    # The superseded pair stays in t_final and stays marked. If the mark ever
    # falls off, the manuscript has two live recovery numbers and no way to
    # tell which one a reader is looking at.
    sup = tf["recovery"]["probe_vs_D_retrained"]
    if sup.get("superseded_by") != "probe_vs_D_same_host_symmetric":
        errs.append("probe_vs_D_retrained lost its superseded mark")
    if snap.get("conf_D") != tf["leakage"]["batches"]["confirmatory"]["interaction"]["SNI-D"]:
        errs.append("snapshot conf_D vs t_final")

    # Build-order staleness: t_final copies the cost display strings from
    # a3_macros, and Fig. 3 prints that copy. If the macros are regenerated
    # afterwards the figure keeps a superseded headline while the prose
    # carries the new one -- a contradiction visible on one reading of the
    # PDF that no lexical scan can see. Assert the copy is current.
    a3 = (CODE_ROOT / "reporting" / "out" / "a3_macros.tex").read_text()
    cost = tf.get("scoreboard_desc", {}).get("cost_display", {})
    for mname, key in (("costRatioSTRange", "single_thread_range"),
                       ("costRatioRange", "grid_range")):
        src = a3 if mname.endswith("STRange") else (
            CODE_ROOT / "reporting" / "out" / "cost_macros.tex").read_text()
        mm = re.search(r"\\newcommand\{\\" + mname
                       + r"\}\{((?:[^{}]|\{[^{}]*\})*)\}", src)
        want = mm.group(1) if mm else None
        if want is not None and cost.get(key) != want:
            errs.append(f"t_final cost_display.{key}={cost.get(key)!r} vs "
                        f"{mname}={want!r} (rebuild t_final after the cost "
                        f"macros, then Fig. 3)")

    # Frozen artifact keys must not reach a rendered table cell. The registry
    # says how each is displayed; a generator that prints the raw key instead
    # produces a table naming the same object two ways on facing pages (the
    # recovery/stability tables printed "P-alone" and "P" beside tables that
    # printed the display term -- found by the manual page inspection of the
    # fourth-review round, P5R-K SS4.2).
    reg = json.loads(REGISTRY.read_text())
    keys = [k for k in reg.get("data_key_display", {}) if not k.startswith("_")]
    cellpat = [(k, re.compile(r"(?:^|&|/)\s*" + re.escape(k)
                              + r"\s*(?:&|\\\\|/|$)", re.M)) for k in keys]
    for f in sorted((CODE_ROOT / "reporting" / "out").glob("*.tex")):
        body = "\n".join(ln for ln in f.read_text().splitlines()
                          if not ln.lstrip().startswith("%"))
        for k, pat in cellpat:
            if pat.search(body):
                errs.append(f"{f.name}: frozen data key {k!r} reaches a "
                            f"rendered cell -- render it through "
                            f"reporting/termmap.data_display")
    return errs



# --------------------------------------------------------------------- #
# Orthography: American canonical (P7-A SS1)
# --------------------------------------------------------------------- #
#: A SPELLING is not a WORDING. The registry's `terms` record claims we
#: withdrew, and a hit there means the manuscript says something we decided it
#: must not say. A hit here means the same word is spelled two ways in one
#: document, which is a copy-editing defect and nothing more -- so it is
#: reported as its own kind rather than folded into the variant count, and the
#: message names the canonical form instead of quoting a retired claim.
_ORTHO = {"pat": None, "map": {}}


def _registry_path() -> Path:
    """The registry, wherever this run can see it.

    REGISTRY is CODE_ROOT/docs/terminology_registry.json, and CODE_ROOT is
    rebound by --from-package to gate_inputs/code_SNI -- which does hold it.
    But a script run from INSIDE the package (gate 17 runs the shipped
    selftests) has CODE_ROOT = <pkg>/code, where docs/ holds only the release
    procedure. That crashed perm_sni_inventory's selftest, which needs the
    orthography map to compare sites. Look in the places a package puts it.
    """
    for c in (REGISTRY,
              CODE_ROOT.parent / "gate_inputs" / "code_SNI" / "docs"
              / "terminology_registry.json",
              CODE_ROOT.parent / "evidence" / "terminology_registry.json"):
        if c.exists():
            return c
    return REGISTRY


def _ortho():
    if _ORTHO["pat"] is None:
        reg = json.loads(_registry_path().read_text()).get("orthography", {})
        m = {}
        for us, brs in reg.get("map", {}).items():
            for br in brs:
                m[br.lower()] = us
        _ORTHO["map"] = m
        _ORTHO["pat"] = (re.compile(r"\b(" + "|".join(
            sorted(m, key=len, reverse=True)) + r")\b", re.I)
            if m else re.compile(r"(?!x)x"))
    return _ORTHO["pat"], _ORTHO["map"]


#: The reference list quotes other people's titles. "A Study of K-Nearest
#: Neighbour as an Imputation Method" is what that paper is CALLED; respelling
#: it to match our house style would be a misquotation, and a gate that
#: demanded it would be demanding one. The rendered-PDF layer therefore stops
#: where the bibliography starts.
_BIB_HEAD = re.compile(r"\bReferences\b")


def scan_orthography(texts: dict = None) -> list:
    """British spellings in the layers P7-A SS1 declares American."""
    pat, m = _ortho()
    if texts is None:
        texts = {}
        for f in TEX_FILES:
            if f.exists():
                texts[f"tex:{f.name}"] = _strip_tex_comments(f.read_text())
        gens = sorted((CODE_ROOT / "reporting").glob("*.py")) + \
            sorted((CODE_ROOT / "experiments").glob("*.py"))
        for py in gens:
            if py.name in EXCLUDE_SOURCES:
                continue
            try:
                tree = ast.parse(py.read_text())
            except SyntaxError:
                continue
            prose = [n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant)
                     and isinstance(n.value, str)
                     and any(c.isspace() for c in n.value.strip())]
            if prose:
                texts[f"gen:{py.name}"] = " \n ".join(prose)
        for f in PDF_FILES:
            if not f.exists():
                continue
            t = subprocess.run(["pdftotext", str(f), "-"],
                               capture_output=True, text=True).stdout
            cut = list(_BIB_HEAD.finditer(t))
            if cut:
                t = t[:cut[-1].start()]
            texts[f"pdf:{f.name}"] = t
        #: The response letter is SUBMITTED, and it was in none of the layers
        #: above -- not a manuscript source, not a generator, not one of the
        #: two PDFs. It carried eleven British spellings for a full round and
        #: the manuscript gate never looked at it; the public-tree scan found
        #: them. A document that goes to the journal belongs in the gate that
        #: says what the journal receives.
        for rel in ("VERA_response/RESPONSE_TO_REVIEWERS_R1.md",
                    "VERA_response/COVER_LETTER_R1.md",
                    "reporting/response_letter_body.md",
                    "reporting/cover_letter_body.md"):
            f = (CODE_ROOT / rel if rel.startswith("reporting/")
                 else CODE_ROOT.parent / rel)
            if f.exists():
                texts[f"letter:{f.name}"] = f.read_text()
    exempt = _asset_words()
    hits = []
    for label, t in texts.items():
        t = _norm(t)
        for mt in pat.finditer(t):
            w = mt.group(0)
            #: Only the rendered page can carry a delivered asset's words; the
            #: sources cannot, and are never exempt. See _asset_words.
            if label.startswith("pdf:") and w in exempt:
                continue
            hits.append({"layer": label, "found": w,
                         "canonical": m[w.lower()],
                         "context": t[max(0, mt.start() - 40):mt.end() + 40]})
    return hits


def _asset_words() -> set:
    """British words printed INSIDE a registered figure asset, re-derived.

    Fig. 1 is a delivered PDF whose text lives in a PowerPoint file the
    designer owns. Respelling it means a re-export, so those words appear on
    the rendered page and in no source we control.

    The exempt set is extracted from the delivered file, never read from the
    registry's prose -- an exemption you can widen by editing a list is not an
    exemption, it is a switch. It is also keyed to the asset's sha256: a new
    export is a different file, the grant does not transfer to it, and the
    gate goes red until someone looks at the new one.

    The loophole this could have been is closed elsewhere: the SOURCE layers
    have no exemption at all, so body text that reintroduced one of these
    words would stop the compile before a PDF existed.
    """
    reg = CODE_ROOT / "docs" / "figure_assets.json"
    ex = json.loads(REGISTRY.read_text()).get(
        "scan_exemptions", {}).get("orthography", {}).get(
            "delivered_asset")
    if not (ex and reg.exists()):
        return set()
    import hashlib
    pat, _ = _ortho()
    out: set = set()
    for a in json.loads(reg.read_text()).get("assets", []):
        if a["figure"] != ex.get("figure"):
            continue
        live = None
        for c in (CODE_ROOT.parent / a["path"], PAPER / Path(a["path"]).name):
            if c.exists():
                live = c
                break
        if live is None:
            return set()
        if hashlib.sha256(live.read_bytes()).hexdigest() != \
                ex.get("granted_sha256"):
            return set()          # a different file; the grant is not for it
        txt = subprocess.run(["pdftotext", str(live), "-"],
                             capture_output=True, text=True).stdout
        out |= {mt.group(0) for mt in pat.finditer(_norm(txt))}
    return out

def run(include_pdf: bool = True) -> dict:
    pats = _load_registry()
    hits = scan_sources(pats)
    if include_pdf:
        hits += scan_pdf(pats=pats)
    errs = numeric_checks()
    sp = scan_orthography() if include_pdf else scan_orthography(
        {f"tex:{f.name}": _strip_tex_comments(f.read_text())
         for f in TEX_FILES if f.exists()})
    return {"registry_version": json.loads(REGISTRY.read_text())["version"],
            "n_variant_hits": len(hits), "hits": hits,
            "n_spelling_hits": len(sp), "spelling_hits": sp[:20],
            #: Layer 2's real reach, printed. It read nothing at all in
            #: package mode for several rounds while the gate reported
            #: "five layers"; a scope is only checked if it is stated.
            "n_generators_scanned": _LAYER2["scanned"],
            "n_identifier_constants_skipped":
                _LAYER2["identifiers_skipped"],
            "n_numeric_errors": len(errs), "numeric_errors": errs,
            "pass": not hits and not errs and not sp}


def _selftest() -> int:
    ok = True

    # The cross-boundary false positive, on its own reproduction case: two
    # comment lines from package_layout.py that between them contain no banned
    # phrase, but whose naive join does. Ruled 2026-09-02: fix the parser, not
    # the comment.
    _repro = ("        # code, so a package handed to a reviewer\n"
              "        # with no repository could not produce the green line\n")
    _joined_raw = _norm(_repro)
    _joined_fixed = _norm(strip_comment_markers(_repro, ".py"))
    print(("[PASS] " if "reviewer #" in _joined_raw.lower() else "[FAIL] ")
          + "fixture reproduces the old cross-boundary join")
    print(("[PASS] " if "reviewer #" not in _joined_fixed.lower() else "[FAIL] ")
          + "stripping the marker removes it")
    print(("[PASS] " if "with no repository" in _joined_fixed else "[FAIL] ")
          + "the comment PROSE is still scanned, only the marker is gone")
    _kept = _norm(strip_comment_markers("# a banned decoy word here\n", ".py"))
    print(("[PASS] " if "decoy" in _kept else "[FAIL] ")
          + "a banned word inside a comment is still visible to the scan")

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    pats = _load_registry()
    hits: list = []
    _scan("t", "the Decoy   column and the matched-FPR rule", pats, hits)
    check(any("decoy" == h["variant"] for h in hits), "case variant caught")
    check(any(h["variant"] == "matched FPR" for h in hits),
          "hyphen/space family caught")
    hits2: list = []
    _scan("t", "the dataset dominated the ranking", pats, hits2)
    check(any("dominated" in h["variant"] for h in hits2), "word-boundary term caught")
    hits3: list = []
    _scan("t", "predominated is fine? no: word boundary", pats, hits3)
    check(not any("dominated" in h["variant"] for h in hits3),
          "no substring false-positive on word-boundary term")
    hits4: list = []
    _scan("t", _strip_tex_comments("clean text % decoy in comment"), pats, hits4)
    check(not hits4, "tex comments stripped")
    check(_close("-0.030", -0.0301) and not _close("-0.030", -0.045),
          "half-ulp closeness")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.tex"
        good.write_text("note). % src: t_final#recovery.probe\nThe next line.\n"
                        "50\\% of rows were held out.\n")
        check(scan_swallowed_prose([good]) == [], "clean src comments pass")
        bad = Path(d) / "bad.tex"
        bad.write_text("settings. % src: t_final#x The comparison that bears\n")
        bad2 = Path(d) / "bad2.tex"
        bad2.write_text("matrix. % src: t51_redundancy A possible mechanism\n")
        check(len(scan_swallowed_prose([bad2])) == 1,
              "a swallowed sentence starting with a one-letter word is caught")
        bad3 = Path(d) / "bad3.tex"
        bad3.write_text("% seventh review SS15.6 verbatim The mechanism "
                        "argument weakens accordingly: it rests on\n"
                        "the third and fourth observations together with\n")
        check(len(scan_swallowed_prose([bad3])) == 1,
              "a sentence joined onto a WHOLE-line comment is caught")
        ok3 = Path(d) / "ok3.tex"
        ok3.write_text("% P5R-O: drawn at its printed size the figure is\n"
                       "% tall. That is more than the column admits.\n"
                       "The float therefore takes a page of its own.\n")
        check(scan_swallowed_prose([ok3]) == [],
              "a real multi-line prose comment is still not flagged")
        whole = Path(d) / "whole.tex"
        whole.write_text("% P5R-O: drawn at its printed size the figure is\n"
                         "% tall. That is more than the column admits.\n")
        check(scan_swallowed_prose([whole]) == [],
              "a whole-line comment is prose by design, not swallowed prose")
        check(len(scan_swallowed_prose([bad])) == 1,
              "prose swallowed by a trailing comment is caught")
    # --- orthography (P7-A SS1.2) ------------------------------------- #
    # The three things this layer has to get right, and the one it must NOT
    # do. The last fixture is the important one: it is the false positive that
    # would have made the gate untrustworthy on its first run.
    o = scan_orthography({"tex:fix.tex": "the behavioural readout was "
                                         "operationalised and labelled"})
    check(len(o) == 3, f"British forms caught in a source layer ({len(o)})")
    check({h["canonical"] for h in o}
          == {"behavioral", "operationalized", "labeled"},
          "each hit names the canonical American form, not just the offence")
    check(scan_orthography({"tex:fix.tex": "the behavioral readout was "
                                           "operationalized and labeled"}) == [],
          "the American forms are silent")
    check(scan_orthography(
        {"tex:fix.tex": "sensitivity analyses accompany the primary analyses"}
    ) == [], "'analyses' is the plural of 'analysis' in BOTH varieties and is "
             "NOT a British form -- flagging it would have been nine false "
             "hits in the main text alone")
    #: The asset exemption was WITHDRAWN on 2026-09-04, when the re-export
    #: made it unnecessary. The fixture asserts the WITHDRAWAL, not the grant:
    #: an exemption nobody needs must come back empty rather than absent, so
    #: the mechanism that would grant one -- deriving the exempt set from the
    #: delivered file at gate time -- is still the mechanism a future asset
    #: would go through.
    check(_asset_words() == set(),
          "no delivered asset needs a spelling exemption; the derived set is "
          "EMPTY, which is not the same as the check being gone")
    _rex = json.loads(_registry_path().read_text()).get(
        "scan_exemptions", {}).get("orthography", {})
    check("delivered_asset" not in _rex
          and "delivered_asset_WITHDRAWN" in _rex,
          "the withdrawal is recorded in the registry, not merely enacted")

    errs = numeric_checks()
    check(errs == [], f"numeric consistency on live artifacts ({errs[:2]})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sources-only", action="store_true",
                    help="pre-compile mode: skip the PDF layer")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = run(include_pdf=not a.sources_only)
    for h in out["hits"][:40]:
        print(f"[HIT] {h['layer']} :: {h['variant']} :: ...{h['context']}...")
    for e in out["numeric_errors"]:
        print(f"[NUM] {e}")
    print(json.dumps({k: out[k] for k in
                      ("registry_version", "n_variant_hits",
                       "n_numeric_errors", "pass")}))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
