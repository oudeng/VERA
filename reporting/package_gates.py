"""Packaging gates for an internal-review bundle.

Nothing ships unless every gate is green. Each gate is an executable
check over the live tree, not a claim in a receipt; the report this
emits is the receipt's evidence. `all_green` is reported ONLY in package
mode (--from-package), because the thing that has to be verified is the
file that gets delivered, not the tree it was built from.

NINETEEN gates. Each was added by a specific failure, and the docstring on
each one names it; the list below is a map, not the specification.

  1  P0-A       single evidence source consistent; every generated output
                stopped at the same clean commit
  2  P0-B-2     downstream independence, per protocol class
  3  P0-B-4     real-pattern-inspired masking named and documented
  4  references DOI/venue verification table present, no placeholders left
  5  six-state  the README status table is machine-generated and honest
  6  pending    only the publication placeholders remain
  7  facts      five-layer single-facts gate: zero banned variants, zero
                numeric drift, and every layer's reach printed  (P5R-H SS0)
  8  rules      the rule-document count in the ESM equals the archive
                manifest's, and every document in it is named
  9  pages      every page count and MD5 a package document states equals
                the real one
 10  dates      no file carries a self-declared date later than the
                package's own, absent a declared exemption
 11  deletions  everything the response letter declares deleted has zero
                hits in BOTH rendered PDFs
 12  figures    no stray figure copy; every figure newer than t_final; no
                withdrawn wording in a rendered figure   (P5R-K SS7.3/7.4)
 13  bibliography  every entry carries one verifiable identifier, every
                reference is VERIFIED, every citation POINT is supported
 14  closed     every "Closed" state resolves to a path inside the package
 15  inventory  the site inventory re-derives here and matches the shipped
                artifact
 16  digests    duplicate basenames census: same name, different content
                is a blocker
 17  scripts    cited scripts by STRENGTH of claim -- selftest / executed /
                syntax-checked / parse-checked, never merged
 18  quarantine the retired verdict label appears only where it is allowed,
                searched over every text file in the package
 19  transcript the GATES.txt that ships is the transcript of THIS build
 20  submission  VERA_paper_R1 is byte-identical to paper_R1 (working
                tree only; the view is not shipped)

    env PYTHONHASHSEED=2025 python reporting/package_gates.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
from pathlib import Path

#: Inspecting the package must not MODIFY the package. Bytecode is off for
#: this process and every subprocess it starts -- but that is one type, and
#: naming types was the mistake. When .pyc stopped appearing, two other writes
#: went on unnoticed for a further round: the gates' working view was being
#: assembled at pkg/_paper (twelve files inside the audited tree), and gate 15
#: wrote its re-derived inventory into whichever checkout the script was
#: driven from. The guarantee that actually holds is the outer invariant at
#: the bottom of this file: the tree is hashed before the run and after it,
#: and any difference at all is red. This line is only true because of that.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
PAPER = ROOT / "paper_R1"


def _pf(name: str) -> Path:
    """The manuscript file `name` under whichever PAPER is in force.

    PAPER is rebound by --from-package to a FLAT assembled view, and in the
    working tree it is a two-directory split (P7-A SS2). The gates ask by
    basename and this answers for both, so a gate cannot be right about one
    layout and silently wrong about the other.
    """
    sys.path.insert(0, str(_REAL_CODE_ROOT / "experiments"))
    from package_layout import paper_file
    return paper_file(name, PAPER)
OUTDIR = CODE_ROOT / "reporting" / "out"
STAGING = ROOT / "internal_review" / "ir_staging"

#: Which tree the gates are reading. "live" is the working repository and is
#: for development; "package" is an unpacked ZIP and is the ONLY mode whose
#: all_green counts (P5R-N SS5.2). Printed with every result, so a green line
#: can never be mistaken for the wrong one.
MODE = "live"

#: The gates' CODE is always this repository's; only the DATA moves when the
#: mode changes. Captured before any rebinding so an import never follows a
#: rebound path into a package that holds no modules.
_REAL_CODE_ROOT = CODE_ROOT


def rebind_to_package(pkg: Path) -> dict:
    """Point the gates at an unpacked review package instead of the repo.

    The package is shaped for a reader; experiments/package_layout.py holds the
    one mapping from that shape to the repository shape these gates were
    written against. Nothing outside the package is read afterwards -- which is
    the point: a gate that passes here passes on what was actually delivered.
    """
    global CODE_ROOT, ROOT, PAPER, OUTDIR, STAGING, MODE
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "experiments"))
    from package_layout import roots, materialise_paper
    #: The working view goes OUTSIDE the audited tree. Assembling it at
    #: pkg/_paper put twelve files into the package the gates were about to
    #: read, so "inspecting the package must not modify the package" was false
    #: at the first line of the run.
    paper_view = materialise_paper(pkg)
    r = roots(pkg, paper_view)
    ROOT, CODE_ROOT = r["ROOT"], r["CODE_ROOT"]
    PAPER, OUTDIR, STAGING = r["PAPER"], r["OUTDIR"], r["STAGING"]
    MODE = "package"
    # The gates that delegate (5, 7, 13) import modules which bound their own
    # roots at import time, from wherever the SCRIPT sits. In the repository
    # those two happen to agree; inside a package they do not, and the
    # delegate then reads the checkout the reviewer ran the script from --
    # which is exactly what package mode exists to prevent. Rebind them from
    # the same mapping, and fail loudly if a module gains a root this does
    # not know about.
    sys.path.insert(0, str(_REAL_CODE_ROOT))
    from reporting import status_table, facts_gate, bib_gate, bib_inventory
    bind = {
        status_table: {"ROOT": r["ROOT"], "CODE_ROOT": r["CODE_ROOT"],
                       "PAPER": r["PAPER"], "OUTDIR": r["OUTDIR"],
                       "STAGING": r["STAGING"],
                       "DECL": r["CODE_ROOT"] / "docs"
                               / "ir_status_declarations.json",
                       "INSPECTION_LOG": r["STAGING"] / "INSPECTION_LOG.md"},
        facts_gate: {"CODE_ROOT": r["CODE_ROOT"], "PAPER": r["PAPER"],
                     "REGISTRY": r["CODE_ROOT"] / "docs"
                                 / "terminology_registry.json",
                     "T_FINAL": r["CODE_ROOT"] / "results" / "T5_stats"
                                / "t_final.json",
                     "TEX_FILES": [r["PAPER"] / "paperY_main.tex",
                                   r["PAPER"] / "paperY_ESM.tex"],
                     "PDF_FILES": [r["PAPER"] / "paperY_main.pdf",
                                   r["PAPER"] / "paperY_ESM.pdf"],
                     "STAGING": r["STAGING"]},
        bib_gate: {"CODE_ROOT": r["CODE_ROOT"], "ROOT": r["ROOT"],
                   "PAPER": r["PAPER"],
                   "BIB": r["PAPER"] / "references_Y.bib",
                   "BBL": r["PAPER"] / "paperY_main.bbl",
                   "VERIF": r["ROOT"] / "reports"
                            / "citation_verification.json",
                   "SUPPORT": r["ROOT"] / "reports" / "citation_support.json",
                   "OPEN": r["ROOT"] / "internal_review"
                           / "bib_open_adjudications.json"},
        bib_inventory: {"CODE_ROOT": r["CODE_ROOT"], "ROOT": r["ROOT"],
                        "PAPER": r["PAPER"],
                        "BIB": r["PAPER"] / "references_Y.bib",
                        "BBL": r["PAPER"] / "paperY_main.bbl",
                        "OUT": r["ROOT"] / "reports" / "bib_inventory.json",
                        "TEXS": [r["PAPER"] / "paperY_main.tex",
                                 r["PAPER"] / "paperY_ESM.tex"],
                        "BST": r["PAPER"] / "sn-vancouver-num.bst"},
    }
    def _is_root(v) -> bool:
        # a bare Path, or a container of them (facts_gate.TEX_FILES was a
        # list, and a list slipped past the first version of this check)
        return isinstance(v, Path) or (
            isinstance(v, (list, tuple, set))
            and any(isinstance(x, Path) for x in v))

    for mod, names in bind.items():
        have = {k for k, v in vars(mod).items()
                if _is_root(v) and k.isupper()}
        unknown = have - set(names)
        if unknown:
            raise RuntimeError(
                f"{mod.__name__} carries path root(s) {sorted(unknown)} that "
                f"package mode does not rebind; a gate would read the "
                f"checkout instead of the package")
        for k, v in names.items():
            setattr(mod, k, v)
    return r


def _sh(cmd: list, cwd: Path = None) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True,
                          text=True).stdout


def _recorded_head() -> str:
    """In package mode there is no git; the build records what it built from."""
    f = STAGING / "BUILD.json"
    if f.exists():
        try:
            return json.loads(f.read_text()).get("head_commit", "")
        except Exception:
            return ""
    return ""


def gate_1_p0a() -> dict:
    """Single source consistent; every consumed output at a clean commit."""
    detail, ok = [], True
    head = (_recorded_head() if MODE == "package"
            else _sh(["git", "rev-parse", "HEAD"], CODE_ROOT).strip())
    dirty = [p.name for p in OUTDIR.glob("*.tex")
             if "-dirty" in p.read_text()[:600]]
    stale = sorted({m for p in OUTDIR.glob("*.tex")
                    for m in re.findall(r"code_SNI commit: ([0-9a-f]{7,40})",
                                        p.read_text()[:600])
                    if m != head})
    detail.append(f"HEAD={head[:8]}; outputs with -dirty: {len(dirty)}; "
                  f"outputs at another commit: {stale[:3]}")
    ok = ok and not dirty and not stale
    tf = CODE_ROOT / "results" / "T5_stats" / "t_final.json"
    # The selftest is CODE and runs as a subprocess. It used to run from the
    # repository the script happened to sit in, which made the package's
    # green line depend on a checkout the reader does not have; the script
    # ships, so it runs from wherever the gates are running.
    r = subprocess.run([sys.executable, "experiments/t_final.py", "--selftest"],
                       cwd=_REAL_CODE_ROOT, capture_output=True, text=True)
    sel = "SELFTEST PASS" in r.stdout
    detail.append(f"t_final selftest: {'PASS' if sel else 'FAIL'}; "
                  f"artifact present: {tf.exists()}")
    ok = ok and sel and tf.exists()
    return {"gate": "1 P0-A single source + clean provenance",
            "pass": ok, "detail": "; ".join(detail)}


def gate_2_independence() -> dict:
    p = CODE_ROOT / "results" / "T4_downstream" / "smoke_independence.json"
    if not p.exists():
        return {"gate": "2 P0-B-2 downstream independence", "pass": False,
                "detail": "smoke_independence.json absent"}
    d = json.loads(p.read_text())
    per = {m: (r["independent_per_class"]
               and r["L_label_cannot_reach_imputer"])
           for m, r in d["methods"].items()}
    ok = d["verdict"] == "ALL-GREEN" and all(per.values())
    return {"gate": "2 P0-B-2 downstream independence", "pass": ok,
            "detail": f"verdict={d['verdict']}; per-method green={per}"}


def gate_3_realpattern(layers: dict) -> dict:
    named = "real-pattern-inspired masking" in layers["pdf"].lower()
    seven = sum(1 for k in ("Where the real pattern comes from",
                            "Which rows and columns are scorable",
                            "Pre-masking observedness",
                            "Natural missing cells",
                            "Pattern-to-table mapping",
                            "Coverage and scored sample size",
                            "Selection bias")
                if k in layers["pdf"])
    ok = named and seven == 7
    return {"gate": "3 P0-B-4 real-pattern-inspired masking", "pass": ok,
            "detail": f"name unified: {named}; construction questions "
                      f"answered in the PDF: {seven}/7"}


def gate_4_references() -> dict:
    bbl = _pf("paperY_main.bbl")
    n = len(re.findall(r"\\bibitem", bbl.read_text())) if bbl.exists() else 0
    reg = ROOT / "reports" / "P5R_G_citation_registration.md"
    unver = 0
    if reg.exists():
        unver = reg.read_text().count("!!")
    ok = n >= 50 and reg.exists() and unver == 0
    return {"gate": "4 references verified", "pass": ok,
            "detail": f"cited entries: {n}; registration table: "
                      f"{reg.exists()}; unverified rows: {unver}"}


def gate_5_status_table() -> dict:
    md = STAGING / "STATUS_TABLE.md"
    aud = STAGING / "STATUS_TABLE.audit.json"
    if not (md.exists() and aud.exists()):
        return {"gate": "5 six-state status table", "pass": False,
                "detail": "status table not generated"}
    a = json.loads(aud.read_text())
    text = md.read_text()
    # an honest table never claims a blanket closure
    blanket = [p for p in ("all items closed", "全部关闭", "all closed",
                           "everything is closed")
               if p.lower() in text.lower()]
    ok = a["n_items"] > 0 and not blanket
    rebuilt = ""
    if MODE == "package":
        # Twelfth review P0-5, and its residue, closed here. The audit JSON
        # was rebuilt beside the markdown and then never read -- so the half a
        # human reads was checked and the half a MACHINE reads, the one
        # carrying `passed: true` per row, was not. Both are compared now, by
        # bytes, using the script's own --check path so the gate and a reader
        # running the command get the same answer.
        sys.path.insert(0, str(_REAL_CODE_ROOT))
        from reporting import status_table
        try:
            r = status_table.check(md, aud)
            rebuilt = f"; package-mode rebuild -- {r['detail']}"
            ok = ok and r["pass"]
        except Exception as exc:                        # pragma: no cover
            rebuilt = (f"; package-mode rebuild FAILED: "
                       f"{type(exc).__name__}: {exc}")
            ok = False
    return {"gate": "5 six-state status table", "pass": ok,
            "detail": f"rows={a['n_items']}, auto-downgraded="
                      f"{a['n_downgraded']}, blanket claims={blanket}"
                      f"{rebuilt}"}


def gate_6_pending() -> dict:
    pend = {}
    for f in ("paperY_main.tex", "paperY_ESM.tex"):
        # comments describe the mechanism; only rendered placeholders count
        body = re.sub(r"(?<!\\)%.*", "", _pf(f).read_text())
        for m in re.findall(r"\\pending\{([^}]*)\}", body):
            pend[m] = pend.get(m, 0) + 1
    allowed = {"REPO-URL", "SWHID"}
    extra = {k: v for k, v in pend.items() if k not in allowed}
    # Same rule as above, applied to the .bib as well: a placeholder counts
    # when it would REACH THE READER. A comment line describing a placeholder
    # that was resolved -- which is how the Kim2024ehratt provenance note
    # reads -- is documentation, and counting it would mean the record of a
    # fixed defect keeps the gate red forever. Fields are still counted.
    def _uncommented(f: str) -> str:
        return re.sub(r"(?<!\\)%.*", "", _pf(f).read_text())
    cite = sum(_uncommented(f).count("CITE-VERIFY")
               for f in ("paperY_main.tex", "paperY_ESM.tex",
                         "references_Y.bib"))
    ok = not extra and cite == 0
    return {"gate": "6 placeholder inventory", "pass": ok,
            "detail": f"pending={pend}; disallowed={extra}; "
                      f"CITE-VERIFY={cite}"}


def gate_7_facts() -> dict:
    sys.path.insert(0, str(_REAL_CODE_ROOT))
    from reporting import facts_gate
    r = facts_gate.run(include_pdf=True)
    layers = sorted({h["layer"].split(":")[0] for h in r["hits"]})
    #: "Five layers" is a claim about REACH, and one of the five read nothing
    #: at all in package mode for several rounds -- the generator tree it
    #: globs does not exist in a package. A layer that scans zero files is a
    #: layer that is not running, so the count is printed and zero is red.
    n_gen = r.get("n_generators_scanned", 0)
    return {"gate": "7 single-facts gate (five layers)",
            "pass": r["pass"] and n_gen > 0,
            "detail": f"variant hits={r['n_variant_hits']} "
                      f"(layers: {layers or 'none'}); "
                      #: P7-A SS1.2 added a sixth reading of the same layers,
                      #: for spelling. It has to be printed here or a red gate
                      #: reports "variant hits=0; numeric errors=0" and gives
                      #: the reader no way to see what stopped it.
                      f"spelling hits={r.get('n_spelling_hits', 0)}"
                      + (" (" + ", ".join(
                          f"{h['found']}->{h['canonical']} in {h['layer']}"
                          for h in r.get("spelling_hits", [])[:4]) + ")"
                         if r.get("n_spelling_hits") else "")
                      + f"; numeric errors="
                      f"{r['n_numeric_errors']}; generator files read by "
                      f"layer 2: {n_gen} "
                      f"({r.get('n_identifier_constants_skipped', 0)} "
                      f"identifier constants skipped as names, not prose)"
                      + (" -- ZERO, so this layer did not run" if not n_gen
                         else "")
                      + f"; registry {r['registry_version']}"}


def gate_8_rule_bundle() -> dict:
    """The rule-document count in the ESM equals the archive manifest's.

    The IR4 package said fourteen in two staging documents and sixteen in
    the ESM (fourth internal review, automation section): the manifest had
    grown by two and the prose had not. The number is now read from the
    newest manifest and every file in it must be named in the ESM.
    """
    arch = ROOT / "VERA_GitHub" / "prereg_archive"
    mans = sorted(arch.glob("manifest_*.json"))
    if not mans:
        return {"gate": "8 rule-bundle count", "pass": False,
                "detail": f"no manifest under {arch}"}
    man = json.loads(mans[-1].read_text())
    files = [Path(f["path"] if isinstance(f, dict) else f).stem
             for f in man["files"]]
    #: A manifest's own header must be possible. The 2026-08-30 manifest was
    #: assembled by hand from the 2026-08-29 one, header included, so it named
    #: a head_commit that PREDATED a file it listed by seven hours -- the
    #: archive claimed to have been built before one of its own members
    #: existed. Nothing read those fields, so nothing noticed. A provenance
    #: field nobody checks is decoration; either it holds, or it says it
    #: cannot be recovered and why.
    prov = []
    head = str(man.get("head_commit", ""))
    if head and head != "not recoverable":
        ht = _sh(["git", "log", "-1", "--format=%cI", head], CODE_ROOT).strip()
        for f in man["files"]:
            lc = str(f.get("last_commit", "")).split()
            if len(lc) < 2 or not ht:
                continue
            if lc[1] > ht:
                prov.append(f"{Path(f['path']).name} at {lc[0][:8]} ({lc[1]}) "
                            f"postdates the declared head_commit {head[:8]} "
                            f"({ht})")
    elif head == "not recoverable" and not man.get("_provenance_note"):
        prov.append("head_commit says 'not recoverable' with no "
                    "_provenance_note giving the reason")
    words = {14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
             18: "eighteen", 19: "nineteen", 20: "twenty"}
    esm = _pf("paperY_ESM.tex").read_text()
    want = words.get(len(files), str(len(files)))
    said = f"{want}\ndecision-rule documents" in esm or \
           f"{want} decision-rule documents" in " ".join(esm.split())
    unnamed = [f for f in files if f.replace("_", r"\_") not in esm]
    # the staging documents must not carry a different count
    other = []
    for name in ("README.md", "CHANGE_SUMMARY.md", "RESPONSE_TO_REVIEW_v3.md",
                 "CROSSCHECK_v3.md"):
        p_ = STAGING / name
        if not p_.exists():
            continue
        for n, w in words.items():
            if n == len(files):
                continue
            for form in (f"{n} 文档规则", f"{n} decision-rule",
                         f"{w} decision-rule"):
                if form in p_.read_text():
                    other.append(f"{name}:{form}")
    ok = said and not unnamed and not other
    return {"gate": "8 rule-bundle count", "pass": ok and not prov,
            "detail": f"manifest {mans[-1].name}: {len(files)} documents; "
                      f"ESM says {want}: {said}; unnamed in ESM: {unnamed}; "
                      f"conflicting counts in staging: {other}; manifest "
                      f"header: "
                      + ("declared not recoverable, with its reason recorded"
                         if head == "not recoverable" else
                         ("consistent with every last_commit it records"
                          if not prov else f"IMPOSSIBLE -- {prov}"))}


def gate_9_page_counts() -> dict:
    """Every page count a package document states equals the real one.

    The IR4 CROSSCHECK said 25 pp for a 26-page manuscript (fourth internal
    review, automation section). Any "<file> ... N pp" claim in a staging
    document is now checked against pdfinfo.
    """
    real = {}
    for f in ("paperY_main.pdf", "paperY_ESM.pdf"):
        p_ = _pf(f)
        if not p_.exists():
            continue
        m = re.search(r"Pages:\s*(\d+)",
                      _sh(["pdfinfo", str(p_)]))
        if m:
            real[f] = int(m.group(1))
    alias = {"paperY_main.pdf": ("主文", "paperY_main.pdf", "paperY_main",
                                 "main text", "manuscript"),
             "paperY_ESM.pdf": ("ESM", "paperY_ESM.pdf", "paperY_ESM",
                                "Supplementary")}
    # The claim must be adjacent: "<name> ... N pp". A wider window pairs a
    # number with whichever name happens to be nearby and reports nonsense
    # (it did, on this round's own README: "主文 26 页 + ESM 22 页").
    # A document that declares itself HISTORICAL on its first lines is
    # describing a past build; its page count is a record, not a claim about
    # this package. The same declaration the builder reads (P5R-N SS5.4) is
    # read here, so the two cannot disagree, and the skips are printed.
    HIST = ("历史记录", "historical record", "describes the state at",
            "HISTORICAL / SUPERSEDED")
    historical = [d.name for d in sorted(STAGING.glob("*.md"))
                  if any(h in d.read_text(errors="replace")[:600] for h in HIST)]
    wrong = []
    for doc in sorted(STAGING.glob("*.md")):
        if doc.name in historical:
            continue
        text = " ".join(doc.read_text().split())
        for f, names in alias.items():
            if f not in real:
                continue
            for a in names:
                for m in re.finditer(
                        re.escape(a) + r"[^0-9\n]{0,20}?(\d{1,3})\s*"
                        r"(?:pp\.?|pages|页)", text):
                    n = int(m.group(1))
                    if "第" in m.group(0):
                        continue          # "ESM 第 11 页" is a page reference
                    if n != real[f]:
                        wrong.append(f"{doc.name}: \"{m.group(0)[:40]}\" "
                                     f"but {f} has {real[f]} pages")
    # An MD5 a package document states for a package PDF must be that
    # PDF's. The inspection log names the build it accepted; without this
    # the log could describe a file the package does not contain.
    import hashlib
    md5 = {}
    for f in real:
        p_ = _pf(f)
        md5[f] = hashlib.md5(p_.read_bytes()).hexdigest()
    checked = 0
    for doc in sorted(STAGING.glob("*.md")):
        if doc.name in historical:
            continue
        text = " ".join(doc.read_text().split())
        for f, digest in md5.items():
            for m in re.finditer(re.escape(f) + r"[^\n]{0,60}?`([0-9a-f]{32})`",
                                 text):
                checked += 1
                if m.group(1) != digest:
                    wrong.append(f"{doc.name}: states MD5 {m.group(1)[:12]}.. "
                                 f"for {f}, actual {digest[:12]}..")
    ok = bool(real) and not wrong
    return {"gate": "9 stated page counts and file identity", "pass": ok,
            "detail": f"actual={real}; MD5 claims checked={checked}; "
                      f"historical documents skipped={historical or 'none'}; "
                      f"mismatches={wrong or 'none'}"}


def gate_10_self_dates(package_date: str = None) -> dict:
    """No file in the package carries a self-declared date after it.

    The 8-29 package carried an addendum stamped 2026-08-31 (fourth
    internal review, automation section). A self-declared date later than
    the package's own date is a provenance error whatever caused it.
    """
    # Eighth review P0-1: in package mode the date is READ FROM THE PACKAGE.
    # It used to come from the running machine's clock, which made all_green a
    # function of the reader's timezone: five files stamped 2026-08-31 read as
    # future-dated to anyone west of JST, and the same delivered ZIP was green
    # here and RED there. A verdict about a file must not depend on where the
    # file is opened.
    pkg = package_date
    if pkg is None and MODE == "package":
        f = STAGING / "BUILD.json"
        pkg = json.loads(f.read_text()).get("package_date") if f.exists() else None
        if not pkg:
            return {"gate": "10 self-dates within the package", "pass": False,
                    "detail": "package mode, but BUILD.json carries no "
                              "package_date; the gate refuses to fall back to "
                              "the system clock (eighth review P0-1)"}
    if pkg is None:
        pkg = _sh(["date", "+%Y-%m-%d"]).strip()
    ex_path = ROOT / "internal_review" / "package_date_exemptions.json"
    ex, self_files = [], []
    if ex_path.exists():
        _e = json.loads(ex_path.read_text())
        ex = _e["exemptions"]
        self_files = _e.get("self_declaring_files", {}).get("files", [])
    absorbed, bad = {}, []
    for f in sorted(STAGING.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".md", ".txt", ".json"):
            continue
        rel = str(f.relative_to(STAGING))
        if rel in self_files:
            absorbed[rel] = "the declaration file itself"
            continue
        for d in sorted(set(re.findall(r"20\d\d-\d\d-\d\d", f.read_text(
                errors="replace")))):
            if d <= pkg:
                continue
            hit = [e for e in ex if e["date"] == d and rel in e["files"]]
            if hit:
                absorbed[f"{rel}:{d}"] = hit[0]["reason"][:60]
            else:
                bad.append(f"{rel}: {d}")
    src = ("BUILD.json (stamped at build time)" if MODE == "package"
           else "the system clock (live mode)")
    return {"gate": "10 self-dates within the package", "pass": not bad,
            "detail": f"package date {pkg} from {src}; future-dated without a declared "
                      f"exemption: {bad or 'none'}; declared exemptions "
                      f"absorbed: {len(absorbed)} {sorted(absorbed)}"}


def gate_11_declared_deletions() -> dict:
    """Everything the response letter declares deleted is gone from the render.

    The IR4 response letter said "orders of magnitude" had been removed
    while the manuscript still carried it (fourth internal review, §2.3).
    The claims are listed in internal_review/declared_deletions.json and
    each is checked against BOTH rendered PDFs, not against the source.
    """
    decl = ROOT / "internal_review" / "declared_deletions.json"
    if not decl.exists():
        return {"gate": "11 declared deletions absent from the render",
                "pass": False,
                "detail": f"missing declaration file: {decl}"}
    items = json.loads(decl.read_text())["deleted"]
    text = ""
    for f in ("paperY_main.pdf", "paperY_ESM.pdf"):
        p_ = _pf(f)
        if p_.exists():
            text += " " + " ".join(_sh(["pdftotext", str(p_), "-"]).split())
    survivors = [it["string"] for it in items
                 if re.search(it.get("pattern") or re.escape(it["string"]),
                              text, re.I)]
    return {"gate": "11 declared deletions absent from the render",
            "pass": not survivors,
            "detail": f"claims checked: {len(items)}; still rendered: "
                      f"{survivors or 'none'}"}


def gate_12_figure_freshness() -> dict:
    """No stale figure copy anywhere, and every figure is newer than t_final.

    The fourth internal review warned that an older Fig. 3 carrying the
    withdrawn wording must not flow back in. This gate looks for any copy
    of a figure outside reporting/out and checks the live ones' mtimes.
    """
    #: P5R-O SS4, executed. A figure whose truth source is a delivered ASSET
    #: is not checked for freshness -- no generator writes it, so "newer than
    #: the evidence it reads" asks a question with no referent. It is checked
    #: for IDENTITY instead: the file in use must be the registered one, byte
    #: for byte, and its declared home is not a stray copy.
    import hashlib as _hl, re as _re
    reg, assets = CODE_ROOT / "docs" / "figure_assets.json", {}
    if reg.exists():
        for a in json.loads(reg.read_text()).get("assets", []):
            assets[a["figure"]] = a
    asset_notes, asset_bad = [], []
    for name, a in sorted(assets.items()):
        cands = [ROOT / a["path"], STAGING / Path(a["path"]).name,
                 _pf(Path(a["path"]).name)]
        live = next((c for c in cands if c.exists()), None)
        if live is None:
            asset_bad.append(f"{name}: registered asset {a['path']} not found")
            continue
        got = _hl.sha256(live.read_bytes()).hexdigest()
        if got != a["pdf_sha256"]:
            asset_bad.append(f"{name}: in use is sha256 {got[:12]}.., "
                             f"registered {a['pdf_sha256'][:12]}..")
        else:
            asset_notes.append(
                f"{name} = registered asset {a['pdf_sha256'][:12]}.. "
                f"({Path(a['superseded_generator']).name} SUPERSEDED-BY-ASSET)")
        #: An asset does not follow t_final, so whatever numbers it prints are
        #: registered here -- and an EMPTY registration is re-derived from the
        #: delivered PDF rather than believed.
        if not a.get("in_figure_numbers"):
            txt = _sh(["pdftotext", str(live), "-"], ROOT)
            found = sorted(set(_re.findall(r"\d+\.\d+|\d+%|\d+/\d+", txt)))
            if found:
                asset_bad.append(f"{name}: registered as printing no data "
                                 f"number, but prints {found[:6]}")
    names = tuple(n for n in ("Fig_vera", "Fig_leakage", "Fig_scoreboard")
                  if n not in assets)
    strays = []
    # In package mode the generated figures live inside the package by
    # design (gate_inputs/code_SNI/reporting/out); scanning the package for
    # "stray copies" of them would report the package's own contents.
    bases = ([PAPER] if MODE == "package"
             else [ROOT / "paper_R1", STAGING, ROOT / "VERA_GitHub"])
    #: P7-A SS2. paper_R1/main/Fig now holds a copy of each generated figure,
    #: because a directory that compiles on its own has to contain what the
    #: document includes. That is a DECLARED view, not a stray -- but only
    #: while it is identical to the file the generator wrote. The rule this
    #: gate enforces was never "no second copy"; it was "no STALE copy", and
    #: a copy proved byte-identical at gate time is not one. Anything under
    #: the declared view that has drifted is still reported, and now with the
    #: sharper message: it is a fork, not merely a duplicate.
    declared_views = [ROOT / "paper_R1" / "main" / "Fig"]
    for base in bases:
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not (f.is_file() and any(f.name.startswith(n) for n in names)):
                continue
            if any(f.parent == v for v in declared_views):
                canon = OUTDIR / f.name
                if canon.exists() and canon.read_bytes() == f.read_bytes():
                    continue
                strays.append(f"{f.relative_to(ROOT)} (declared view, but it "
                              f"DIFFERS from {canon.relative_to(ROOT)})")
                continue
            strays.append(str(f.relative_to(ROOT)))
    tf = CODE_ROOT / "results" / "T5_stats" / "t_final.json"
    stale = []
    # Unzipping does not restore mtimes: inside a package every file's mtime
    # is the moment it was extracted, so comparing them answers at random.
    # The builder recorded the real ones in BUILD.json; use those there.
    src_mt = {}
    if MODE == "package":
        f = STAGING / "BUILD.json"      # same file gate 1 reads for the head
        if f.exists():
            src_mt = json.loads(f.read_text()).get("source_mtimes", {})
        if not src_mt:
            stale.append("BUILD.json carries no source_mtimes: this package "
                         "cannot answer the freshness question at all")

    def _mt(rel, live):
        return src_mt.get(rel) if src_mt else (live.stat().st_mtime
                                               if live.exists() else None)

    t_mt = _mt("results/T5_stats/t_final.json", tf)
    if t_mt is not None and not (MODE == "package" and not src_mt):
        for n in names:
            p_ = OUTDIR / f"{n}.pdf"
            f_mt = _mt(f"reporting/out/{n}.pdf", p_)
            if not p_.exists():
                stale.append(f"{n}.pdf missing")
            elif f_mt is None:
                stale.append(f"{n}.pdf has no recorded build time")
            elif f_mt < t_mt:
                stale.append(f"{n}.pdf predates t_final.json")
    withdrawn = []
    for n in names:
        p_ = OUTDIR / f"{n}.pdf"
        if p_.exists():
            txt = _sh(["pdftotext", str(p_), "-"]).lower()
            withdrawn += [w for w in ("mandatory floor", "one ruler",
                                      "mechanical verdict") if w in txt]
    ok = not strays and not stale and not withdrawn
    return {"gate": "12 figure freshness / asset identity",
            "pass": ok and not asset_bad,
            "detail": f"stray copies={strays or 'none'}; "
                      f"stale={stale or 'none'}; "
                      f"withdrawn wording={withdrawn or 'none'}; "
                      f"asset-backed: {asset_notes or 'none'}"
                      + (f"; ASSET FAILURES: {asset_bad}" if asset_bad else "")}


def gate_13_bibliography() -> dict:
    """Field policy, authoritative verification, and per-citation-point support.

    A real, well-formatted, resolvable reference attached to a claim it never
    made is the defect no format check sees; this gate requires a support
    record per citation POINT, because one work cited in four places is asked
    to carry four different claims (P5R-M SS4).
    """
    sys.path.insert(0, str(_REAL_CODE_ROOT))
    from reporting import bib_gate
    strict = Path(bib_gate.SUPPORT).exists()
    r = bib_gate.check(strict_support=strict)
    detail = (f"cited {r['n_cited']}, citation points {r['n_points']}, "
              f"support requirement enforced: {strict}")
    if r["failures"]:
        detail += "; " + "; ".join(f"{k}={len(v)}" for k, v in
                                   r["failures"].items())
    return {"gate": "13 bibliography (fields, verification, support)",
            "pass": r["pass"] and strict, "detail": detail}


def gate_14_closed_states_resolve() -> dict:
    """Every path a Closed row cites exists in what was delivered.

    Sixth review SS5/SS6: the status table declared rows Closed and pointed at
    evidence the package did not contain, so "Closed" could not be checked by
    the person it was written for. A status is a claim about the package, so it
    is checked against the package.
    """
    decl = CODE_ROOT / "docs" / "ir_status_declarations.json"
    if not decl.exists():
        return {"gate": "14 Closed states resolve inside the package",
                "pass": False, "detail": f"declarations absent at {decl}"}
    items = json.loads(decl.read_text())
    items = items["items"] if isinstance(items, dict) else items
    # .sh added: the freshness row cites regenerate_all.sh, and the regex
    # that did not know about shell scripts is why that citation was never
    # checked (ninth review P0-3 B).
    pat = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*"
                     r"\.(?:json|md|tex|pdf|csv|py|txt|bib|bbl|sh)")
    #: where a cited path may live inside the package
    def _resolves(rel: str) -> bool:
        """The cited path, resolved as written -- no basename fallback.

        Ninth review P0-3: the fallback let any citation succeed as long as
        SOME file somewhere in the package shared its basename, so a document
        could name `reporting/change_summary.py`, the package could contain
        no such file at any path, and the gate stayed green because it had
        stopped checking the part of the claim that was wrong. A path in a
        document is an instruction to a reader; it either resolves as typed
        or it does not.
        """
        rel = rel.lstrip("./")
        cands = [base / rel for base in (STAGING, STAGING / "evidence",
                                         STAGING / "source", ROOT, CODE_ROOT,
                                         PAPER, OUTDIR)]
        if rel.startswith("paper_R1/"):
            if MODE == "package":
                cands.append(STAGING / "source" / rel[len("paper_R1/"):])
            else:
                #: paper_R1 is main/ + esm/ since P7-A SS2, so the path a
                #: declaration writes -- paper_R1/paperY_main.tex -- is not a
                #: path that exists. It used to resolve anyway, through a
                #: seven-day-stale copy under internal_review/ir_staging/source
                #: that no build reads and nothing refreshes. Removing that
                #: copy is what exposed this: the mapping was always missing,
                #: and a stale directory was standing in for it.
                cands.append(_pf(rel[len("paper_R1/"):]))
        if MODE == "package" and rel.startswith("internal_review/ir_staging/"):
            # The staging directory IS the package root: a declaration citing
            # internal_review/ir_staging/GATES.txt means the GATES.txt a
            # reader opens at the top of the ZIP.
            cands.append(STAGING / rel[len("internal_review/ir_staging/"):])
        if MODE == "package" and rel.startswith("code_SNI/"):
            # The mirror image of the mapping below. Declarations are written
            # in REPOSITORY terms (code_SNI/reporting/status_table.py); the
            # package carries that same file under code/. One line, both
            # directions, and the rest of the path still has to match exactly.
            cands.append(STAGING / "code" / rel[len("code_SNI/"):])
        if MODE != "package" and rel.startswith("code/"):
            # Documents cite the path a READER follows, which is the package
            # path. In a working tree that directory does not exist yet;
            # code/<x> there is code_SNI/<x>. The mapping is one line and is
            # NOT a basename fallback -- the rest of the path still has to
            # match exactly.
            cands.append(CODE_ROOT / rel[len("code/"):])
        return any(c.exists() for c in cands)

    def _check_paths(chk) -> list:
        """Every path a CHECK names, including nested all_of branches.

        Twelfth review P0-5. This gate read the `evidence` prose and nothing
        else, so a row could declare an `attested` check against
        reports/P5R_H_receipt.md, that file could be absent from the package,
        and the audit still recorded `passed = true, attested in
        reports/P5R_H_receipt.md`. The machine audit claimed to have read a
        file the delivered package does not contain, and nineteen gates went
        green over it. The prose is what a reader follows; the check is what
        the MACHINE followed, and it is the one that was never verified.
        """
        found = []
        if isinstance(chk, dict):
            for key in ("file", "artifact"):
                if isinstance(chk.get(key), str):
                    found.append(chk[key])
            for key in ("files", "sources"):
                v = chk.get(key)
                if isinstance(v, list):
                    found += [x for x in v if isinstance(x, str)]
            for sub in chk.get("checks", []) or []:
                found += _check_paths(sub)
        return found

    closed, missing = 0, []
    for it in items:
        st = str(it.get("status") or it.get("declared") or "")
        if not st.lower().startswith(("closed", "verified")):
            continue
        closed += 1
        for rel in dict.fromkeys(pat.findall(str(it.get("evidence", "")))):
            if not _resolves(rel):
                missing.append(f"{it.get('id')}: {rel}")
        for rel in dict.fromkeys(_check_paths(it.get("check", {}))):
            if not _resolves(rel):
                missing.append(f"{it.get('id')}: check path {rel} "
                               f"(the machine audit reads this; it is not in "
                               f"the package)")

    # Eighth review P1-4: five current documents told the reader to run
    # `code/recompute_fair_pair.py` and `code/package_gates.py`, and neither
    # path exists -- the scripts moved into code/experiments/ and
    # code/reporting/ when the package started keeping the repository's own
    # directory shape. A command a reviewer cannot run is worse than no
    # command: it reads as "we did not try this ourselves". Every code/ path a
    # current package document names must therefore resolve inside the package.
    HIST = ("历史记录", "historical record", "describes the state at",
            "HISTORICAL / SUPERSEDED")
    # Not only code/-prefixed paths: the documents cite
    # `reporting/change_summary.py` and `experiments/...` too, and the
    # code/-only pattern walked straight past them (ninth review P0-3).
    code_pat = re.compile(
        r"(?<![\w/])((?:code|reporting|experiments|evidence|source|"
        r"gate_inputs|docs)/[A-Za-z0-9_./-]+\.(?:py|sh|md|txt|json|csv))")
    SCOPE_OUT = {"evidence": "frozen artifacts, including committed rule "
                              "documents whose repository paths are what they "
                              "recorded at commit time",
                 "gate_inputs": "the gates' own mirror of the repository, "
                                "where a repository path is the right path"}
    #: The documents that speak to the reader about THIS package. They live at
    #: the package root; an excluded tree must not hold one, or the exclusion
    #: would become a place to put a claim the gate cannot see.
    CURRENT_FACING = {"README.md", "CHANGE_SUMMARY.md", "STATUS_TABLE.md",
                      "INSPECTION_LOG.md", "INSPECTION_DEFERRED.md",
                      "CROSSCHECK_v4.md", "PACKAGE_GATES_HOWTO.md"}
    misplaced = [str(f.relative_to(STAGING))
                 for tree in SCOPE_OUT
                 for f in (STAGING / tree).rglob("*.md")
                 if (STAGING / tree).is_dir()
                 and ((f.name in CURRENT_FACING
                       and f.parent == STAGING / tree)
                      or f.name.startswith("RESPONSE_TO_IR"))]
    missing[:0] = [f"current-facing document inside an excluded tree: {m}"
                   for m in misplaced]
    bad_cmd, docs_scanned, skipped_trees = [], 0, []
    for doc in (sorted(STAGING.rglob("*.md")) if MODE == "package" else []):
        if "_paper" in doc.parts:
            continue
        top = doc.relative_to(STAGING).parts[0]
        if top in SCOPE_OUT:
            if top not in skipped_trees:
                skipped_trees.append(top)
            continue
        raw = doc.read_text(errors="replace")
        if any(h in raw[:600] for h in HIST):
            continue
        docs_scanned += 1
        for rel in dict.fromkeys(code_pat.findall(raw)):
            if not (STAGING / rel).exists():
                bad_cmd.append(f"{doc.relative_to(STAGING)}: {rel}")
    missing += bad_cmd

    return {"gate": "14 Closed states resolve inside the package",
            "pass": not missing,
            "detail": f"rows checked: {closed}; current documents scanned for "
                      f"code paths: {docs_scanned}"
                      + (f" (out of scope, by declaration: "
                         f"{'/, '.join(skipped_trees)}/ -- "
                         + '; '.join(f'{k}: {v}' for k, v in SCOPE_OUT.items()
                                     if k in skipped_trees) + ")"
                         if skipped_trees else "")
                      + ("" if MODE == "package" else
                         " (package mode only: code/ exists in the package, "
                         "not in the working tree)")
                      + f"; unresolved evidence paths: "
                      f"{len(missing)}"
                      + ("; " + "; ".join(missing[:8]) if missing else "")}


def gate_15_inventory_reproduces() -> dict:
    """Re-run the information-symmetry inventory ON THE PACKAGE.

    The package carries a JSON saying how many comparison sites there are and
    how many are undisclosed, and release condition (1) is stated from it. But
    nothing re-derived it from the package's own text -- so the number could
    be, and once was, computed on an earlier state of the manuscript and the
    generated fragments. Re-running the enumeration here is the only way the
    stated count is a claim about what was delivered.

    An undeclared site counts as a failure, not as a pass: the tool's rule is
    that a mention which nobody has classified is exactly the thing an
    inventory exists to stop.
    """
    sys.path.insert(0, str(_REAL_CODE_ROOT / "experiments"))
    import importlib
    import perm_sni_inventory as inv
    importlib.reload(inv)
    inv.PAPER = PAPER
    inv.GEN = OUTDIR
    inv.DECL = CODE_ROOT / "docs" / "perm_sni_inventory_declarations.json"
    # a RECOMPUTED row's artifact must be in THIS tree, not in somebody's
    # checkout -- the seventh review's SS4.1 in one line
    inv.ARTIFACT_ROOT = CODE_ROOT
    stated = None
    f = CODE_ROOT / "results" / "T6_symmetry" / "perm_sni_comparison_inventory.json"
    if f.exists():
        j = json.loads(f.read_text())
        stated = {"n_sites_found": j.get("n_sites_found"),
                  "n_blockers": j.get("n_blockers")}
    try:
        r = inv.build(strict=False, write=False)
    except Exception as e:                       # noqa: BLE001
        return {"gate": "15 inventory reproduces on the package", "pass": False,
                "detail": f"the inventory could not be re-run here: {e}"}
    undeclared = r["n_sites_found"] - r["n_declared"]
    drift = (stated is not None
             and (stated["n_sites_found"] != r["n_sites_found"]
                  or stated["n_blockers"] != r["n_blockers"]))
    ok = undeclared == 0 and r["n_blockers"] == 0 and not drift
    return {"gate": "15 inventory reproduces on the package", "pass": ok,
            "detail": f"re-derived here: {r['n_sites_found']} sites, "
                      f"{r['n_declared']} declared, {undeclared} undeclared, "
                      f"{r['n_blockers']} blockers; the packaged artifact "
                      f"states {stated}; drift={drift}"}


def gate_16_duplicate_digests() -> dict:
    """Any two files in the package with the same name must be the same file.

    The package carried two `t_final.json`s that were a round apart: the
    manuscript read one, the status table pointed a reader at the other, and
    the older one still said the recovery effect was +0.154. "Single source of
    truth" is not a claim you can make about a tree nobody has checked for
    forks, so this checks it.
    """
    import hashlib
    if MODE != "package":
        # The question is about ONE tree that ships as a unit. A repository is
        # not that tree -- it legitimately holds the same mask file under three
        # mask families -- so this gate answers only where the answer means
        # something, and says so rather than passing quietly.
        return {"gate": "16 duplicate-basename digest census", "pass": True,
                "detail": "package mode only: a working tree is not a "
                          "delivered unit, so a duplicate name there is not a "
                          "fork. Run --from-package for the real answer."}
    root = STAGING                       # the package root, not gate_inputs
    by_name = {}
    for f in sorted(root.rglob("*")):
        # No carve-out for the gates' own scratch any more: the working view
        # is assembled outside the package, so everything under this root is
        # package content by construction.
        if not f.is_file():
            continue
        by_name.setdefault(f.name, []).append(f)
    #: Names that legitimately belong to different documents. Declared here
    #: with the reason and printed with a count, so an exemption cannot grow
    #: quietly into a loophole -- the same rule gates 10 and 11 follow.
    NOT_ONE_ARTIFACT = {
        "Fig_vera.pdf": "two files that are SUPPOSED to differ: the delivered "
                        "design asset under paper_R1/main/Fig, and the "
                        "superseded generator output under reporting/out kept "
                        "as the historical content truth the asset was "
                        "accepted against (SUPERSEDED-BY-ASSET). Gate 12 "
                        "checks the delivered one against its registered "
                        "sha256; treating the pair as a fork would demand they "
                        "be equal, which would defeat the point of replacing "
                        "one with the other",
        "README.md": "the package's front page and the prereg archive's own "
                     "index are different documents that share a filename",
        # a package initializer belongs to its package; two of them sharing a
        # filename is what Python requires, not a forked artifact
        "__init__.py": "each Python package needs its own initializer; "
                        "code/reporting/ and code/experiments/ are different "
                        "packages, so these are different files by design",
    }
    forks, absorbed = [], []
    for name, fs in sorted(by_name.items()):
        if len(fs) < 2:
            continue
        digests = {hashlib.sha256(x.read_bytes()).hexdigest() for x in fs}
        if len(digests) == 1:
            continue
        if name in NOT_ONE_ARTIFACT:
            absorbed.append(f"{name} ({NOT_ONE_ARTIFACT[name]})")
            continue
        forks.append({"name": name,
                      "copies": [str(x.relative_to(root)) for x in fs],
                      "n_digests": len(digests)})
    dups = sum(1 for fs in by_name.values() if len(fs) > 1)
    return {"gate": "16 duplicate-basename digest census", "pass": not forks,
            "detail": f"names appearing more than once: {dups}; "
                      f"forked (same name, different content): "
                      f"{forks if forks else 'none'}; "
                      f"declared exemptions absorbed: {absorbed or 'none'}"}


def gate_17_cited_scripts_run() -> dict:
    """Every script a current document offers as evidence actually starts.

    Ninth review P0-3 C. `fair_same_host_recovery.py --selftest` was cited in
    the response letter as in-package verification and died on
    `ModuleNotFoundError: pilot_r21` the moment a reviewer ran it from the
    unpacked directory. Gate 14 checks that a cited path EXISTS; existing and
    running are different claims, and the letter made the second one.

    So: each shipped script named by a current document is executed here, in
    the package, with the cheapest invocation that proves the interpreter can
    load it and its imports resolve -- its own selftest where it declares one,
    `--help` otherwise. A non-zero exit is RED. Nothing is trained and nothing
    is written; the point is only that the instruction in the document is one
    a reader can follow.
    """
    if MODE != "package":
        return {"gate": "17 cited scripts run inside the package", "pass": True,
                "detail": "package mode only: there is no code/ in a working "
                          "tree, and running the repository's own scripts "
                          "would prove nothing about the delivered file"}
    code = STAGING / "code"
    if not code.is_dir():
        return {"gate": "17 cited scripts run inside the package",
                "pass": False, "detail": "the package carries no code/"}
    #: Scripts the package presents as RUNNABLE: the gate machinery, the
    #: recomputation a reviewer is invited to perform, and the generators
    #: whose selftests are cited as evidence. These are executed.
    RUNNABLE = {"fair_same_host_recovery.py": "--selftest",
                "change_summary.py": "--check=CHANGE_SUMMARY.md",
                "t_final.py": "--selftest",
                "perm_sni_inventory.py": "--selftest",
                "recompute_fair_pair.py": "--help",
                "package_layout.py": "--help",
                "render_checks.py": "--help",
                "facts_gate.py": "--help",
                "status_table.py": "--check=STATUS_TABLE.md",
                "bib_inventory.py": "--help"}
    #: The rest ship so a reader can READ them -- a Closed row cites the
    #: generator that produced a table, not a command to run. Asking them for
    #: --help would demand the whole training environment (torch, the config
    #: tree, the results tree), which a review package deliberately does not
    #: carry. For those the checkable claim is that the file is intact and
    #: parses, and that is what is checked and what is reported. Two classes,
    #: two claims, both stated -- rather than one claim covering both and
    #: being false of half.
    # Tenth review P0-3. "actually starts" was one word covering four
    # different strengths of claim, and the weakest of them -- `bash -n`,
    # which parses shell and executes nothing -- was being counted among the
    # executions. Four tiers, four counts, never merged:
    #
    #   selftest       the script ran its own assertions and they passed
    #   executed       the script ran and exited zero
    #   syntax-checked shell parsed, nothing run
    #   parse-checked  Python compiled, nothing run
    #
    # A script shipped for INSPECTION is labeled as such and is not counted
    # as anything stronger.
    ran, selftested, read_only, bad = [], [], [], []
    for f in sorted(code.rglob("*.py")):
        if f.name == "__init__.py":
            continue
        rel = str(f.relative_to(STAGING))
        arg = RUNNABLE.get(f.name)
        if arg is None:
            try:
                compile(f.read_text(), str(f), "exec")
                read_only.append(f.name)
            except SyntaxError as e:
                bad.append(f"{rel} does not parse: {str(e)[:110]}")
            continue
        argv = arg.split("=", 1) if arg.startswith("--check=") else [arg]
        r = subprocess.run([sys.executable, str(f.relative_to(code))] + argv,
                           cwd=code, capture_output=True, text=True,
                           timeout=900,
                           env={**os.environ, "PYTHONHASHSEED": "2025",
                                "PYTHONPATH": str(code)})
        (selftested if arg.startswith("--selftest") or arg.startswith("--check")
         else ran).append(f"{f.name} {arg}")
        if r.returncode != 0:
            tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            bad.append(f"{rel} {arg} -> exit {r.returncode}: {tail[:110]}")
    # A shell script is parsed, not run. regenerate_all.sh is a record of the
    # repository's regeneration procedure: it calls generators this package
    # deliberately does not carry and needs the training environment. It ships
    # so a reader can see the order things are built in -- and `bash -n` says
    # only that it is syntactically a shell script.
    syntax = []
    for f in sorted(code.rglob("*.sh")):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True,
                           text=True, timeout=60)
        syntax.append(f"{f.name} (bash -n; repository-environment procedure, "
                      f"included for inspection, not executable in this "
                      f"review package)")
        if r.returncode != 0:
            bad.append(f"{f.relative_to(STAGING)} bash -n -> "
                       f"{r.stderr.strip()[:110]}")
    return {"gate": "17 cited scripts, by strength of claim", "pass": not bad,
            "detail": f"selftest passed: {len(selftested)} "
                      f"({', '.join(sorted(selftested))}); "
                      f"executed (exit 0): {len(ran)} "
                      f"({', '.join(sorted(ran))}); "
                      f"syntax-checked only: {len(syntax)} "
                      f"({'; '.join(sorted(syntax))}); "
                      f"parse-checked only: {len(read_only)} "
                      f"({', '.join(sorted(read_only))}); "
                      f"-- selftest, execute, syntax-check and parse-check "
                      f"are four strengths of claim and are never merged "
                      f"(tenth review P0-3); "
                      f"failures: {bad if bad else 'none'}"}


def gate_18_retired_label_quarantined() -> dict:
    """The retired rule's label appears ONLY in files that declare themselves
    historical -- searched over the WHOLE package, no namespace exempt.

    Tenth review P0-1 item 5, and the shape of the failure that made it
    necessary. The ninth round asserted "the retired label appears NOWHERE --
    no namespace exempted" while searching exactly one JSON file; two copies
    of the canonical fact store shipped with the retired label inside
    them, and the assertion passed because it never looked. Scope blindness is
    not a weaker version of the check, it is the absence of the check.

    So the search is over every TEXT file in the package -- .pdf, .png and .zip
    hold no readable text and are skipped, which is stated in the output rather
    than left to be discovered. A hit is legal in exactly four places: a file
    carrying experiments/audit_history.MARKER near its top (it says on its
    first line that it is not current); audit_history.py itself, the one module
    that owns the literal; a named narrative response letter; and a .py whose
    every occurrence sits on a comment line. Compiled bytecode is never legal,
    whatever it appears to declare about itself.
    """
    sys.path.insert(0, str(_REAL_CODE_ROOT / "experiments"))
    try:
        from audit_history import RETIRED_LABEL as LABEL, MARKER
    except ImportError as exc:                      # pragma: no cover
        return {"gate": "18 the retired label is quarantined", "pass": False,
                "detail": f"cannot import the label to search for "
                          f"({exc}); a gate that cannot name what it looks "
                          f"for is not a check"}
    #: The response letters record what the label was and how it was retired;
    #: they are the project's own narrative of the correction, and a reader
    #: meets them as history. They are allowed to name it, and are listed here
    #: rather than pattern-matched, so the allowance stays visible.
    NARRATIVE = {"RESPONSE_TO_IR7.md", "RESPONSE_TO_IR8.md",
                 "RESPONSE_TO_IR9.md", "RESPONSE_TO_IR10.md",
                 "RESPONSE_TO_IR11.md", "RESPONSE_TO_IR13.md",
                 "CHANGELOG_ARCHIVE.md"}
    bad, allowed, scanned = [], [], 0
    #: Compiled bytecode is never a document and can never declare itself
    #: historical -- but it CONTAINS its module's string constants, so a
    #: .pyc of audit_history.py satisfies the marker test by accident. The
    #: first run of this gate was green partly on that coincidence. Byproducts
    #: are therefore reported separately and are always red: nothing the run
    #: itself creates gets to be evidence about what the run inspected.
    byproducts = [str(f.relative_to(STAGING))
                  for f in sorted(STAGING.rglob("*"))
                  if f.is_file() and (f.suffix == ".pyc"
                                      or "__pycache__" in f.parts)]
    for f in sorted(STAGING.rglob("*")):
        if not f.is_file() or f.suffix.lower() in (".pdf", ".zip", ".png"):
            continue
        if f.suffix == ".pyc" or "__pycache__" in f.parts:
            continue                      # counted as a byproduct, not a file
        try:
            raw = f.read_text(errors="replace")
        except Exception:
            continue
        scanned += 1
        if LABEL not in raw:
            continue
        rel = str(f.relative_to(STAGING))
        if MARKER in raw[:2000] or f.name in NARRATIVE:
            allowed.append(rel)
        elif f.name == "audit_history.py":
            allowed.append(f"{rel} (the one module that carries the literal; "
                           f"everything else imports it from here)")
        elif f.suffix == ".py" and all(
                ln.lstrip().startswith("#")
                for ln in raw.splitlines() if LABEL in ln):
            allowed.append(f"{rel} (comments only: explains the removal)")
        else:
            bad.append(rel)
    return {"gate": "18 the retired label is quarantined",
            "pass": not bad and not byproducts,
            "detail": f"files searched: {scanned} (whole package, no "
                      f"namespace exempt; PDF/PNG/ZIP hold no text and are "
                      f"listed in the package manifest); the label appears in "
                      f"{len(allowed)} declared-historical file(s) "
                      f"{sorted(allowed)}; illegal occurrences: "
                      f"{bad if bad else 'none'}; compiled byproducts in the "
                      f"tree (never legal; the general guarantee that nothing "
                      f"was written is the read-only invariant, not this "
                      f"type-shaped check): "
                      f"{byproducts if byproducts else 'none'}"}


def gate_19_transcript_is_of_this_build() -> dict:
    """GATES.txt is the transcript of THIS build, not of an earlier candidate.

    Eleventh round. The package ships its own gate transcript, and a reader is
    told it was produced by running the packaged script against this ZIP. It
    was not checked that this was true, and it stopped being true: four
    rebuilds in a row copied forward a GATES.txt written at an earlier commit,
    so the delivered package carried a transcript from a superseded script --
    one that still displayed the .pyc accident gate 18 was added to refuse,
    and that under the shipped code would have been RED rather than GREEN.

    The count audit could not see it, because it READS the gate count off that
    same transcript: a stale file agrees with itself.

    Two invariants, both cheap and both fatal to a stale transcript:
      * its HEAD line equals BUILD.json's head_commit (the candidate and the
        final build share a commit, so this holds for an honest transcript and
        fails for one carried over from another build), and
      * it carries a line for EVERY gate this module defines -- which catches
        a transcript made at the right commit by an older script.
    """
    #: A CANDIDATE build is the bootstrap: the transcript of a ZIP cannot
    #: exist before the ZIP does, so the candidate necessarily carries the
    #: previous one. It says so in its own BUILD.json and is never delivered;
    #: the DELIVERED package carries no such flag, so this gate binds there.
    b = STAGING / "BUILD.json"
    if b.exists():
        try:
            if json.loads(b.read_text()).get("candidate"):
                return {"gate": "19 the gate transcript is of this build",
                        "pass": True,
                        "detail": "candidate build: the transcript of this ZIP "
                                  "cannot exist yet, so the check binds on the "
                                  "delivered package, which carries no "
                                  "candidate flag"}
        except Exception:
            pass
    t = STAGING / "GATES.txt"
    if not t.exists():
        return {"gate": "19 the gate transcript is of this build", "pass": False,
                "detail": "GATES.txt is not in the package"}
    text = t.read_text()
    head = _recorded_head() if MODE == "package" else _sh(
        ["git", "rev-parse", "HEAD"], CODE_ROOT).strip()
    stamped = re.search(r"HEAD=([0-9a-f]{7,40})", text)
    got = stamped.group(1) if stamped else ""
    head_ok = bool(got) and head.startswith(got)
    defined = sorted(int(m) for m in re.findall(
        r"^def gate_(\d+)_", (_REAL_CODE_ROOT / "reporting" /
                              "package_gates.py").read_text(), re.M))
    present = sorted({int(m) for m in re.findall(
        r"^\[(?:GREEN|RED)\s*\]\s*(\d+) ", text, re.M)})
    missing = [g for g in defined if g not in present]
    ok = head_ok and not missing
    verdict = ("match" if head_ok else
               "MISMATCH -- the transcript is from another build")
    detail = (f"transcript HEAD={got or 'absent'} vs BUILD.json "
              f"{head[:8]}: {verdict}; gates defined by the shipped module: "
              f"{len(defined)}; gates reported in the transcript: "
              f"{len(present)}")
    if missing:
        detail += (f"; MISSING from the transcript: {missing} "
                   f"(it was made by an older script)")
    return {"gate": "19 the gate transcript is of this build", "pass": ok,
            "detail": detail}




def gate_20_submission_view_in_sync() -> dict:
    """VERA_paper_R1 is a faithful view of paper_R1, or it is a liability.

    P5R-X. The submission source set is flattened for Editorial Manager, which
    cannot process subfolders. Flattening means a SECOND copy of the
    manuscript exists, and a second copy is the oldest way for two versions of
    a paper to drift apart -- the one that was reviewed and the one that was
    submitted. The direction is one-way by rule (paper_R1 is the source), and
    this is the rule's enforcement: every file byte-identical to its source,
    nothing extra, nothing missing.

    It runs against the WORKING TREE only, and that is a RULING, not an
    accident of implementation: the submission view is delivered as its own
    archive beside the review package and never inside it (first author,
    2026-09-01). A package already carries the manuscript twice -- the sources
    under source/ and the rendered PDFs -- and a third copy would be a third
    thing that can drift from the other two. So in package mode this says so
    and passes, rather than inventing a verdict about a directory that is
    deliberately absent.
    """
    if MODE == "package":
        return {"gate": "20 the submission view is in sync", "pass": True,
                "detail": "not applicable in package mode. The submission "
                          "view is deliberately NOT shipped (first author, "
                          "2026-09-01): the package would then carry a third "
                          "copy of the manuscript beside source/ and the "
                          "rendered PDFs, and a third copy is a third thing "
                          "that can drift. It is delivered as its own archive "
                          "and checked in the working tree at build time"}
    sys.path.insert(0, str(_REAL_CODE_ROOT / "experiments"))
    import importlib
    import sync_submission_sources as sync
    importlib.reload(sync)
    r = sync.check()
    return {"gate": "20 the submission view is in sync", "pass": r["pass"],
            "detail": r["detail"]}


def tree_manifest(root: Path) -> dict:
    """relative path -> sha256, for every file under root."""
    import hashlib
    out = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            out[str(f.relative_to(root))] = hashlib.sha256(
                f.read_bytes()).hexdigest()
    return out


def _stat_manifest(root: Path) -> dict:
    """relative path -> (size, mtime_ns), for a tree too large to hash."""
    out = {}
    if not root.is_dir():
        return out
    for f in sorted(root.rglob("*")):
        if f.is_file():
            st = f.stat()
            out[str(f.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def read_only_invariant(before: dict, after: dict, label: str) -> dict:
    """Did the run change the tree it was auditing? Answered by comparison.

    Eleventh review, SS5. Every previous guarantee here was type-shaped: gate
    18 learned to refuse .pyc, so .pyc stopped appearing, and the two other
    ways the run wrote into its own subject went on unseen -- materialise_paper
    building pkg/_paper (twelve files, eight of them text, which is why gate 18
    reported 152 scannable files where the ZIP holds 144), and gate 15 writing
    its re-derived inventory into whichever checkout the script was driven
    from. Naming byproduct TYPES was the mistake. This names none: the tree
    goes in, the tree comes out, and every added, removed or altered path is
    listed.
    """
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after)
                     if before[k] != after[k])
    ok = not (added or removed or changed)
    detail = (f"{label}: {len(before)} files before, {len(after)} after; "
              f"added {len(added)}, removed {len(removed)}, "
              f"changed {len(changed)}")
    if not ok:
        detail += (f" -- ADDED {added[:12]}; REMOVED {removed[:12]}; "
                   f"CHANGED {changed[:12]}")
    return {"gate": f"READ-ONLY INVARIANT ({label})", "pass": ok,
            "detail": detail}


def run() -> dict:
    sys.path.insert(0, str(_REAL_CODE_ROOT))
    from reporting import status_table
    layers = status_table._layers()
    gates = [gate_1_p0a(), gate_2_independence(), gate_3_realpattern(layers),
             gate_4_references(), gate_5_status_table(), gate_6_pending(),
             gate_7_facts(), gate_8_rule_bundle(), gate_9_page_counts(),
             gate_10_self_dates(), gate_11_declared_deletions(),
             gate_12_figure_freshness(), gate_13_bibliography(),
             gate_14_closed_states_resolve(),
             gate_15_inventory_reproduces(),
             gate_16_duplicate_digests(),
             gate_17_cited_scripts_run(),
             gate_18_retired_label_quarantined(),
             gate_19_transcript_is_of_this_build(),
             gate_20_submission_view_in_sync()]
    ok = all(g["pass"] for g in gates)
    # SS5.2: a green line only counts when it was read off the package.
    return {"mode": MODE, "all_green": ok if MODE == "package" else False,
            "all_gates_pass": ok, "gates": gates}


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    # every gate function must be callable and return the contract shape
    g = gate_6_pending()
    check(set(g) == {"gate", "pass", "detail"}, "gate contract shape")
    check(isinstance(g["pass"], bool), "verdict is a boolean")
    g2 = gate_4_references()
    check("cited entries" in g2["detail"], "reference gate reports its count")

    # Gate 18's byproduct branch, on a fixture that reproduces the accident:
    # a .pyc holds its module's string constants, so it carries BOTH the
    # retired label and the HISTORICAL marker and passes the marker test by
    # coincidence. The first green run of this gate listed exactly such a file
    # among its "declared-historical" ones.
    import tempfile
    global STAGING
    keep = STAGING
    try:
        sys.path.insert(0, str(_REAL_CODE_ROOT / "experiments"))
        from audit_history import RETIRED_LABEL, MARKER
        with tempfile.TemporaryDirectory() as td:
            STAGING = Path(td)
            (STAGING / "__pycache__").mkdir()
            (STAGING / "__pycache__" / "audit_history.cpython-310.pyc"
             ).write_text(f"\x00\x00{MARKER}\x00{RETIRED_LABEL}\x00")
            g3 = gate_18_retired_label_quarantined()
            check(g3["pass"] is False,
                  "gate 18: a .pyc carrying the label is RED even though the "
                  "marker sits in its constant pool")
            check("audit_history.cpython-310.pyc" in g3["detail"],
                  "gate 18 names the byproduct it found")
            (STAGING / "__pycache__" / "audit_history.cpython-310.pyc").unlink()
            (STAGING / "__pycache__").rmdir()
            (STAGING / "AUDIT_HISTORY.md").write_text(
                f"# {MARKER}\n\nthe label was {RETIRED_LABEL}\n")
            g4 = gate_18_retired_label_quarantined()
            check(g4["pass"] is True,
                  "gate 18: a text file declaring itself historical may "
                  "carry the label")
            check("none" in g4["detail"].rsplit(":", 1)[-1],
                  "gate 18 prints 'none' when no byproduct is present")
    finally:
        STAGING = keep
    check(sys.dont_write_bytecode and
          os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
          "inspecting the package cannot write bytecode into it")

    # Gate 19 on the failure that created it: a transcript copied forward from
    # an earlier build. The delivered package carried one for four rebuilds.
    keep2, keep_mode = STAGING, MODE
    try:
        with tempfile.TemporaryDirectory() as td:
            STAGING = Path(td)
            globals()["MODE"] = "package"
            n = len(re.findall(r"^def gate_(\d+)_",
                               (_REAL_CODE_ROOT / "reporting" /
                                "package_gates.py").read_text(), re.M))
            (STAGING / "BUILD.json").write_text(json.dumps(
                {"head_commit": "abcdef1234567890" + "0" * 24}))
            lines = ["[mode] package: x.zip"] + [
                f"[GREEN] {i} gate {i}: fine" for i in range(1, n + 1)]
            (STAGING / "GATES.txt").write_text(
                "\n".join(["[GREEN] 1 P0-A: HEAD=abcdef12; ok"] + lines[1:]))
            check(gate_19_transcript_is_of_this_build()["pass"] is True,
                  "gate 19: a transcript at this commit with every gate passes")
            (STAGING / "GATES.txt").write_text(
                "\n".join(["[GREEN] 1 P0-A: HEAD=15d9c536; ok"] + lines[1:]))
            g5 = gate_19_transcript_is_of_this_build()
            check(g5["pass"] is False and "MISMATCH" in g5["detail"],
                  "gate 19: a transcript from another build is RED")
            (STAGING / "GATES.txt").write_text(
                "\n".join(["[GREEN] 1 P0-A: HEAD=abcdef12; ok"] + lines[1:-1]))
            g6 = gate_19_transcript_is_of_this_build()
            check(g6["pass"] is False and "older script" in g6["detail"],
                  "gate 19: a transcript missing the newest gate is RED")
    finally:
        STAGING, globals()["MODE"] = keep2, keep_mode

    # The read-only invariant, on the shapes that actually got past the
    # type-shaped checks: a plain .json and a plain .md written into the
    # audited root -- not a .pyc, which is the only kind anything previously
    # looked for. Eleventh review SS5.5.6 asks for exactly this fixture.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a.txt").write_text("one")
        (d / "sub").mkdir()
        (d / "sub" / "b.tex").write_text("two")
        base = tree_manifest(d)
        check(read_only_invariant(base, tree_manifest(d), "x")["pass"] is True,
              "invariant: an untouched tree passes")
        (d / "perm_sni_comparison_inventory.json").write_text("{}")
        g = read_only_invariant(base, tree_manifest(d), "x")
        check(g["pass"] is False and "perm_sni_comparison_inventory.json"
              in g["detail"],
              "invariant: a plain .json written into the tree is RED and named")
        (d / "perm_sni_comparison_inventory.json").unlink()
        (d / "_paper").mkdir()
        (d / "_paper" / "paperY_main.tex").write_text("x")
        check(read_only_invariant(base, tree_manifest(d), "x")["pass"] is False,
              "invariant: a working view built inside the tree is RED")
        (d / "_paper" / "paperY_main.tex").unlink()
        (d / "_paper").rmdir()
        (d / "a.txt").write_text("one modified")
        g2 = read_only_invariant(base, tree_manifest(d), "x")
        check(g2["pass"] is False and "CHANGED" in g2["detail"],
              "invariant: an edited file is RED, by digest not by mtime")
        (d / "a.txt").unlink()
        g3 = read_only_invariant(base, tree_manifest(d), "x")
        check(g3["pass"] is False and "REMOVED" in g3["detail"],
              "invariant: a deleted file is RED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--from-package", metavar="ZIP",
                    help="unpack this review package and run every gate "
                         "against it, reading nothing else. This is the only "
                         "mode in which all_green is reported.")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    tmp = None
    if a.from_package:
        import tempfile, zipfile
        tmp = tempfile.mkdtemp(prefix="pkg_gates_")
        with zipfile.ZipFile(a.from_package) as z:
            z.extractall(tmp)
        rebind_to_package(Path(tmp))
        print(f"[mode] package: {a.from_package}\n[unpacked] {tmp}")
    else:
        print("[mode] live working tree -- all_green is NOT reported in this "
              "mode; run --from-package on the delivered ZIP")
    # SS1.1: photograph the audited tree, run every gate, photograph it again.
    # The DRIVER tree is watched too, by size and mtime rather than by digest
    # -- it is a whole working repository -- because the write that got past
    # every type-shaped check landed there, not in the package.
    pre = tree_manifest(Path(tmp)) if tmp else {}
    drv_pre = _stat_manifest(_REAL_CODE_ROOT) if tmp else {}
    rep_pre = _stat_manifest(_REAL_CODE_ROOT.parent / "reports") if tmp else {}
    r = run()
    for g in r["gates"]:
        print(f"[{'GREEN' if g['pass'] else 'RED  '}] {g['gate']}: {g['detail']}")
    if tmp:
        checks = [read_only_invariant(pre, tree_manifest(Path(tmp)),
                                      "unpacked package, by SHA-256"),
                  read_only_invariant(drv_pre,
                                      _stat_manifest(_REAL_CODE_ROOT),
                                      "driver code tree, by size+mtime"),
                  read_only_invariant(rep_pre, _stat_manifest(
                      _REAL_CODE_ROOT.parent / "reports"),
                      "driver reports/, by size+mtime")]
        for c in checks:
            print(f"[{'GREEN' if c['pass'] else 'RED  '}] {c['gate']}: "
                  f"{c['detail']}")
        if not all(c["pass"] for c in checks):
            r["all_green"] = False
            r["all_gates_pass"] = False
        r["read_only"] = checks
    print(json.dumps({k: r[k] for k in ("mode", "all_green", "all_gates_pass")}))
    return 0 if r["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
