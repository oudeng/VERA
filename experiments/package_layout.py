"""What a review package contains, and where the gates find it.

P5R-N SS5.2/SS5.3, from the sixth review's finding that the package's own
documents claimed "thirteen gates all green" while the delivered ZIP did not
contain the evidence those gates and those documents point at.

One declaration, two consumers:

  * experiments/build_review_package.py  copies these files INTO the package
  * reporting/package_gates.py --from-package  reads them back OUT of it

So a file the gates need and the builder forgets cannot pass unnoticed: the
gate run against the package is the only run whose all_green counts, and it
reads only what the package holds.

The package is shaped for a READER -- documents and PDFs at the top, evidence
beside them -- and `--from-package` maps that shape onto the repository shape
the gates were written against.
"""
from __future__ import annotations

import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
PAPER = ROOT / "paper_R1"
STAGING = ROOT / "internal_review" / "ir_staging"

#: Where a manuscript file lives inside paper_R1, in search order.
#: P7-A SS2 split the flat paper_R1/ into main/ and esm/, one compilable
#: directory per document. Three layouts now exist and all three are legal:
#: the nested working tree, the flat view a package assembles, and the flat
#: view Editorial Manager is handed. Fifteen modules address manuscript files
#: as PAPER / "<basename>"; rather than teach each of them the new shape --
#: fifteen chances to teach one of them wrongly -- they ask here.
#: Flat FIRST, so materialise_paper()'s assembled directory keeps answering
#: for the package gates exactly as it did before.
PAPER_SUBDIRS = ("", "main", "esm", "main/Table", "esm/Table", "main/Fig")


def paper_file(name: str, paper: Path = None) -> Path:
    """The manuscript file `name`, wherever this layout keeps it.

    Returns the FLAT path when nothing matches, so a missing file reports the
    place a reader would look for it rather than the last place searched.
    """
    base = Path(paper) if paper is not None else PAPER
    for sub in PAPER_SUBDIRS:
        p = (base / sub / name) if sub else (base / name)
        if p.exists():
            return p
    return base / name

def cited_code_paths() -> list:
    """Every repository code path a status declaration cites."""
    import json
    import re
    f = CODE_ROOT / "docs" / "ir_status_declarations.json"
    if not f.exists():
        return []
    items = json.loads(f.read_text())
    items = items["items"] if isinstance(items, dict) else items
    pat = re.compile(r"(?:experiments|reporting|common|sni|stats|baselines)"
                     r"/[A-Za-z0-9_./-]+\.py")
    seen = []
    for it in items:
        for m in pat.findall(str(it.get("evidence", ""))):
            if m not in seen:
                seen.append(m)
    return seen


#: (source path, path inside the package). Directories are copied whole.
#: Where each evidence file's canonical copy lives in the repository.
#: `None` means the package is its only home (it is written for the reader and
#: has no working-tree original). Everything else is copied from the canonical
#: path at build time -- never from the staging folder, which is how four of
#: these went stale by up to a full round.
EVIDENCE_SOURCES = {
    "t_final.json": "results/T5_stats/t_final.json",
    "ir_status_declarations.json": "docs/ir_status_declarations.json",
    "terminology_registry.json": "docs/terminology_registry.json",
    "tapfam_summary.json": "results/T5_family/tapfam_summary.json",
    "declared_deletions.json": None,     # lives in internal_review/
    # NOT None: internal_review/package_date_exemptions.json is its home,
    # and the evidence/ copy drifted from it the moment one was edited
    # and the other was not (gate 16 caught the fork).
    "package_date_exemptions.json": "../internal_review/package_date_exemptions.json",
    "lambda_check.json": "results/T5_stats/lambda_check.json",
    "mask_bitcheck.json": "results/T5_stats/mask_bitcheck.json",
    "mask_digest_manifest.json": "results/T5_stats/mask_digest_manifest.json",
    "probe2_qualifiers.json": "results/T5_probe2/probe2_qualifiers.json",
    "probe2_summary.json": "results/T5_probe2/probe2_summary.json",
    "smoke_independence.json": "results/T4_downstream/smoke_independence.json",
    "t42_confirmatory.json": "results/T4_leakage/t42_confirmatory.json",
    "target_role_audit.json": "results/target_role_audit.json",
    "T52_probe_triangulation_rules.md": "docs/T52_probe_triangulation_rules.md",
    "T53_tap_family_rules.md": "docs/T53_tap_family_rules.md",
    # Eighth review P1-5: the rule above names the wrong input for two
    # variants. It now carries a banner saying so -- and the document
    # that corrects it travels beside it as a loose file, not only
    # inside the archive tarball, so a reader who opens one sees the
    # other.
    "T53_input_correction_rules.md": "docs/T53_input_correction_rules.md",
    # written into evidence/ by experiments/audit_history.py; the package
    # is their only home
    "AUDIT_HISTORY.json": None,
    "AUDIT_HISTORY.md": None,
    "CITATION_REGISTRATION.md": None,
    "prereg_manifest_20260829.json": None,
    # written by the builder into the package; no working-tree original
    "FAIR_PAIR_PROVENANCE.md": "__written_by_builder__",
    # the seventh review's P0-2: the fair same-host pair's own artifacts, so
    # the +0.159 in the manuscript can be recomputed from the package
    "fair_same_host_recovery.json": "results/T6_symmetry/fair_same_host_recovery.json",
    "fair_same_host_recovery_cells.csv": "results/T6_symmetry/fair_same_host_recovery_cells.csv",
}


#: The gate machinery, carried in the package so the reader can re-run it.
#: Listed rather than derived: these are the files `package_gates.py` imports
#: at module scope or inside a gate, and an import that silently fails would
#: turn a check into a skip.
GATE_CODE = [
    "reporting/__init__.py",
    "reporting/package_gates.py",
    # gate 13 imports these two, and a gate whose import fails inside the
    # package is not a gate the reader can run
    "reporting/bib_gate.py",
    "reporting/bib_inventory.py",
    "reporting/render_checks.py",
    "reporting/facts_gate.py",
    "reporting/status_table.py",
    # The freshness and figure rows of the status table name these two
    # generators as SOURCES, so a package-mode rebuild of the table has
    # to be able to find them (twelfth review P0-5, generalized).
    "reporting/fig_leakage.py",
    "reporting/fig_scoreboard.py",
    "reporting/compile_gate.py",
    "experiments/package_layout.py",
    # The builder itself. It holds the COUNT AUDIT -- the check that
    # every number a package document states equals the package's own
    # -- and RESPONSE_TO_IR11 cites it for two fixes to that audit, so
    # a reader who wants to check the claim needs the file. Gate 14
    # caught the dangling citation before this package shipped.
    "experiments/build_review_package.py",
    "experiments/perm_sni_inventory.py",
    "experiments/fair_same_host_recovery.py",
    "experiments/recompute_fair_pair.py",
    # gate 1 runs t_final's selftest as a SUBPROCESS. It used to run it from
    # the checkout even in package mode, so a package handed to a reviewer
    # with no repository could not produce the green line it claims; the
    # script and the one module its selftest imports now ship too.
    "experiments/t_final.py",
    "experiments/t51_cluster_stats.py",
    # Ninth review P0-3 A and B. CHANGE_SUMMARY says on its own first page
    # that change_summary.py generates it, and STATUS_TABLE's freshness row
    # cites regenerate_all.sh as the thing that scripts the build order.
    # Neither was in the package. A closure claim whose evidence is a script
    # the reader does not have is not a closure claim.
    "reporting/change_summary.py",
    # emits INSPECTION_LOG's per-page coverage table, so the same rule
    # applies: a document generated by a script the package does not carry is
    # a claim the reader cannot check (ninth-round adjudication condition)
    "reporting/inspection_coverage.py",
    # emits evidence/AUDIT_HISTORY.{json,md}, the one place the retired
    # label is allowed to live (tenth review P0-1)
    "experiments/audit_history.py",
    "reporting/regenerate_all.sh",
]


def lead_diff_name() -> str:
    """The lead reading copy diffs against what the LAST review saw.

    Hard-coding the baseline is how a package ships an IR4-era diff to a
    seventh reviewer: the file exists, the builder is happy, and the reader is
    handed a document that does not contain the round they are reviewing. So
    the name is read from the README, which is where it is stated to the
    reader, and the file must exist.
    """
    txt = (STAGING / "README.md").read_text()
    names = sorted(set(re.findall(r"paperY_main_diff_vs_(IR\d+)\.pdf", txt)))
    if len(names) != 1:
        raise ValueError(
            f"the README names {names or 'no'} IR-baseline diff; it must name "
            f"exactly one, because that is the package's lead reading copy")
    f = paper_file(f"paperY_main_diff_vs_{names[0]}.pdf")
    if not f.exists():
        raise FileNotFoundError(
            f"the README names {f.name} as the lead reading copy, but it does "
            f"not exist. Build it before packaging: a diff against the wrong "
            f"baseline is worse than none, because the reader trusts it.")
    return f.name


def entries() -> list:
    out = []

    # --- what a reader opens ------------------------------------------- #
    for name in ("README.md", "CHANGE_SUMMARY.md", "CROSSCHECK_v4.md",
                 "RESPONSE_TO_REVIEW_v4.md", "INSPECTION_LOG.md",
                 # what the inspection found and this round did NOT fix, with
                 # the reason for each: a reviewer rediscovering the same list
                 # costs both sides time we already spent
                 "INSPECTION_DEFERRED.md",
                 # the history CHANGE_SUMMARY no longer carries, labeled
                 # HISTORICAL / SUPERSEDED so no count in it reads as current
                 "CHANGELOG_ARCHIVE.md",
                 "STATUS_TABLE.md", "STATUS_TABLE.audit.json", "GATES.txt"):
        out.append((STAGING / name, name))
    for name in ("RESPONSE_TO_REVIEW_v5.md",
                 # the seventh review SS12.5: the six-item response the README
                 # pointed at did not exist; and this round's own
                 "RESPONSE_TO_IR6.md", "RESPONSE_TO_IR7.md",
                 "RESPONSE_TO_IR8.md", "RESPONSE_TO_IR9.md",
                 "RESPONSE_TO_IR10.md", "RESPONSE_TO_IR11.md",
                 "RESPONSE_TO_IR13.md"):
        p = ROOT / "internal_review" / name
        if p.exists():
            out.append((p, name))
    for name in ("paperY_main.pdf", "paperY_ESM.pdf",
                 lead_diff_name(), "paperY_main_diff_vs_R0.pdf"):
        out.append((paper_file(name), name))

    # --- the manuscript sources, as the README promises ----------------- #
    for name in ("paperY_main.tex", "paperY_ESM.tex", "references_Y.bib",
                 "esm_corrections.tex", "paperY_main.bbl",
                 "sn-vancouver-num.bst"):
        p = paper_file(name)
        if p.exists():
            out.append((p, f"source/{name}"))
    #: The package promises the reader the sources, materialise_paper()
    #: assembles the gates' working view out of them, and both fail QUIETLY if
    #: this loop finds nothing -- the build succeeds, the ZIP is smaller, and
    #: the first thing that notices is a gate crashing on a missing .tex. A
    #: P7-A edit to this very block silently matched nothing and shipped
    #: exactly that. Six files are promised; six must be found.
    n_src = sum(1 for _, rel in out if rel.startswith("source/"))
    if n_src != 6:
        raise FileNotFoundError(
            f"the package promises six manuscript sources under source/ and "
            f"this build found {n_src}. paper_R1 is main/ + esm/ since P7-A "
            f"SS2 -- resolve through paper_file(), not PAPER / name.")

    # --- evidence the documents cite ------------------------------------ #
    # NOT a copytree of the staging folder. Four of its files had drifted from
    # the artifacts they claim to be -- t_final.json there still carried the
    # superseded +0.154 recovery reading while the manuscript used +0.159, and
    # the status table pointed a reader at that stale copy. Evidence whose
    # canonical home is elsewhere is taken FROM that home at build time, so a
    # stale evidence copy cannot exist.
    for name, rel in EVIDENCE_SOURCES.items():
        if rel == "__written_by_builder__":
            continue                     # build_review_package.py writes it
        src = (STAGING / "evidence" / name) if rel is None else (CODE_ROOT / rel)
        if rel is not None and not src.exists():
            raise FileNotFoundError(
                f"evidence/{name} is declared to come from {rel}, which does "
                f"not exist. An evidence file must be a copy of the artifact "
                f"it claims to be, or it is not evidence.")
        out.append((src, f"evidence/{name}"))
    staged = {q.name for q in (STAGING / "evidence").glob("*") if q.is_file()}
    undeclared = staged - set(EVIDENCE_SOURCES)
    if undeclared:
        raise ValueError(
            f"evidence/ holds files with no declared origin: "
            f"{sorted(undeclared)}. Declare each in EVIDENCE_SOURCES -- with "
            f"its canonical path, or None if the package IS its only home.")

    # --- gate inputs: everything the thirteen gates read ----------------- #
    G = "gate_inputs"
    for rel in ("results/T5_stats/t_final.json",
                "results/T4_downstream/smoke_independence.json",
                "results/T6_symmetry/perm_sni_comparison_inventory.json",
                "results/T6_symmetry/no_oracle_band.json",
                "results/T6_symmetry/no_oracle_recovery.json",
                "results/T6_symmetry/no_oracle_noprior_band.json",
                "results/T6_lineage/data_lineage_audit.json",
                "results/T6_lineage/consumer_census.json",
                "results/T6_lineage/t53_correction_record.json",
                "results/T6_lineage/tap_input_provenance.json",
                "results/aux_windows.json",
                "docs/terminology_registry.json",
                # Gate 12 reads this to learn which figures are asset-backed.
                # Without it the gate found no assets, silently fell back to
                # the freshness rule, and applied it to the SUPERSEDED
                # generator output -- a confident answer to the wrong question.
                "docs/figure_assets.json",
                "docs/ir_status_declarations.json",
                "docs/perm_sni_inventory_declarations.json",
                # the paths the RECOMPUTED rows of the inventory name: the
                # existence check has to resolve them INSIDE the package
                "results/T6_symmetry/fair_same_host_recovery.json",
                "results/T6_symmetry/fair_same_host_recovery_cells.csv",
                # Twelfth review P0-5, generalized. The status-table
                # DECLARATIONS name these; the machine audit reads them and
                # records passed=true. They shipped under evidence/ by
                # basename, which is not the path the declaration names, so a
                # reader following the declaration found nothing. The mirror
                # exists precisely so a repository-shaped path resolves.
                "results/target_role_audit.json",
                "results/T5_stats/mask_digest_manifest.json",
                "results/T5_family/tapfam_summary.json",
                "results/T4_leakage/t42_summary.json"):
        p = CODE_ROOT / rel
        if p.exists():
            out.append((p, f"{G}/code_SNI/{rel}"))
    out.append((CODE_ROOT / "reporting" / "out", f"{G}/code_SNI/reporting/out"))
    #: The DELIVERED figure assets, at the repository path the registry names.
    #: Gate 12 stopped asking whether an asset-backed figure is fresh and
    #: started asking whether it is the REGISTERED one, byte for byte -- and a
    #: package that does not carry the file cannot answer that question. It
    #: reported "asset-backed: none" and fell through to the freshness rule on
    #: the superseded generator output instead, which is the wrong question
    #: answered confidently. Read from the registry, so a new asset ships
    #: without anyone remembering to add it.
    import json as _json
    _reg = CODE_ROOT / "docs" / "figure_assets.json"
    if _reg.exists():
        for _a in _json.loads(_reg.read_text()).get("assets", []):
            for _k in ("path", "source_pptx"):
                _p = ROOT / _a[_k]
                if _p.exists():
                    out.append((_p, f"{G}/{_a[_k]}"))
    for rel in ("reports/P5R_G_citation_registration.md",
                "reports/citation_verification.json",
                "reports/citation_support.json",
                "reports/citation_audit.md",
                "reports/bib_inventory.json",
                "reports/bibkey_consistency.json",
                "reports/bib_author_fidelity.json",
                "reports/perm_sni_comparison_inventory.md",
                "internal_review/declared_deletions.json",
                "internal_review/package_date_exemptions.json",
                "internal_review/bib_open_adjudications.json",
                # A receipt a CURRENT document points at has to be in the
                # package. P5R_H is what the ethics row's `attested` check
                # reads -- the audit recorded "attested in
                # reports/P5R_H_receipt.md" for a file the package did not
                # carry (twelfth review P0-5). P5R_V is the one
                # RESPONSE_TO_IR11 points at for the five final checks
                # (P0-4).
                "reports/P5R_H_receipt.md",
                "reports/P5R_V_receipt.md"):
        p = ROOT / rel
        if p.exists():
            out.append((p, f"{G}/{rel}"))
    # --- the code a Closed row points at ------------------------------- #
    # Derived from the declarations, not listed by hand: a row that cites a
    # script the package does not carry is a Closed state a reader cannot
    # check, which is gate 14's whole subject.
    for rel in cited_code_paths():
        p = CODE_ROOT / rel
        if p.exists():
            # code/<repo-relative path>, the same shape the gate code uses.
            # Flat placement here plus nested placement there put
            # status_table.py in the package TWICE (P5R-R): byte-identical
            # today, and one edit away from being the fork gate 16 exists to
            # catch. One artifact, one home.
            out.append((p, f"code/{rel}"))
    # --- the gates themselves, so a reviewer can re-run them ------------- #
    # The seventh review could not check why gate 14 missed a missing
    # artifact, because the gate script was not in the package. A gate a
    # reader cannot run is a claim, not a check.
    for rel in GATE_CODE:
        p = CODE_ROOT / rel
        if not p.exists():
            raise FileNotFoundError(
                f"{rel} is declared as gate code the package must carry, and "
                f"it is missing. Without it the package's all_green line "
                f"cannot be re-derived by the reader it is written for.")
        # code/reporting/... and code/experiments/..., not a flat directory:
        # the runner does `from reporting import status_table`, so a flat
        # code/ makes the shipped scripts unrunnable inside the package --
        # which is the difference between carrying a gate and carrying a
        # claim that it was run (seventh review SS12.2).
        out.append((p, f"code/{rel}"))
    out.append((ROOT / "internal_review" / "PACKAGE_GATES_HOWTO.md",
                "code/PACKAGE_GATES_HOWTO.md"))
    # The README cites this as code/docs/T70_release_procedure.md, and it is
    # what licenses INSPECTION_LOG.md's method section. A package document
    # naming a file the package does not carry is the defect class this whole
    # layout exists to prevent (P5R-R SS2).
    out.append((CODE_ROOT / "docs" / "T70_release_procedure.md",
                "code/docs/T70_release_procedure.md"))

    arch = ROOT / "VERA_GitHub" / "prereg_archive"
    if arch.exists():
        out.append((arch, f"{G}/VERA_GitHub/prereg_archive"))
    for name in ("paperY_main.log", "paperY_ESM.log"):
        p = paper_file(name)
        if p.exists():
            out.append((p, f"{G}/paper_R1/{name}"))
    return out


#: Where `--from-package` finds each repository root inside an unpacked package.
#: The gates were written against the repository shape; the package is written
#: for a reader. This is the only place the two are reconciled.
def roots(pkg: Path, paper_view: Path = None) -> dict:
    """Where each gate root lives once the package is unpacked.

    PAPER is the assembled working view and is NOT inside the package: see
    materialise_paper. Callers that already built one pass it here; the
    default is kept only so an old call site fails visibly rather than
    silently reading a directory that no longer exists.
    """
    g = pkg / "gate_inputs"
    return {"ROOT": g, "CODE_ROOT": g / "code_SNI",
            "PAPER": Path(paper_view) if paper_view else pkg / "_paper",
            "STAGING": pkg, "OUTDIR": g / "code_SNI" / "reporting" / "out"}


def materialise_paper(pkg: Path, workdir: Path = None) -> Path:
    """The gates read .tex, .bbl, .pdf and .log from one directory; the package
    keeps sources under source/ and PDFs at the top, which is what a reader
    wants. Assemble the directory the gates expect.

    It is assembled OUTSIDE the unpacked copy. It used to be built at
    pkg/_paper, which put twelve new files -- eight of them text -- into the
    very tree the gates were about to audit, and gate 18 then reported
    scanning 152 text files when the delivered ZIP holds 144. The audited tree
    must come out of the run byte-for-byte as it went in, so the working view
    the gates need is assembled somewhere else and pointed at.
    """
    import shutil
    import tempfile
    d = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="pkg_paper_"))
    d.mkdir(parents=True, exist_ok=True)
    for src in list((pkg / "source").glob("*")) + list(pkg.glob("*.pdf")) + \
            list((pkg / "gate_inputs" / "paper_R1").glob("*")):
        if src.is_file():
            shutil.copy2(src, d / src.name)
    return d
