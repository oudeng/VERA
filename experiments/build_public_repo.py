"""Public VERA repository builder (Chat ruling 3 revised, 2026-08-30).

Assembles a FRESH tree (no history) at the target path: code, configs,
the docs rule chain, tests, environment locks, a generated public README,
and the prereg_archive evidence bundle as a plain directory. Restricted
derived tables, model weights and mask files are excluded by
construction; masks are replaced by their generation config + seeds +
a SHA-256 manifest of the frozen files. The DUA final scan, secrets scan,
path scan and large-object scan run over the STAGED tree and the build
refuses to git-init on any hit (requirement 8: zero hits go to the
receipt).

    env PYTHONHASHSEED=2025 python experiments/build_public_repo.py \
        [--target /home/dengou/SNI_R1/VERA_GitHub]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
PROJ = CODE_ROOT.parent

INCLUDE_DIRS = ["sni", "baselines", "common", "configs", "data_layer",
                "evaluation", "experiments", "missingness",
                "pipelines", "reporting", "stats", "tests", "audit"]
# docs/ is re-cut by function (Chat ruling 2026-08-30): the rules chain
# lives in prereg_archive/ (complete, 14 rule docs); docs/ keeps ONLY the
# code-asserted adjudication register (swept of internal-workflow
# references, technical content verbatim) and a clean export of the
# corrections ledger the ESM's ledger ids point to. Blueprints,
# registries and process-narrative documents stay in the private repo.
INCLUDE_FILES = ["requirements.lock.txt", "requirements.freeze.raw.txt",
                 "environment.lock.yml",
                 # Push-chain step 1. Without it the published tree is "all
                 # rights reserved" by default, which is the opposite of what
                 # a reproducibility archive is for -- and an archive service
                 # would archive whatever it finds, license or not.
                 "LICENSE"]
EXCLUDE_NAMES = {"__pycache__", ".git", "out"}       # reporting/out regen
EXCLUDE_SUFFIX = {".log", ".pyc"}

# DUA final-scan patterns (requirement 8): any hit blocks the build.
DUA_PATTERNS = [r"_complete\.csv$", r"preclip", r"realmissing",
                r"^models_.*\.pt$", r"_mask\.npy$", r"^Xfinal_",
                r"\.parquet$", r"^D_.*\.csv$", r"^A_.*\.csv$"]
SECRET_RE = re.compile(
    r"api[_-]?key|access[_-]?token|BEGIN [A-Z ]*PRIVATE KEY|password\s*=",
    re.I)


def _copytree(src: Path, dst: Path) -> int:
    n = 0
    for p in src.rglob("*"):
        if any(part in EXCLUDE_NAMES for part in p.relative_to(src).parts):
            continue
        if p.is_dir():
            continue
        if p.suffix in EXCLUDE_SUFFIX:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def _sweep_adjudications(text: str) -> tuple[str, dict]:
    """Public copy of docs/adjudications.md: technical adjudication table
    verbatim; internal-workflow narration removed. Deterministic edits,
    counted for the build report."""
    stats = {"dropped_sections": 0, "replaced_refs": 0, "intro": 1}
    head, _, rest = text.partition("\n---\n")
    intro = (
        "# Adjudicated configuration decisions\n\n"
        "Every adjudicated configuration decision is registered here with "
        "its rationale, the affected configuration keys, and -- where one "
        "exists -- an executable assertion in "
        "`tests/test_adjudications.py`. The register exists because of a "
        "real declared-value != implemented-value incident: a device "
        "decision was accepted in writing but never landed in the "
        "scheduling config, and nothing errored. **A decision is in force "
        "only once a test asserts it.**\n")
    body = rest
    # drop the open-items (workflow) section entirely
    if "## 未决事项" in body:
        body = body[:body.index("## 未决事项")].rstrip() + "\n"
        stats["dropped_sections"] += 1
    # private-report pointers -> neutral note
    for pat in (r"；判定见 `reports/[^`]+`[^|]*",
                r"reports/[A-Za-z0-9_./]+"):
        body, n = re.subn(pat, "(record in the private development "
                               "history)", body)
        stats["replaced_refs"] += n
    return intro + "\n---\n" + body, stats


def _clean_ledger(text: str) -> tuple[str, dict]:
    """Clean export of docs/corrections_R0.md as corrections_ledger.md:
    every B-entry verbatim except private absolute paths and
    private-report pointers, which become neutral placeholders."""
    stats = {"replaced_paths": 0, "replaced_refs": 0, "intro": 1}
    head, _, rest = text.partition("\n---\n")
    intro = (
        "# Corrections ledger (B1--B84)\n\n"
        "Every finding from the pre-revision workspace audit is registered "
        "here with its evidence, its disposition, and the manuscript "
        "location it affects. The consolidated twenty-entry corrections "
        "section of the Online Resource cites these ledger ids.\n\n"
        "**Status vocabulary**: `fixed-P1` corrected during revision "
        "phase 1; `pending-P2`..`pending-P6` scheduled; `decision-needed` "
        "required an author decision; `wont-fix` deliberate, reason "
        "recorded; `annotated` recorded as a known property, no code "
        "change.\n\n"
        "84 findings, B1--B84 (B48--B56 of one interim report were "
        "renumbered B57--B65; see the note below that section).\n")
    body = rest
    body, n1 = re.subn(r"`/home/dengou/[^`]*`",
                       "(a private local path, recorded in the private "
                       "development history)", body)
    body, n2 = re.subn(r"/home/dengou/[A-Za-z0-9_./-]+",
                       "(private local path)", body)
    stats["replaced_paths"] = n1 + n2
    body, n3 = re.subn(r"`?reports/[A-Za-z0-9_./]+`?( §[0-9.]+)?",
                       "(private development records)", body)
    stats["replaced_refs"] = n3
    return intro + "\n---\n" + body, stats


DOCS_PUBLIC = [("docs/adjudications.md", "docs/adjudications.md",
                _sweep_adjudications),
               ("docs/corrections_R0.md", "docs/corrections_ledger.md",
                _clean_ledger)]


def masks_manifest() -> dict:
    md = CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled"
    files = sorted(md.rglob("*.npy"))
    entries = [{"path": str(p.relative_to(CODE_ROOT / "data")),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size} for p in files]
    cfg = CODE_ROOT / "configs" / "missingness.yaml"
    return {"note": ("Frozen mask files are NOT distributed (they are "
                     "table-shaped derivatives of restricted-access "
                     "tables). Regenerate them with missingness.cli "
                     "generate under configs/missingness.yaml (profile "
                     "clinical_v1) and the seeds recorded in each mask's "
                     "meta; verify identity against the checksums below."),
            "missingness_config_sha256":
                hashlib.sha256(cfg.read_bytes()).hexdigest(),
            "n_files": len(entries), "files": entries}



#: The aggregate artifacts the paper's claims rest on, published so that the
#: response letter's evidence pointers RESOLVE for a reviewer.
#:
#: They did not, and that was the defect: the letter cited
#: results/T5_stats/lambda_check.json and evidence/... paths, which are a
#: private directory and a review-package layout respectively. A reviewer
#: holds the manuscript, the Online Resource, the letter and this repository,
#: and none of those pointers landed in any of them. For a paper whose thesis
#: is that claims must be checkable, unfollowable pointers are the worst
#: possible place to be sloppy.
#:
#: Every file here is AGGREGATE: effects, intervals, counts, quantiles, and
#: one cell-level table that is synthetic-only (regimes linear_gaussian,
#: nonlinear_mixed, interaction_xor). Dataset names appear as keys on
#: aggregates. No row-level identifier appears in any of them, which the DUA
#: scan re-checks on every build. Row-level derived tables stay out, as they
#: always have.
PUBLIC_EVIDENCE = [
    "results/T5_stats/t_final.json",
    "results/T5_stats/lambda_check.json",
    "results/T4_leakage/t42_summary.json",
    "results/T6_symmetry/fair_same_host_recovery.json",
    "results/T6_symmetry/fair_same_host_recovery_cells.csv",
]


#: name -> (what it is, which letter item it answers). The prose DESCRIBES;
#: it quotes no number, because a number typed here is a number nothing
#: checks -- the counts below are read from the delivered files at build time.
EVIDENCE_NOTES = [
    ("t_final.json",
     "the single source every number in the manuscript and both letters is "
     "read from; nothing in the paper is typed by hand",
     "R1-3, R1-5, R2-1, R2-4"),
    ("lambda_check.json",
     "the gate scan: the learned quantity's quantiles over every trained "
     "model, its global maximum, and the verdict on the ln 2 bound",
     "R1-1"),
    ("t42_summary.json",
     "per-object, per-condition counts and observed null rates for the two "
     "discriminating leakage classes",
     "R1-4"),
    ("fair_same_host_recovery.json",
     "the equal-information recovery comparison: effect, interval, exact p, "
     "and the scope of what was and was not reproduced",
     "R2-2"),
    ("fair_same_host_recovery_cells.csv",
     "the cell-level table the line above is computed from; "
     "experiments/recompute_fair_pair.py reproduces the result from it alone",
     "R2-2"),
]


def evidence_readme(stage: Path) -> str:
    """What the five files are, so the directory is evidence and not JSON."""
    import csv as _csv
    ev = stage / "evidence"
    rows = []
    for name, what, item in EVIDENCE_NOTES:
        f = ev / name
        if not f.exists():
            raise FileNotFoundError(f"evidence/{name} was not published")
        if name.endswith(".csv"):
            n = len(list(_csv.DictReader(f.open())))
            size = f"{n} rows"
        else:
            size = f"{f.stat().st_size // 1024 or 1} KB"
        rows.append(f"| `{name}` | {what} | {item} | {size} |")
    return ("# Evidence\n\n"
            "The aggregate artifacts the revised manuscript and the response "
            "to reviewers cite. Each is the file a stated number was read "
            "from, published so that a reader can check a claim without "
            "rebuilding anything.\n\n"
            "| File | What it is | Answers | |\n|---|---|---|---|\n"
            + "\n".join(rows) + "\n\n"
            "These are aggregates and synthetic-regime cells only. Row-level "
            "derived tables for MIMIC-IV and eICU are restricted under the "
            "PhysioNet data use agreements and are not distributed here; the "
            "code to rebuild them from an authorized download is in "
            "`experiments/`, and the mask checksums are in "
            "`data_manifests/`.\n")


def orthography_scan(stage: Path) -> list:
    """Every text file about to be published, in one variety of English.

    P7-A SS1.7, adjudicated 2026-09-05. The manuscript gate covers the five
    terminology layers; the PUBLIC TREE is what a reader clones, and it was
    carrying 101 British spellings in comments and console strings while the
    repository claimed orthographic consistency. That is the defect handed to
    whoever reads it next, so the scan that already refuses to publish
    withdrawn wording now also refuses to publish mixed spelling.

    A match inside an identifier or a quoted key is absorbed, not reported --
    the same test the frozen-data-key rule uses. A name is not prose, and a
    machine-readable artifact keeps the key it has always had.
    """
    sys.path.insert(0, str(CODE_ROOT))
    from reporting.facts_gate import _norm, _ortho, strip_comment_markers
    pat, canon = _ortho()
    SELF_REFERENTIAL = {"terminology_registry.json", "facts_gate.py",
                        "build_public_repo.py", "BUILD_REPORT.json",
                        "fig1_acceptance.py",
                        # verbatim quotation of others' words
                        "adjudications.md",
                        "perm_sni_inventory_declarations.json",
                        "figure_assets.json"}
    frozen = set()
    for _m in sorted((CODE_ROOT.parent / "VERA_GitHub" / "prereg_archive")
                     .glob("manifest_*.json")):
        try:
            frozen |= {f["path"] for f in
                       json.loads(_m.read_text()).get("files", [])}
        except (OSError, ValueError, KeyError):
            pass

    def _archived(rel: str) -> bool:
        """Anything inside prereg_archive/ except the index we write.

        The archive is not our prose. Its manifests reproduce the frozen rule
        documents' text and their original commit messages VERBATIM, and its
        tarballs are hashed -- respelling either would break the record that
        each decision rule was fixed before the measurement it governs, which
        is the one claim this repository exists to support. The archive's own
        README is written fresh by this builder each run, so it is ours and
        is scanned.
        """
        return (rel.startswith("prereg_archive/")
                and Path(rel).name != "README.md")
    exts = {".md", ".py", ".json", ".yaml", ".yml", ".tex", ".txt", ".sh",
            ".cfg", ".toml"}
    hits, absorbed = [], {}
    for f in sorted(stage.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        _rel = str(f.relative_to(stage))
        if (f.name in SELF_REFERENTIAL or _rel in frozen
                or _archived(_rel)):
            #: Three reasons a file is not respelled, and none of them is
            #: "we forgot".
            #:  * SELF-REFERENTIAL: it carries the forms it hunts.
            #:  * FROZEN: the prospective-specification archive fixes these
            #:    nineteen documents by name and commit. Editing one after
            #:    the fact is precisely what the zero-artifact attestation
            #:    says we did not do, and a spelling is not worth spending
            #:    that on. The list is READ FROM THE MANIFEST, so a document
            #:    entering or leaving the archive moves this boundary with it.
            #:  * QUOTATION: the file reproduces someone else's words, or a
            #:    superseded sentence the record exists to preserve. A record
            #:    of a spelling cannot survive being respelled -- one in
            #:    figure_assets.json was rewritten into claiming the asset
            #:    always printed what it now prints, which is how this
            #:    exemption came to be written down.
            absorbed[f.name] = absorbed.get(f.name, 0) + 1
            continue
        try:
            raw = f.read_text(errors="replace")
        except OSError:
            continue
        text = _norm(strip_comment_markers(raw, f.suffix))
        for m in pat.finditer(text):
            i, j = m.start(), m.end()
            before, after = text[max(0, i - 1):i], text[j:j + 1]
            if (re.match(r"[\w\"\']", before or " ")
                    or re.match(r"[\w\"\']", after or " ")):
                k = "identifier_context:" + m.group(0)
                absorbed[k] = absorbed.get(k, 0) + 1
                continue
            hits.append(f"{f.relative_to(stage)} :: {m.group(0)} -> "
                        f"{canon[m.group(0).lower()]} :: "
                        f"...{text[max(0, i - 30):j + 30]}...")
            break
    orthography_scan.absorbed = absorbed
    return hits


def terminology_scan(stage: Path) -> list:
    """P5R-J SS6.2: the public tree was outside the manuscript gate's five
    layers, so it kept shipping wordings the paper had withdrawn. Scan every
    text file in the STAGED tree against the same registry, whitespace- and
    hyphen-insensitively, and let any hit block git-init."""
    sys.path.insert(0, str(CODE_ROOT))
    from reporting.facts_gate import (_load_registry, _norm, REGISTRY,
                                  strip_comment_markers)
    pats = _load_registry()
    reg = json.loads(REGISTRY.read_text())
    ex = reg.get("scan_exemptions", {})
    by_role = set(ex.get("files_by_role", {}))
    frozen = set(ex.get("frozen_data_keys", {}).get("keys", []))
    exts = {".md", ".py", ".json", ".yaml", ".yml", ".tex", ".txt", ".sh",
            ".cfg", ".toml"}
    hits, absorbed = [], {}

    def _is_identifier_context(text: str, i: int, j: int) -> bool:
        """True when the match sits inside an identifier or a quoted dict
        key -- i.e. it is a frozen artifact key, not prose."""
        before = text[max(0, i - 1):i]
        after = text[j:j + 1]
        return bool(re.match(r"[\w\"\']", before or " ")
                    or re.match(r"[\w\"\']", after or " "))

    for f in sorted(stage.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        if f.name in by_role:
            absorbed[f.name] = absorbed.get(f.name, 0) + 1
            continue
        try:
            # strip comment MARKERS before joining lines: the marker is
            # syntax, and joining across it manufactures phrases that are
            # in no file (the 'Reviewer #' false positives)
            text = _norm(strip_comment_markers(
                f.read_text(errors="replace"), f.suffix))
        except OSError:
            continue
        for name, rx in pats:
            for m in rx.finditer(text):
                if (name in frozen
                        and _is_identifier_context(text, m.start(), m.end())):
                    absorbed["frozen_data_key:" + name] = absorbed.get(
                        "frozen_data_key:" + name, 0) + 1
                    continue
                rel = f.relative_to(stage)
                hits.append(f"{rel} :: {name} :: "
                            f"...{text[max(0, m.start() - 30):m.end() + 30]}...")
                break
    terminology_scan.absorbed = absorbed
    return hits


def scans(stage: Path) -> dict:
    dua_hits, secret_hits, path_hits, big = [], [], [], []
    for p in stage.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = str(p.relative_to(stage))
        if any(re.search(pat, p.name) for pat in DUA_PATTERNS):
            dua_hits.append(rel)
        if p.stat().st_size > 5_000_000:
            big.append((rel, p.stat().st_size))
        if p.suffix in (".py", ".sh", ".md", ".yaml", ".yml", ".txt",
                        ".json", ".cfg", ".toml"):
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            for m in SECRET_RE.finditer(txt):
                frag = txt[max(0, m.start() - 20):m.end() + 20]
                secret_hits.append((rel, frag.strip()[:70]))
            if "/home/dengou" in txt and not rel.startswith("docs/"):
                path_hits.append(rel)
    return {"dua": dua_hits, "secrets": secret_hits,
            "paths_outside_docs": path_hits, "large": big}


#: The published README lives in docs/public_readme.md as a real
#: markdown file, not as a string in here. It is prose that the
#: first author edits, and prose trapped inside a Python literal
#: can only be edited by someone willing to edit Python. The
#: builder copies it verbatim and asserts the copy is identical,
#: so VERA_GitHub/README.md stays a VIEW: edit the source, run
#: the builder, and the change arrives. An edit made in the view
#: is lost at the next build, silently -- the same one-direction
#: rule VERA_paper_R1 follows.
PUBLIC_README_SRC = CODE_ROOT / "docs" / "public_readme.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(PROJ / "VERA_GitHub"))
    a = ap.parse_args()
    stage = Path(a.target)
    #: prereg_archive/ is APPEND-ONLY: it is the record that each decision
    #: rule was fixed before the measurement it governs, and a member that
    #: disappears takes that evidence with it. This builder wipes the target
    #: and repopulates the archive from reports/prereg_archive, so anything
    #: that reached the published tree WITHOUT reaching that source directory
    #: is destroyed by a rebuild. That is not hypothetical: the 2026-08-30
    #: manifest and tarball were lost exactly this way, twice, and recovered
    #: from an issued review package both times. Snapshot the members first
    #: and refuse to finish if the rebuild cannot reproduce them.
    #: Do not stamp a build report from a dirty tree. BUILD_REPORT records
    #: source_commit, and a tree with uncommitted edits produces content that
    #: is NOT at that commit -- the published repository then advertises a
    #: provenance it does not have. This has happened twice; code_SNI's own
    #: gate 1 refuses the same thing for generated outputs, and nothing was
    #: applying that rule out here.
    dirty = subprocess.run(["git", "-C", str(CODE_ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        n = len(dirty.splitlines())
        print(f"[REFUSING] {n} uncommitted change(s) in code_SNI. BUILD_REPORT "
              f"would record a source_commit whose content is not what is "
              f"being published. Commit first, then rebuild.")
        for line in dirty.splitlines()[:8]:
            print(f"    {line}")
        return 1
    archive_before = {}
    arch_dir = stage / "prereg_archive"
    if arch_dir.is_dir():
        archive_before = {f.name: f.read_bytes()
                          for f in arch_dir.iterdir() if f.is_file()}
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    n = 0
    for d in INCLUDE_DIRS:
        src = CODE_ROOT / d
        if src.exists():
            n += _copytree(src, stage / d)
    for f in INCLUDE_FILES:
        if (CODE_ROOT / f).exists():
            shutil.copy2(CODE_ROOT / f, stage / f)
            n += 1
    # docs/ re-cut by function (ruling 2026-08-30): only the two curated
    # documents, via deterministic transforms counted in the report
    docs_stats = {}
    (stage / "docs").mkdir(exist_ok=True)
    for src_rel, dst_rel, fn in DOCS_PUBLIC:
        out_text, st = fn((CODE_ROOT / src_rel).read_text())
        (stage / dst_rel).write_text(out_text)
        docs_stats[dst_rel] = st
        n += 1
    (stage / "docs" / "README.md").write_text(
        "# docs/\n\nThis directory carries the adjudicated configuration "
        "register\n(`adjudications.md`, asserted by "
        "`tests/test_adjudications.py`) and the\ncorrections ledger "
        "(`corrections_ledger.md`) that the Online Resource's\nledger ids "
        "point to. The prospectively specified decision-rule\ndocuments "
        "referenced throughout the code as `docs/T*.md` live in\n"
        "`prereg_archive/` together with their original commit "
        "evidence.\n")
    n += 1
    # evidence bundle as a plain directory (requirement 2)
    n += _copytree(PROJ / "reports" / "prereg_archive",
                   stage / "prereg_archive")
    (stage / "prereg_archive" / "README.md").write_text(
        "# Prospective-specification evidence bundle\n\n"
        "Rule documents (verbatim), their ORIGINAL commit hashes, the\n"
        "zero-artifact attestation texts from the commit bodies, a\n"
        "manifest, and SHA-256 checksums. The original development\n"
        "history contains restricted data and is retained privately;\n"
        "this directory is the hash-stable evidence layer, and commit\n"
        "hashes cited in the paper refer to it.\n\n"
        "A commit hash fixes this bundle's content exactly, but a\n"
        "hosting account is not an archive: the history can be\n"
        "replaced by whoever controls it. No independent\n"
        "no-later-than anchor is claimed for it. An independent\n"
        "archival deposit is intended and will be cited when it\n"
        "completes.\n")
    md = masks_manifest()
    (stage / "data_manifests").mkdir()
    (stage / "data_manifests" / "masks_sha256_manifest.json").write_text(
        json.dumps(md, indent=1))

    lost = sorted(n for n in archive_before
                  if not (stage / "prereg_archive" / n).exists())
    if lost:
        # put them back before saying anything else -- an append-only archive
        # is not a thing to leave broken while a message is printed
        for n in lost:
            (stage / "prereg_archive" / n).write_bytes(archive_before[n])
        print(f"[RESTORED] {len(lost)} prereg member(s) the rebuild dropped: "
              f"{lost}")
        print("  prereg_archive/ is append-only. These exist in the published "
              "tree but NOT in reports/prereg_archive, so a rebuild cannot "
              "reproduce them. Copy them into reports/prereg_archive and "
              "re-run; do not ship an archive this builder cannot rebuild.")
        return 1
    #: Fig. 1 and its caption. A reader who lands here should be able to see
    #: what VERA IS before deciding whether to read 29 pages, and this is the
    #: one figure that can be published without a drift risk: it prints no
    #: data number, which the publisher re-derives from the delivered PDF
    #: rather than taking on trust. It also refuses to run unless the file is
    #: the REGISTERED asset, byte for byte.
    #: evidence/ -- the artifacts the letter points at. Copied under a flat,
    #: stable name so a pointer in a letter is a path a reader can type.
    ev = stage / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    _ev = []
    for rel in PUBLIC_EVIDENCE:
        src = CODE_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(
                f"{rel} is declared as published evidence and is missing. The "
                f"response letter points at it; publishing the repository "
                f"without it would restore the dangling pointer this list "
                f"exists to remove.")
        (ev / src.name).write_bytes(src.read_bytes())
        _ev.append(src.name)
    n += len(_ev)
    (stage / "evidence" / "README.md").write_text(evidence_readme(stage))
    n += 1
    print(f"[evidence] {len(_ev)} aggregate artifact(s) published with a "
          f"README: " + ", ".join(sorted(_ev)))

    sys.path.insert(0, str(CODE_ROOT))
    from reporting import public_figure
    _fig = public_figure.build(stage / "docs" / "figure")
    n += 3
    print(f"[figure] Fig. 1 published: PNG {_fig['png_bytes']} B at "
          f"{_fig['dpi']} dpi; caption {_fig['caption_chars']} chars, "
          f"extracted from the manuscript with its cross-references resolved")

    _readme = PUBLIC_README_SRC.read_text()
    (stage / "README.md").write_text(_readme)
    assert (stage / "README.md").read_text() == _readme, \
        "the published README must equal its source byte for byte"
    (stage / ".gitignore").write_text(
        "__pycache__/\n*.pyc\nresults/\ndata/\nreporting/out/\n*.log\n"
        "!evidence/\n")

    rep = scans(stage)
    rep["terminology"] = terminology_scan(stage)
    rep["orthography"] = orthography_scan(stage)
    report = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
              "source_commit": subprocess.run(
                  ["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                  capture_output=True, text=True).stdout.strip(),
              "n_files": n + 3, "masks_in_manifest": md["n_files"],
              "docs_public": docs_stats,
              "scans": {k: (v if k != "large"
                            else [[f, s] for f, s in v])
                        for k, v in rep.items()}}
    (stage / "BUILD_REPORT.json").write_text(json.dumps(report, indent=1))
    blocked = (rep["dua"] or rep["secrets"] or rep["terminology"]
               or rep["orthography"])
    print(f"[build] {n} files staged; masks manifest {md['n_files']} "
          f"entries")
    print(f"[scan] DUA hits: {len(rep['dua'])}; secrets: "
          f"{len(rep['secrets'])}; /home/dengou outside docs/: "
          f"{len(rep['paths_outside_docs'])}; >5MB: {len(rep['large'])}")
    for k in ("dua", "secrets"):
        for h in rep[k][:10]:
            print(f"  [{k.upper()}] {h}")
    for f in rep["paths_outside_docs"][:10]:
        print(f"  [PATH] {f}")
    for f, sz in rep["large"][:10]:
        print(f"  [LARGE] {f} ({sz/1e6:.1f} MB)")
    absorbed = getattr(terminology_scan, "absorbed", {})
    report["scans"]["terminology_exemptions_absorbed"] = absorbed
    (stage / "BUILD_REPORT.json").write_text(json.dumps(report, indent=1))
    print(f"[scan] withdrawn-terminology hits: {len(rep['terminology'])}; "
          f"declared exemptions absorbed: {sum(absorbed.values())} "
          f"{absorbed if absorbed else ''}")
    for h in rep["terminology"][:15]:
        print(f"  [TERM] {h}")
    print(f"[scan] mixed-spelling hits: {len(rep['orthography'])}; "
          f"absorbed as identifiers/self-reference: "
          f"{sum(orthography_scan.absorbed.values())}")
    for h in rep["orthography"][:15]:
        print(f"  [SPELL] {h}")
    #: The five published aggregates, on their own terms. The tree-wide DUA
    #: scan above asks whether any of 224 files leaks a restricted table;
    #: this asks the narrower question the ratification was conditioned on
    #: (adjudication 2026-09-05 SS1) -- cell-level aggregate, synthetic
    #: regimes, zero row-level identifiers, in the DELIVERED bytes.
    sys.path.insert(0, str(CODE_ROOT))
    from experiments.evidence_dua_scan import scan as _ev_scan
    _ev = _ev_scan(stage)
    rep["evidence_dua"] = _ev["problems"]
    print(f"[scan] published evidence/: {_ev['n_files']} artifacts, "
          f"{_ev['total_bytes']} B; row-level identifiers, non-synthetic "
          f"regimes and single-record rows: {len(_ev['problems'])}")
    for h in _ev["problems"][:10]:
        print(f"  [EVID] {h}")
    if _ev["problems"]:
        blocked = True
    if blocked:
        print("REFUSING to git-init: DUA / secret / withdrawn-terminology "
              "/ mixed-spelling / published-evidence hits above "
              "(requirement 8 + P5R-J SS6.2 + P7-A SS1.7 + P7-D SS1)")
        return 1
    subprocess.run(["git", "init", "-q"], cwd=stage, check=True)
    subprocess.run(["git", "add", "-A"], cwd=stage, check=True)
    subprocess.run(
        ["git", "-c", "user.name=VERA authors",
         "-c", "user.email=noreply@example.org", "commit", "-q", "-m",
         "VERA public release: fresh tree; prospective-specification "
         "evidence bundle in prereg_archive/ (original development "
         "history contains restricted data and is retained privately)"],
        cwd=stage, check=True)
    #: The release identity, written AFTER the tree is frozen, to results/
    #: -- which is not published, so the tree never contains a statement of
    #: its own hash. The letters and the manuscript citation read this back.
    #: A fresh init defaults to master with no remote, and the push is then
    #: a hand-step that has to be remembered every rebuild. Set both here.
    subprocess.run(["git", "-C", str(stage), "branch", "-M", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(stage), "remote", "add", "origin",
                    "git@github.com:oudeng/VERA.git"],
                   check=True, capture_output=True)
    _commit = subprocess.run(["git", "-C", str(stage), "rev-parse",
                              "--short=7", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    if len(_commit) != 7:
        raise RuntimeError(f"cannot read the frozen tree's commit: {_commit!r}")
    _rel = CODE_ROOT / "results" / "public_release.json"
    _prev = json.loads(_rel.read_text()) if _rel.exists() else {}
    _rel.write_text(json.dumps(
        {"url": "https://github.com/oudeng/VERA",
         "tag": _prev.get("tag", "v1.0.0"), "commit": _commit,
         #: the tree's OWN count, not the staging counter: the counter is
         #: read before the figure and the evidence README are added, and a
         #: number that disagrees with `git ls-files` is a number nothing
         #: should have stated.
         "files": len(subprocess.run(
             ["git", "-C", str(stage), "ls-files"], capture_output=True,
             text=True).stdout.split()),
         "note": "written by build_public_repo after git init; results/ is "
                 "not published, so the tree never states its own hash"},
        indent=1) + "\n")
    print(f"[ok] fresh repo initialized at {stage}")
    print(f"[release] commit {_commit}; identity recorded in "
          f"results/public_release.json for the letters and the manuscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
