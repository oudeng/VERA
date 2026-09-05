"""Build the review package from one declaration, and check it against itself.

P5R-N SS5, from the sixth review's finding that the delivered ZIP did not
contain the evidence its own documents pointed at, and that README,
CROSSCHECK, STATUS_TABLE and GATES disagreed with each other on page counts,
rule counts, citation counts and gate counts.

Two properties this build has that hand-assembly did not:

  1. the file list is `experiments/package_layout.py`, which is also what
     `package_gates.py --from-package` reads back, so a file the gates need
     and the build forgets fails the gate run on the package rather than
     passing quietly in the working tree;
  2. after zipping, it re-reads the package and checks every count its own
     documents state -- pages, gates, rule documents, references, citation
     points -- against what the package actually holds.

    env PYTHONHASHSEED=2025 python experiments/build_review_package.py \
        --out reports/rep0829_6.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "experiments"))


def assemble(dest: Path) -> dict:
    from package_layout import entries
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for src, rel in entries():
        tgt = dest / rel
        if not src.exists():
            missing.append(str(src))
            continue
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, tgt, dirs_exist_ok=True)
            copied += [str(Path(rel) / f.relative_to(src))
                       for f in src.rglob("*") if f.is_file()]
        else:
            shutil.copy2(src, tgt)
            copied.append(rel)
    return {"copied": copied, "missing_sources": missing}


def _pdf_pages(p: Path) -> int:
    out = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s*(\d+)", out)
    return int(m.group(1)) if m else -1


#: How many states the status table defines, read from the module that
#: defines them -- a label that says "five-state" is checkable only against
#: something that knows the answer (eighth review P1-7).
def _n_states() -> int:
    sys.path.insert(0, str(CODE_ROOT))
    from reporting.status_table import STATES
    return len(STATES)


try:
    _N_STATES = _n_states()
except Exception:                                  # pragma: no cover
    _N_STATES = 0


def _banned_archive_phrases() -> list:
    """The prohibited phrases, read from the ONE place that defines them.

    This declaration has to name what it forbids, and the terminology scanner
    has to flag the phrase wherever it appears -- so writing the literals here
    made the builder's own rule text a hit against the rule. Same shape as the
    retired label, same answer: one literal home, everybody else reads it.
    The home is the terminology registry, which is what the scanner loads.
    """
    reg = json.loads((CODE_ROOT / "docs" / "terminology_registry.json"
                      ).read_text())
    for e in reg.get("terms", []):
        if e.get("canonical") == "name the archive":
            return list(e.get("banned", []))
    raise KeyError("terminology_registry.json has no 'name the archive' entry")


def audit_counts(pkg: Path) -> dict:
    """Every count a package document states, against the package itself.

    The sixth review found four documents disagreeing on page counts, rule
    counts, citation counts and gate counts. A number about the package is a
    claim about the package, and this is where it is checked.
    """
    facts, bad = {}, []
    for name in ("paperY_main.pdf", "paperY_ESM.pdf"):
        if (pkg / name).exists():
            facts[name] = _pdf_pages(pkg / name)
    man = sorted((pkg / "gate_inputs" / "VERA_GitHub" / "prereg_archive")
                 .glob("manifest_*.json"))
    if man:
        facts["rule_documents"] = len(json.loads(man[-1].read_text())["files"])
    # the number a reader can count: entries in the rendered bibliography.
    # citation_verification.json's n_references counts declaration RECORDS,
    # which include works withdrawn from the text and still on file.
    bbl = pkg / "source" / "paperY_main.bbl"
    if bbl.exists():
        facts["references"] = len(re.findall(r"\\bibitem", bbl.read_text()))
    cs = pkg / "gate_inputs" / "reports" / "citation_support.json"
    if cs.exists():
        facts["citation_points"] = json.loads(cs.read_text())["summary"]["n_points"]
    gt = pkg / "GATES.txt"
    if gt.exists():
        #: NUMBERED gate lines only. The transcript also carries the outer
        #: read-only invariant, which is not a gate and must not inflate the
        #: count every document in the package is checked against.
        facts["gates"] = len(set(re.findall(
            r"^\[(?:GREEN|RED)\s*\]\s*(\d+) ", gt.read_text(), re.M)))
    #: The gate count above is read off the transcript, so a STALE transcript
    #: agrees with itself and the audit sees nothing. Four rebuilds shipped a
    #: GATES.txt from an earlier commit for exactly that reason. The transcript
    #: must therefore be checked against something it cannot supply: the commit
    #: this build is being made from. (Packaging gate 19 makes the same check
    #: from inside the delivered ZIP; this one fails earlier, at the build.)
    facts["_transcript_head"] = (
        m.group(1) if gt.exists() and
        (m := re.search(r"HEAD=([0-9a-f]{7,40})", gt.read_text())) else "")

    words = {13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
             17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
    claims = [
        (r"主文[^0-9\n]{0,20}?(\d{1,3})\s*页", "paperY_main.pdf"),
        (r"ESM[^0-9\n]{0,20}?(\d{1,3})\s*页", "paperY_ESM.pdf"),
        (r"(\d{1,3})\s*(?:道)?\s*(?:打包)?门", "gates"),
        (r"(\d{1,3})\s*份规则文档", "rule_documents"),
        (r"被引文献\s*\**(\d{1,3})", "references"),
        (r"引用点\s*\**(\d{1,3})", "citation_points"),
    ]
    #: A document may declare itself a record of an earlier round. It is then
    #: exempt, because its counts are true of the round it describes -- but the
    #: declaration has to be IN the document, on its first lines, where a
    #: reader meets it (P5R-N SS5.4).
    HIST = ("历史记录", "historical record", "describes the state at")
    exf = ROOT / "internal_review" / "package_count_exemptions.json"
    EX = json.loads(exf.read_text())["exemptions"] if exf.exists() else []
    absorbed = []
    #: Every Markdown document in the package, not just the top-level ones.
    #: README says the count audit covers "any document in the package"; it
    #: globbed one directory, so code/PACKAGE_GATES_HOWTO.md -- which states a
    #: gate count -- was never audited.
    for doc in sorted(pkg.rglob("*.md")):
        raw = doc.read_text(errors="replace")
        if any(h in raw[:600] for h in HIST):
            facts.setdefault("_historical", []).append(doc.name)
            continue
        text = " ".join(raw.split())
        for pat, key in claims:
            if key not in facts:
                continue
            PAST = ("当时", "彼时", "曾为", "原为")
            for m in re.finditer(pat, text):
                n = int(m.group(1))
                if "第" in m.group(0):
                    continue
                if any(q in text[max(0, m.start() - 8):m.start()] for q in PAST):
                    absorbed.append(f"{doc.name}:{m.group(0)} (marked past)")
                    continue
                if n != facts[key]:
                    hit = [e for e in EX if e["file"] == doc.name
                           and e["fragment"] in m.group(0)]
                    if hit:
                        absorbed.append(f"{doc.name}:{hit[0]['fragment']}")
                        continue
                    bad.append(f"{doc.name}: \"{m.group(0)[:34]}\" but the "
                               f"package has {facts[key]}")
        for n, w in words.items():
            if "rule_documents" in facts and n != facts["rule_documents"]:
                if f"{w} decision-rule" in text:
                    bad.append(f"{doc.name}: says {w} decision-rule documents, "
                               f"package has {facts['rule_documents']}")
        # Eighth review P1-7 / SS5.5: a count spelled as a WORD inside a label
        # is still a count claim, and four of them were stale ("five-state
        # status table", "the fifteen gates"). They change no computation,
        # which is why nobody noticed, and why a reader who does notice
        # trusts the rest less. Same scan surface as the numeric claims.
        def _exempt(phrase: str) -> bool:
            hit = [e for e in EX if e["file"] == doc.name
                   and e["fragment"] in phrase]
            if hit:
                absorbed.append(f"{doc.name}:{hit[0]['fragment']}")
            return bool(hit)

        for n, w in words.items():
            if "gates" in facts and n != facts["gates"]:
                for phrase in (f"{w} gates", f"{w} packaging gates",
                               f"{w}道门", f"{w} 道门"):
                    if phrase in text and not _exempt(phrase):
                        bad.append(f"{doc.name}: says '{phrase}', package has "
                                   f"{facts['gates']}")
        # ...and the table under a GATE heading has to agree with it. Which
        # table that is has to be decided by what precedes it: the inspection
        # log's "ten defect families" is also numbered 1..N, and a rule that
        # counted every numbered table flagged it as a short gate list.
        GATE_HEAD = re.compile(r"(?:packaging gates|打包门)", re.I)
        #: Walk the document and remember which heading we are under. The
        #: previous version looked back a fixed 1200 characters, so a long
        #: enough table lost its own last rows -- it read the nineteen-row
        #: gate table as eighteen rows and would have reported the shortfall
        #: as a documentation error rather than as its own blind spot. A
        #: window that silently truncates is the same defect as a search that
        #: silently narrows.
        rows, under_gate_heading = [], False
        for line in raw.splitlines():
            if line.startswith("#"):
                under_gate_heading = bool(GATE_HEAD.search(line))
                continue
            m = re.match(r"^\|\s*(\d{1,2})\s*\|", line)
            if m and under_gate_heading:
                rows.append(int(m.group(1)))
        gate_rows = [r for r in rows if 1 <= r <= 40]
        if gate_rows and "gates" in facts and len(set(gate_rows)) >= 10:
            if max(gate_rows) != facts["gates"]:
                bad.append(f"{doc.name}: its gate table's highest number is "
                           f"{max(gate_rows)}, package has {facts['gates']}")
            if len(set(gate_rows)) != facts["gates"]:
                bad.append(f"{doc.name}: its gate table has "
                           f"{len(set(gate_rows))} rows, package has "
                           f"{facts['gates']} gates")
        for w, n in (("five", 5), ("six", 6), ("seven", 7)):
            if f"{w}-state" in text and n != _N_STATES \
                    and not _exempt(f"{w}-state"):
                bad.append(f"{doc.name}: says '{w}-state', the status table "
                           f"defines {_N_STATES}")
    return {"facts": facts, "contradictions": bad,
            "declared_exemptions_absorbed": absorbed}


def build(out_zip: Path, candidate: bool = False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="review_pkg_"))
    pkg = tmp / "package"
    rec = assemble(pkg)
    # There is no git inside a ZIP, so the build records what it built from and
    # gate 1 reads that instead. Recorded, not asserted: if the tree was dirty
    # the record says so, and gate 1 goes red on the package exactly as it
    # would have on the tree.
    head = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(CODE_ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    # ---- the fair pair's provenance, written for a reader --------------- #
    # The seventh review could not check the manuscript's +0.159 because the
    # package carried only the summary value inside t_final.json. This states
    # which artifact each printed number comes from, with digests, and names
    # the script that re-derives them.
    import csv as _csv
    cells_p = CODE_ROOT / "results/T6_symmetry/fair_same_host_recovery_cells.csv"
    sum_p = CODE_ROOT / "results/T6_symmetry/fair_same_host_recovery.json"
    tf_p = CODE_ROOT / "results/T5_stats/t_final.json"
    fair = json.loads(sum_p.read_text())
    tfr = json.loads(tf_p.read_text())["recovery"]
    with cells_p.open() as fh:
        rows = list(_csv.DictReader(fh))
    sym = fair["probe_vs_D_same_host_no_oracle"]
    orc = fair["probe_vs_D_same_host_oracle_control"]

    def _s(p_):
        return hashlib.sha256(p_.read_bytes()).hexdigest()

    means = fair["means"]
    (pkg / "evidence" / "FAIR_PAIR_PROVENANCE.md").write_text(
        "# The fair same-host recovery pair: where every printed number "
        "comes from\n\n"
        "One training run per cell produced, from the SAME trained model, the "
        "attention matrix `D`, the probe's ablation with the error signal "
        "taken from that host's own completed table, and the same ablation "
        "with the signal taken from the values withheld from the imputer. "
        "The pair is formed within the host, so the host draw cancels inside "
        "it.\n\n"
        "## Artifacts in this package\n\n"
        "| file | rows | sha256 |\n|---|---|---|\n"
        f"| `evidence/fair_same_host_recovery_cells.csv` | {len(rows)} "
        f"(15 regime x seed cells x {len({r['method'] for r in rows})} "
        f"objects) | `{_s(cells_p)}` |\n"
        f"| `evidence/fair_same_host_recovery.json` | summary | "
        f"`{_s(sum_p)}` |\n"
        f"| `gate_inputs/code_SNI/results/T5_stats/t_final.json` | the fact "
        f"store | `{_s(tf_p)}` |\n\n"
        "The same two files also appear under "
        "`gate_inputs/code_SNI/results/T6_symmetry/`, byte-identical, because "
        "that is the path the inventory's RECOMPUTED rows name; gate 16 "
        "checks that every duplicate name in this package is the same file.\n\n"
        "## Recomputing it\n\n"
        "    PYTHONHASHSEED=2025 python code/experiments/recompute_fair_pair.py\n\n"
        "It reads the cell CSV only, forms the within-host paired differences, "
        "and re-derives T, the seed-only bootstrap CI, the exact sign "
        "enumeration and the cell counts. It mirrors the archived bootstrap "
        "exactly (seed 20260831, 10,000 draws of the seed-level medians), so "
        "its interval is the printed one.\n\n"
        "## Which number appears where\n\n"
        "| printed as | value | from |\n|---|---|---|\n"
        f"| main text SS4.2, Table 4 note, Fig. 3 recovery cell: T | "
        f"{sym['T_mean_of_seed_medians']:+.3f} | cells -> mean of seed "
        f"medians; `t_final.json#recovery.probe_vs_D_same_host_symmetric.T` |\n"
        f"| the same, 95% CI | [{sym['ci95_T_seedboot'][0]:+.3f}, "
        f"{sym['ci95_T_seedboot'][1]:+.3f}] | seed-only bootstrap over the "
        f"same cells |\n"
        f"| the same, exact p | "
        f"{sym['exact_sign_enumeration']['p_two_sided']:.4f} | sign "
        f"enumeration over 5 seed blocks; attainable floor "
        f"{sym['exact_sign_enumeration']['floor_two_sided']:.4f} |\n"
        f"| cells favoring the probe | "
        f"{sym['cells_favouring_probe']}/{sym['cells_total']} | cells |\n"
        f"| withheld-truth control T | "
        f"{orc['T_mean_of_seed_medians']:+.3f} | same cells, oracle column; "
        f"`...probe_vs_D_same_host_oracle_control.T` |\n"
        f"| the difference between them | "
        f"{orc['T_mean_of_seed_medians'] - sym['T_mean_of_seed_medians']:+.3f}"
        f" | `oracle_contribution_T` |\n"
        f"| Table 4 main rows, per regime | see the cell file | "
        f"per-regime means of each object's AUROC |\n\n"
        "Object means over the fifteen cells, as the table prints them:\n\n"
        + "".join(f"- `{k}` {v:.4f}\n" for k, v in sorted(means.items()))
        + "\nThe retrained run reproduced the **pilot \\(D\\) archive** readout "
        "and the **T61 oracle/no-oracle recomputation archive** permutation "
        "readouts bitwise on all 15 cells: "
        "max |D_fresh - D_pilot| = "
        f"{fair['host_reproduction']['max_abs_D_fair_minus_pilot_D']}, "
        f"max |probe_fresh - probe_T61| = "
        f"{fair['host_reproduction']['max_abs_probe_fair_minus_T61_recompute']}"
        ". This does **not** assert identity with the separate **T4F archived "
        "withheld-truth row** printed in Table 4, and because archived "
        "model-state digests were unavailable it does not assert bitwise "
        "identity of the full trained host either. It establishes "
        "readout-level reproduction for the quantities used here "
        "(`host_reproduction.relevant_readouts_bitwise_match_archived`).\n\n"
        "The three archives this package refers to, by their only names:\n\n"
        "- **pilot D archive** --- `results/T4F_pilot/D_{regime}_s{seed}_"
        "SNI-D.csv`, the pilot run's attention matrices;\n"
        "- **T4F archived withheld-truth row** --- the "
        "`Permutation-on-SNI (archived; withheld-truth signal)` row of "
        "Table 4, from `results/T4_perm_on_sni/`;\n"
        "- **T61 oracle/no-oracle recomputation archive** --- "
        "`results/T6_symmetry/PERM_{oracle,noOracle}_*.csv`, the "
        "information-symmetry recomputation.\n\n"
        "They are different objects with different numbers. In the "
        "readout-reproduction claim, every archive reference is resolved to "
        "one of the three named artifacts; the ambiguous phrases "
        + " and ".join(f'"{b}"' for b in _banned_archive_phrases()) +
        " are prohibited. "
        "Ordinary uses of \"archived\" elsewhere in the package -- an "
        "archived band, an archived model, the archived row label -- are not "
        "restricted (tenth review P1-2).\n")

    # Extraction does not restore mtimes, so a package cannot be asked which
    # of its files is newer -- and gate 12's whole question is exactly that
    # ("is every figure newer than the fact store it draws from?"). Record
    # the answer here, where the filesystem still knows it.
    fresh = {}
    #: Every artifact and source named by a `fresher_than` check, so the
    #: status table can answer freshness inside a package too. Extraction
    #: discards mtimes, so a package-mode rebuild of the table used to report
    #: "Fig_leakage.pdf predates t42_summary.json" purely because both files
    #: came out of a ZIP with the same timestamp (twelfth review P0-5).
    #: The list is DERIVED, not typed: whatever the declarations name in a
    #: `fresher_than` check is what has to be recorded. A hand-kept list went
    #: stale twice in one sitting -- first missing the summary JSONs, then the
    #: generators -- because each fix only added the files that had just
    #: failed.
    def _freshness_paths() -> list:
        out, decl = set(), CODE_ROOT / "docs" / "ir_status_declarations.json"
        def walk(chk):
            if not isinstance(chk, dict):
                return
            if chk.get("type") == "fresher_than":
                if isinstance(chk.get("artifact"), str):
                    out.add(chk["artifact"])
                out.update(x for x in chk.get("sources", [])
                           if isinstance(x, str))
            for sub in chk.get("checks", []) or []:
                walk(sub)
        if decl.exists():
            items = json.loads(decl.read_text())
            for it in (items["items"] if isinstance(items, dict) else items):
                walk(it.get("check", {}))
        return sorted(out)

    #: ...unioned with what gate 12 reads directly. Deriving alone dropped
    #: Fig_vera.pdf, which no `fresher_than` check names but gate 12 does, and
    #: a figure with no recorded build time is a red gate. Two consumers, one
    #: record, and neither may silently shrink it.
    GATE12 = ("reporting/out/Fig_vera.pdf", "reporting/out/Fig_leakage.pdf",
              "reporting/out/Fig_scoreboard.pdf",
              "results/T5_stats/t_final.json")
    for rel in sorted(set(_freshness_paths()) | set(GATE12)):
        f = ROOT / rel
        if not f.exists():
            f = CODE_ROOT / rel
        if f.exists():
            fresh[rel.split("code_SNI/", 1)[-1]] = f.stat().st_mtime
    # Eighth review P0-1. The package date used to be whatever `date` said on
    # the machine running the gates, so the same ZIP was all-green in JST and
    # RED in UTC -- and "nothing outside the package is read" was false. It is
    # stamped here, once, at build time, and gate 10 reads only this.
    pdf_digests = {}
    for name in ("paperY_main.pdf", "paperY_ESM.pdf"):
        f = pkg / name
        if f.exists():
            b = f.read_bytes()
            pdf_digests[name] = {"md5": hashlib.md5(b).hexdigest(),
                                 "sha256": hashlib.sha256(b).hexdigest(),
                                 "bytes": len(b)}
    import time as _time
    build_date = _time.strftime("%Y-%m-%d")
    build_tz = _time.strftime("%Z") or "UTC"
    (pkg / "BUILD.json").write_text(json.dumps(
        {"head_commit": head, "working_tree_clean": not dirty,
         "built_by": "experiments/build_review_package.py",
         #: A candidate exists only so the gates have something to run
         #: against; its GATES.txt is necessarily the previous one. Gate 19
         #: reads this flag and defers, and a candidate is never delivered.
         "candidate": bool(candidate),
         "package_date": build_date,
         "package_timezone": build_tz,
         "source_mtimes": fresh,
         #: The delivered PDFs' own digests. RESPONSE_TO_IR11 pointed a reader
         #: at BUILD.json for them and BUILD.json did not carry them
         #: (thirteenth review 5.1) -- a pointer to a field that does not
         #: exist is worse than no pointer, because it reads as checked.
         "pdf_digests": pdf_digests,
         "note": "gate 1 reads head_commit from here in --from-package mode; "
                 "there is no git inside a package. gate 12 reads "
                 "source_mtimes, because unzipping discards them and would "
                 "otherwise make a freshness check answer at random. gate 10 "
                 "reads package_date, and in package mode reads NOTHING else "
                 "-- the system clock and the reader's timezone must not be "
                 "able to change a delivered package's verdict."},
        indent=1))
    rec["head_commit"] = head
    rec["working_tree_clean"] = not dirty
    rec["counts"] = audit_counts(pkg)
    #: Fail at the BUILD, not two reviews later. The transcript that ships as
    #: GATES.txt must have been produced from this commit; a copied-forward one
    #: is how a superseded gate script's output reached a delivered package.
    ts_head = rec["counts"]["facts"].pop("_transcript_head", "")
    if candidate:
        # The transcript cannot exist before the ZIP it describes. A CANDIDATE
        # build is the bootstrap: assemble it, run the gates against it, write
        # the path-sanitised transcript to GATES.txt, then build the final
        # package -- which does enforce the check below. A candidate is never
        # delivered, and it says so in BUILD.json.
        rec["candidate"] = True
    elif not ts_head or not head.startswith(ts_head):
        raise SystemExit(
            f"[STALE TRANSCRIPT] GATES.txt was made at "
            f"{ts_head or '(no HEAD line)'}, this build is {head[:8]}.\n"
            f"  Run the gates on a candidate ZIP built from this commit and\n"
            f"  write the path-sanitised transcript to\n"
            f"  internal_review/ir_staging/GATES.txt, then build again.")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    n_entries = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(pkg))
                n_entries += 1
    rec["zip"] = str(out_zip)
    # What SHIPS, counted as it is written -- not the length of the copy
    # list, which is one short because BUILD.json is written after it and
    # ships anyway. Two numbers named "files" that disagreed by one is how
    # a delivery note came to state a count no reader could reproduce.
    rec["n_files"] = n_entries
    rec["n_copied"] = len(rec["copied"])
    rec["staged_at"] = str(pkg)
    return rec


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    from package_layout import entries, roots
    e = entries()
    chk(len(e) > 30, f"the layout declares the package ({len(e)} entries)")
    chk(not [s for s, _ in e if not s.exists()],
        "every declared source exists in the working tree")
    rels = [r for _, r in e]
    chk(any(r.startswith("evidence") for r in rels), "evidence/ is included")
    chk(any(r.startswith("source/") for r in rels), "source/ is included")
    from package_layout import lead_diff_name
    chk(sum(r == lead_diff_name() or r.endswith("_diff_vs_R0.pdf")
            for r in rels) == 2,
        f"both diff PDFs are included (lead: {lead_diff_name()})")
    chk(any("t_final.json" in r for r in rels), "the single evidence source is in")
    r = roots(Path("/tmp/x"))
    chk(set(r) == {"ROOT", "CODE_ROOT", "PAPER", "STAGING", "OUTDIR"},
        "the package-to-repository mapping names every root the gates use")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/review_package.zip")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--candidate", action="store_true",
                    help="bootstrap build: skip the GATES.txt "
                         "freshness check, because the transcript "
                         "of this ZIP cannot exist yet. Never "
                         "deliver a candidate.")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = build(ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out),
              candidate=a.candidate)
    print(f"files {r['n_files']}   -> {r['zip']}")
    if r["missing_sources"]:
        print(f"MISSING SOURCES: {r['missing_sources']}")
    c = r["counts"]
    print(f"package facts: {c['facts']}")
    if c["contradictions"]:
        print("COUNT CONTRADICTIONS:")
        for x in c["contradictions"]:
            print("   ", x)
    return 1 if (r["missing_sources"] or c["contradictions"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
