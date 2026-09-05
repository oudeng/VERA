"""External-anchor package builder (P5R-C-A SS A2): assembles the
prospective-specification evidence -- every rule document, the corrigenda
and addenda, and the zero-artifact attestation texts from the commit
bodies -- into one archive with a manifest and SHA-256 checksums, ready
for the first author to deposit (GitHub public push, Software Heritage
trigger, or OSF/Zenodo DOI). Read-only with respect to the repository.

    env PYTHONHASHSEED=2025 python experiments/make_prereg_archive.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = CODE_ROOT.parent / "reports" / "prereg_archive"

RULE_DOCS = [
    "docs/T25_pilot_decision_rule.md",
    "docs/T2d1_decision_rule.md",
    "docs/T32_faithfulness_decision_rule.md",
    "docs/T32_R1_redundancy_precheck.md",
    "docs/T42_leakage_rules.md",
    "docs/T42_confirmatory_replication_rules.md",
    "docs/T43_noprior_decision_rule.md",
    "docs/T43_interpretation_supplement.md",
    "docs/T44_downstream_rules.md",
    "docs/T45_statistics_rules.md",
    "docs/T4F_presentation_rule.md",
    "docs/T4F_score_verdict_rule.md",
    "docs/T51_statistical_analysis_rules.md",
    "docs/T52_probe_triangulation_rules.md",
    # P5R-I: the branch blueprints are a prospective-specification artifact
    # (pre-written wording branches, committed before the verdicts ran) and
    # the baseline-family rules were committed this round, before any family
    # artifact existed. Both are cited in the ESM chain, so both must be in
    # the bundle the ESM points at.
    "docs/D51_branch_blueprints.md",
    "docs/T53_tap_family_rules.md",
    # P5R-L/M (2026-08-29): three more, each committed with a zero-artifact
    # attestation BEFORE the measurements it governs, and each governing a
    # number the manuscript now reports. Leaving them out would make the
    # ESM's claim about the bundle false for exactly the analyses the
    # Discussion rests on.
    #   T53 input correction  -- the corrected TAP-family inputs (ESM 8.1)
    #   T60 lineage audit     -- the audit that found the defect
    #   T61 information symmetry -- the recomputes behind legs (i) and (iii)
    "docs/T53_input_correction_rules.md",
    "docs/T60_tap_lineage_audit_rules.md",
    "docs/T61_information_symmetry_rules.md",
]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(CODE_ROOT), *args],
                          capture_output=True, text=True).stdout


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    manifest: dict = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "head_commit": _git("rev-parse", "HEAD").strip(),
                      "files": [], "commit_evidence": []}
    present = []
    for rel in RULE_DOCS:
        p = CODE_ROOT / rel
        if not p.exists():
            manifest["files"].append({"path": rel, "status": "MISSING"})
            continue
        data = p.read_bytes()
        manifest["files"].append({
            "path": rel, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "last_commit": _git("log", "-1", "--format=%H %cI", "--",
                                rel).strip()})
        present.append(p)
    # full commit history of every rule doc, with bodies (the
    # zero-artifact attestations live in the bodies)
    log = _git("log", "--format=%H%n%cI%n%B%n==END==", "--",
               *[r for r in RULE_DOCS if (CODE_ROOT / r).exists()])
    manifest["commit_evidence"] = [b.strip() for b in log.split("==END==")
                                   if b.strip()]
    mpath = OUT_DIR / f"manifest_{stamp}.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    tpath = OUT_DIR / f"prereg_rules_{stamp}.tar.gz"
    with tarfile.open(tpath, "w:gz") as tf:
        for p in present:
            tf.add(p, arcname=f"docs/{p.name}")
        tf.add(mpath, arcname=mpath.name)
    sha = hashlib.sha256(tpath.read_bytes()).hexdigest()
    (OUT_DIR / f"prereg_rules_{stamp}.tar.gz.sha256").write_text(
        f"{sha}  {tpath.name}\n")
    n_att = sum(1 for c in manifest["commit_evidence"]
                if "attest" in c.lower() or "zero-artifact" in c.lower()
                or "zero artifact" in c.lower())
    print(f"[ok] {tpath.name}: {len(present)}/{len(RULE_DOCS)} rule docs, "
          f"{len(manifest['commit_evidence'])} commits in evidence "
          f"({n_att} with attestation language), sha256 {sha[:16]}...")
    missing = [f["path"] for f in manifest["files"]
               if f.get("status") == "MISSING"]
    if missing:
        print(f"[note] missing (verify names): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
