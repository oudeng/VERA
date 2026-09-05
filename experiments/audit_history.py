"""The retired rule's label, kept where nothing current can read it.

Tenth review P0-1 item 4. The T4F cell-level rule produced a verdict, the
rule was retired for pseudo-replication, and the label kept turning up in
places that made it look current -- renamed under a scary key in the
canonical fact store, recomputed as an "integrity check". Deleting it
outright would hide that a rule was followed and then found wanting, which is
itself a thing an audit trail should not do.

So it lives here: one file, marked HISTORICAL - NOT CANONICAL on its first
line, holding what the rule produced, when it was retired, why, the digests
of the frozen artifacts that carry it, and what replaced it. Nothing reads
this file. A package-level assertion (gate 18) allows the string to appear
only in files that carry the marker below.

    PYTHONHASHSEED=2025 python experiments/audit_history.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
T4F = CODE_ROOT / "results" / "T4_perm_on_sni"
OUT_DIR = ROOT / "internal_review" / "ir_staging" / "evidence"

#: Any file allowed to contain the retired label must carry this, verbatim,
#: near its top. Gate 18 looks for exactly this string.
MARKER = "HISTORICAL - NOT CANONICAL"

#: The label the retired rule produced. This module is the ONLY place
#: it is written as a literal; gate 18 and the generators' selftests
#: import it from here, so a search for it cannot be confused with a
#: value some generator might emit (tenth review P0-1).
RETIRED_LABEL = "SAME_HOST_POSTHOC_WINS"

FROZEN = ("t4f_verdict.json", "t4f_sixway_cells.csv")


def record() -> dict:
    files = {}
    for name in FROZEN:
        f = T4F / name
        files[name] = {
            "path": f"results/T4_perm_on_sni/{name}",
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()
            if f.exists() else None}
    label = (json.loads((T4F / "t4f_verdict.json").read_text())["verdict"]
             if (T4F / "t4f_verdict.json").exists() else None)
    return {
        "_marker": MARKER,
        "_doc": "The label a retired rule once produced. Nothing current "
                "reads this file; it exists so the retirement is visible "
                "rather than silent.",
        "rule_document": "docs/T4F_score_verdict_rule.md",
        "label_produced": label,
        "retired_on": "2026-08-30",
        "retired_because": "the rule's inference unit -- 15 regime-by-seed "
                           "cells treated as 15 independent pairs -- is not "
                           "independent: three regimes are nested inside each "
                           "seed. That is the pseudo-replication this "
                           "revision corrects everywhere else.",
        "arithmetic_status": "DELETED from the current generators "
                             "(tenth review P0-1). It is not applied to this "
                             "study's data, and not re-executed on the frozen "
                             "record either -- the frozen record's integrity "
                             "is checked by digest and schema.",
        "current_inferential_replacement":
            "recovery.probe_vs_D_same_host_symmetric, read under the "
            "seed-block exact enumeration: 5 seed blocks, two-sided exact "
            "p = 0.0625 at its attainable floor, classification INDET",
        "frozen_artifacts": files,
    }


def markdown(r: dict) -> str:
    files = "\n".join(f"- `{v['path']}` --- SHA-256 `{v['sha256']}`"
                      for v in r["frozen_artifacts"].values())
    return f"""# {MARKER}

**This file is audit history. It is not a current fact, and no generator,
figure, table or claim reads it.** It exists so that a rule which was
followed and then found wanting stays visible instead of disappearing.

| | |
|---|---|
| rule document | `{r['rule_document']}` |
| label it produced | `{r['label_produced']}` |
| retired on | {r['retired_on']} |
| arithmetic | {r['arithmetic_status']} |

**Why it was retired.** {r['retired_because']}

**What replaced it.** {r['current_inferential_replacement']}

**The frozen artifacts that carry the label**, by digest --- integrity is
checked against these, never by re-running the rule:

{files}
"""


def build() -> list:
    r = record()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    j = OUT_DIR / "AUDIT_HISTORY.json"
    m = OUT_DIR / "AUDIT_HISTORY.md"
    j.write_text(json.dumps(r, indent=1) + "\n")
    m.write_text(markdown(r))
    return [j, m]


def _selftest() -> int:
    ok = True

    def chk(c, msg):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + msg)
        ok = ok and bool(c)

    r = record()
    chk(r["_marker"] == MARKER, "the record carries the non-canonical marker")
    chk(markdown(r).splitlines()[0].endswith(MARKER),
        "the marker is on the markdown's first line, where a reader meets it")
    chk(all(v["sha256"] for v in r["frozen_artifacts"].values()),
        "every frozen artifact has a digest")
    chk("DELETED" in r["arithmetic_status"],
        "the record says the arithmetic is deleted, not merely unused")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    for f in build():
        print(f"[OK] wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
