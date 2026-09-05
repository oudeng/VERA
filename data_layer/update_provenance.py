"""T2.1: regenerate the summary table in docs/provenance.md from what was built.

Only the table between the two markers is replaced; the prose sections around it
are hand-written and stay. The numbers come from the derived tables themselves,
because a provenance document that disagrees with the data it describes is worse
than none.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "provenance.md"
BEGIN = "<!-- BEGIN GENERATED SUMMARY -->"
END = "<!-- END GENERATED SUMMARY -->"

#: Hand-written per dataset: everything that is a fact about the *source*, not
#: about the file we produced.
SOURCES = {
    "MIMIC": ("MIMIC-IV (PhysioNet, credentialed)",
              "Yes, `data_layer/build_mimic.py`",
              "No — upstream CSV only, no database access",
              "No (DUA)"),
    "eICU": ("eICU-CRD (PhysioNet, credentialed)",
             "Yes, `data_layer/build_eicu_cdc.py`",
             "No — needs PostgreSQL for the upstream extract",
             "No (DUA)"),
    "NHANES": ("NHANES 2017–2018 (CDC, public)",
               "Yes, `data_layer/build_nhanes.py`",
               "Yes — all 11 XPT modules are local",
               "Yes"),
    "CDC2022": ("CDC BRFSS 2022 heart-disease table (public domain)",
                "Yes, `data_layer/build_eicu_cdc.py --which cdc`",
                "Yes", "Yes"),
    "AutoMPG": ("UCI Auto MPG", "n/a (direct download)", "Yes", "Yes"),
    "ComCri": ("UCI Communities & Crime", "n/a", "Yes", "Yes"),
    "Concrete": ("UCI Concrete Compressive Strength", "n/a", "Yes", "Yes"),
}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())["datasets"]

    lines = [BEGIN, "",
             "| Dataset | n | d | Drivers held out | Source | Build script? | "
             "Re-runnable here? | Redistributable? | md5 |",
             "|---|---:|---:|---:|---|---|---|---|---|"]
    missing = []
    for name, (src, script, rerun, redist) in SOURCES.items():
        if name not in cfg:
            missing.append(name)
            continue
        b = cfg[name]
        p = Path(b["complete_path"])
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            missing.append(f"{name} ({p})")
            continue
        n_drv = len(b.get("always_observed", []) or [])
        lines.append(
            f"| {name} | {b['n_rows']} | {b['n_imputable']} | {n_drv} | {src} | "
            f"{script} | {rerun} | {redist} | `{md5(p)}` |")

    if missing:
        for m in missing:
            print(f"[MISSING] {m}")
        raise SystemExit(f"{len(missing)} dataset(s) not available; "
                         f"provenance would be incomplete")

    lines += ["",
              "`d` counts **imputable** columns: the table also carries an `ID`, a "
              "downstream target, and the always-observed MAR drivers, none of "
              "which are ever masked. \"Drivers held out\" is the evaluation "
              "coverage those drivers cost — see `configs/missingness.yaml`, "
              "profile `clinical_v1`.",
              "", END]
    block = "\n".join(lines)

    text = DOC.read_text()
    if BEGIN in text and END in text:
        head = text.split(BEGIN)[0]
        tail = text.split(END, 1)[1]
        new = head + block + tail
    else:
        # First run: insert under the "## Summary" heading, leaving the old
        # table visible below it rather than deleting prose we did not write.
        marker = "## Summary\n"
        i = text.index(marker) + len(marker)
        new = text[:i] + "\n" + block + "\n\n<!-- superseded table below -->\n" + text[i:]

    print(block)
    if a.dry_run:
        print("\n[dry-run] provenance.md not written")
        return 0
    DOC.write_text(new)
    print(f"\nwrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
