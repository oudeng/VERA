"""ESM table: per-object peak memory (second internal review SS13).

Single source: the GNU-time records of the A3 single-thread cost probes
(results/A3_cost_context/{ds}_{obj}_time.txt, `Maximum resident set
size`). One row per (dataset, audit-object) probe.

    env PYTHONHASHSEED=2025 python reporting/esm_rss.py [--selftest]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

SRC = CODE_ROOT / "results" / "A3_cost_context"
OUT = CODE_ROOT / "reporting" / "out" / "tab_esm_rss.tex"
#: P5R-H SS3 / third review SS7.3: one probe process per audit object, so
#: this is a per-object table, not a host-family envelope. The legacy
#: bundled "MF" probe is excluded here; it survives in the archive.
DISPLAY = {"SNI": r"SNI \Dm{} (host + readout)",
           "PermSNI": r"Permutation-on-SNI (host + readout)",
           "P": r"\TAP{}",
           "MFimp": "MF importance (fit + readout)",
           "MFshap": "SHAP-on-MF (fit + readout)",
           "MFperm": "Permutation-on-MF (fit + readout)"}


def _parse(path: Path) -> dict:
    t = path.read_text()
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1))
    wall = re.search(r"Elapsed \(wall clock\).*?(\d+:[\d:.]+)", t).group(1)
    # kbytes / 1e6: the convention of the main-text macro
    # (\costPeakRssRange, reporting/table_cost.py _a3_ratios).
    return {"rss_gb": rss / 1e6, "wall": wall}


def build(src: Path = SRC, out_path: Path = OUT) -> Path:
    rows = []
    for f in sorted(src.glob("*_time.txt")):
        ds, obj = f.stem.replace("_time", "").split("_", 1)
        if obj not in DISPLAY:
            continue
        if "REFUSING TO RUN" in f.read_text():
            continue          # the idle guard refused; no measurement exists
        r = _parse(f)
        rows.append({"Dataset": ds, "Object": DISPLAY.get(obj, obj),
                     "Peak RSS (GB)": f"{r['rss_gb']:.2f}",
                     "Process wall clock": r["wall"]})
    if not rows:
        raise FileNotFoundError(f"no usable *_time.txt under {src}")
    body = pd.DataFrame(rows).sort_values(["Dataset", "Object"]).reset_index(drop=True)
    style = TableStyle(environment="table", notes=(
        r"Peak resident set size (GNU time, `Maximum resident set size') of "
        r"the single-thread idle-machine cost probes (kbytes as "
        r"recorded by GNU time, divided by $10^6$ -- the same convention "
        r"as the range given in the main text), one full "
        r"artifact construction per row on the stated dataset. Memory is "
        r"reported per audit object: each MissForest readout is measured "
        r"in its own process and therefore carries the shared forest fit, "
        r"which is the honest accounting because no readout exists "
        r"without it. Peak resident set size is process-local and "
        r"unaffected by other processes; the probe refuses to start above "
        r"a 1-minute load average of 2.0, and a refused probe writes no "
        r"record, so no number here was taken under contention. These are "
        r"the same runs that anchor the primary cost ratios. The clock "
        r"here is the whole process as GNU time saw it, including "
        r"interpreter start-up, imports and I/O; the main-text cost table "
        r"reports that clock beside the instrumented algorithm segment, "
        r"which is the shorter of the two and differs most for \TAP{}.",))
    tex = dataframe_to_tex(
        body, caption=(r"Per-object peak memory of the single-thread cost "
                       r"probes."),
        label="tab:esm_rss", column_format="llcc",
        header=["Dataset", "Object", "Peak RSS (GB)",
                "Process wall clock"],
        style=style, escape_data=False)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/esm_rss.py",
        "input": str(src) + " (*_time.txt)",
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    import tempfile
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "X_SNI_time.txt").write_text(
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.03\n"
            "\tMaximum resident set size (kbytes): 1048576\n")
        p = build(d, d / "o.tex")
        t = p.read_text()
        check("1.05" in t, "kbytes -> GB conversion (macro convention)")
        check("1:02.03" in t, "wall clock carried")
        try:
            build(d / "none", d / "o2.tex")
            check(False, "missing dir refused")
        except FileNotFoundError:
            check(True, "missing dir refused")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(_selftest())
    print(f"[OK] wrote {build()}")
    raise SystemExit(0)
