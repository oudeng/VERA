"""Primary cost table (P5R-H SS4; third review SS7.1/SS7.4).

The single-thread idle-machine measurement is the paper's primary cost
benchmark, so it is the main-text table. One process per audit object,
one BLAS thread, idle-machine guard enforced by the probe itself; the
grid-concurrency record is retained as an operational log in the Online
Resource.

Every ratio is computed here, never typed, and each dataset's
denominator is NAMED in the table: the most expensive non-SNI-hosted
comparator on that dataset, selected by the measurement itself. Three
quantities per object, as the review asks: raw seconds, the ratio
against TAP, and the ratio against that named denominator.

    env PYTHONHASHSEED=2025 python reporting/table_cost_primary.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

A3 = CODE_ROOT / "results" / "A3_cost_context"
OUT = CODE_ROOT / "reporting" / "out" / "tab_cost_primary.tex"
DATASETS = ["MIMIC", "eICU"]
#: probe tag -> (display name, is_sni_hosted)
OBJECTS = [("SNI", r"SNI \Dm{}", True),
           ("PermSNI", r"Permutation-on-SNI", True),
           ("P", r"\TAP{}", False),
           ("MFimp", "MF importance", False),
           ("MFshap", "SHAP-on-MF", False),
           ("MFperm", "Permutation-on-MF", False)]


def _ratio(x: float) -> str:
    """Ratios span five orders of magnitude here; never print a
    significant value as 0.0."""
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:.1f}"
    if x >= 0.01:
        return f"{x:.2f}"
    return f"{x:.1g}"


def _load(a3: Path, ds: str, tag: str):
    """(total_seconds, peak_rss_gb) for one probe, or None if not probed."""
    j = a3 / f"{ds}_{tag}.json"
    t = a3 / f"{ds}_{tag}_time.txt"
    if not j.exists():
        return None
    rec = json.loads(j.read_text())
    rss = proc = None
    if t.exists():
        txt = t.read_text()
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        if m:
            rss = int(m.group(1)) / 1e6
        w = re.search(r"Elapsed \(wall clock\).*?(?:(\d+):)?(\d+):([\d.]+)",
                      txt)
        if w:
            h = int(w.group(1) or 0)
            proc = h * 3600 + int(w.group(2)) * 60 + float(w.group(3))
    # P5R-K SS5 (fourth review SS7): two different clocks, named separately.
    # "sec" is the instrumented algorithm segment the probe times itself;
    # "proc" is the whole process as GNU time saw it, which also carries
    # interpreter start-up, imports and I/O. For TAP the two differ by more
    # than an order of magnitude, so calling both "total" was wrong.
    return {"sec": float(rec["wall_total_sec"]), "proc": proc, "rss": rss,
            "load_start": rec.get("load_at_start_1min"),
            "load_end": rec.get("load_at_end_1min")}


def gather(a3: Path = A3) -> dict:
    data = {}
    for ds in DATASETS:
        per = {}
        for tag, _disp, _sni in OBJECTS:
            r = _load(a3, ds, tag)
            if r is not None:
                per[tag] = r
        if "P" not in per:
            raise FileNotFoundError(f"{ds}: the TAP probe is the ratio "
                                    f"floor and is missing")
        non_sni = {t: v["sec"] for t, v in per.items()
                   if not dict((o[0], o[2]) for o in OBJECTS)[t]}
        denom_tag = max(non_sni, key=non_sni.get)
        data[ds] = {"per": per, "denom_tag": denom_tag,
                    "denom_sec": non_sni[denom_tag],
                    "denom_proc": per[denom_tag].get("proc")}
    return data


def build(out_path: Path = OUT, a3: Path = A3) -> Path:
    data = gather(a3)
    disp = dict((t, d) for t, d, _s in OBJECTS)
    rows, missing = [], []
    for ds in DATASETS:
        blk = data[ds]
        rows.append({"Audit object": rf"\multicolumn{{7}}{{l}}{{{{{ds}: "
                                     rf"denominator = "
                                     rf"{disp[blk['denom_tag']]}}}}}",
                     "_span": True})
        tap = blk["per"]["P"]["sec"]
        for tag, d, _sni in OBJECTS:
            if tag not in blk["per"]:
                missing.append(f"{ds}/{tag}")
                continue
            v = blk["per"][tag]
            def _sec(x):
                if x is None:
                    return "---"
                return f"{x:,.1f}" if x >= 1 else f"{x:.2f}"
            rows.append({
                "Audit object": d,
                "Algorithm (s)": _sec(v["sec"]),
                "Process (s)": _sec(v.get("proc")),
                r"$\times$ \TAP{}": _ratio(v["sec"] / tap),
                r"$\times$ denom.": _ratio(v["sec"] / blk["denom_sec"]),
                r"$\times$ denom. (proc.)":
                    (_ratio(v["proc"] / blk["denom_proc"])
                     if v.get("proc") and blk.get("denom_proc") else "---"),
                "Peak RSS (GB)": (f"{v['rss']:.2f}" if v["rss"] is not None
                                  else "---")})
    body = pd.DataFrame(rows).drop(columns=["_span"],
                                   errors="ignore").fillna("")
    ratios = ", ".join(
        rf"{ds} {data[ds]['per']['SNI']['sec'] / data[ds]['denom_sec']:.0f}"
        rf"$\times$ ({disp[data[ds]['denom_tag']]})"
        for ds in DATASETS if "SNI" in data[ds]["per"])
    loads = ", ".join(
        f"{ds} {min(v['load_start'] for v in data[ds]['per'].values() if v['load_start'] is not None):.2f}"
        f"--{max(v['load_end'] for v in data[ds]['per'].values() if v['load_end'] is not None):.2f}"
        for ds in DATASETS
        if any(v["load_start"] is not None for v in data[ds]["per"].values()))
    # the headline ratio must hold under BOTH clocks or the caliber choice
    # would be doing the work; state each explicitly.
    st_proc = {ds: (data[ds]["per"]["SNI"].get("proc"),
                    data[ds].get("denom_proc"))
               for ds in DATASETS if "SNI" in data[ds]["per"]}
    proc_ratios = ", ".join(
        f"{ds} {a / b:.0f}$\\times$"
        for ds, (a, b) in st_proc.items() if a and b)
    note = (
        r"\textit{Primary cost benchmark. Two clocks, named separately.} "
        r"\emph{Algorithm} is the instrumented segment the probe times "
        r"itself (fit and readout); \emph{Process} is the whole process as "
        r"GNU time saw it, which additionally carries interpreter start-up, "
        r"imports and I/O. The two differ by more than an order of "
        r"magnitude for \TAP{}, whose computation is sub-second, so they "
        r"are never added or compared across rows. The headline ratio is "
        r"stable under either: " + ratios + r" on the algorithm clock and "
        + (proc_ratios or "---") + r" on the process clock. "
        r"One audit object per process, "
        r"one BLAS thread, idle-machine guard enforced by the probe (it "
        r"refuses to start above a 1-minute load average of 2.0); "
        r"MAR@30\%, seed 1, the faithfulness condition. Ratios are "
        r"computed from the seconds in this table, never typed. Each "
        r"dataset's denominator is the most expensive comparator not "
        r"hosted by SNI, selected by the measurement itself and named in "
        r"the block header. "
        r"Peak resident set size is process-local (kbytes as recorded by "
        r"GNU time, divided by $10^6$); each MissForest readout is probed "
        r"in its own process and therefore carries the shared forest fit, "
        r"which is the honest accounting because no readout exists "
        r"without it. Neither absolute times nor ratios are assumed to "
        r"transfer across systems: they are records of this "
        r"implementation on one machine with heterogeneous "
        r"performance/efficiency cores. The grid-concurrency record of "
        r"the campaign as executed is an operational log in Online "
        r"Resource~1.")
    if loads:
        note += rf" One-minute load averages observed across these probes: {loads}."
    if missing:
        note += (r" Objects without a single-thread probe in this run: "
                 + ", ".join(m.replace("_", r"\_") for m in missing)
                 + r" (their grid-concurrency figures are in the "
                   r"operational log).")
    tex = dataframe_to_tex(
        body,
        caption=(r"Cost of each audit object under the single-thread "
                 r"idle-machine benchmark: two clocks named separately "
                 r"(instrumented algorithm segment and end-to-end process "
                 r"wall clock), ratio against "
                 r"\TAP{}, ratio against the named most expensive "
                 r"non-SNI-hosted comparator, and peak memory."),
        label="tab:cost_primary", column_format="lrrrrrr",
        header=["Audit object", "Algorithm (s)", "Process (s)",
                r"$\times$ \TAP{}", r"$\times$ denom.",
                r"$\times$ denom. (proc.)", "Peak RSS (GB)"],
        style=TableStyle(environment="table*", size=r"\scriptsize",
                         notes=(note,)),
        escape_data=False)
    # A block-header row spans the table; the empty tail cells the frame
    # carries would be extra columns to LaTeX. Strip them.
    tex = re.sub(r"(\\multicolumn\{\d+\}\{l\}\{.*?\}\})(?:\s*&\s*)+(\\\\)",
                 r"\1 \2", tex)
    # seven columns need tighter separation than the default in table*
    tex = tex.replace(r"\scriptsize",
                      r"\scriptsize\setlength{\tabcolsep}{3pt}", 1)
    # The prose macros come from THIS computation, so the abstract, the
    # contribution list and this table cannot drift apart (third review
    # SS7.1: the main table and the stated primary benchmark must agree).
    st = {ds: data[ds]["per"]["SNI"]["sec"] / data[ds]["denom_sec"]
          for ds in DATASETS if "SNI" in data[ds]["per"]}
    rss = [v["rss"] for ds in DATASETS for v in data[ds]["per"].values()
           if v["rss"] is not None]
    lo, hi = min(st.values()), max(st.values())
    macros = (
        "% Single-thread standardized macros -- emitted by "
        "reporting/table_cost_primary.py from the SAME probe records as "
        "the primary cost table (idle machine, 1 BLAS thread, one audit "
        "object per process; peak RSS via /usr/bin/time -v).\n"
        + "".join(f"% {ds}: {v:.1f}x vs {disp[data[ds]['denom_tag']]}\n"
                  for ds, v in st.items())
        + f"\\newcommand{{\\costRatioSTRange}}"
          f"{{{round(lo)}--{round(hi)}$\\times$}}\n"
        f"\\newcommand{{\\costPeakRssRange}}"
        f"{{{min(rss):.2f}--{max(rss):.2f}}}\n"
        "\\newcommand{\\costDReadoutSecs}{$<0.01$}\n")
    (out_path.parent / "a3_macros.tex").write_text(macros)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_cost_primary.py",
        "input": str(a3) + " (per-object single-thread probes)",
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    import tempfile
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    with tempfile.TemporaryDirectory() as td:
        a3 = Path(td)
        secs = {"P": 0.5, "MFimp": 10.0, "MFshap": 100.0, "MFperm": 60.0,
                "SNI": 3000.0, "PermSNI": 3100.0}
        for ds in DATASETS:
            for tag, sec in secs.items():
                (a3 / f"{ds}_{tag}.json").write_text(json.dumps(
                    {"wall_total_sec": sec, "load_at_start_1min": 0.1,
                     "load_at_end_1min": 0.2}))
                (a3 / f"{ds}_{tag}_time.txt").write_text(
                    "\tMaximum resident set size (kbytes): 1000000\n")
        g = gather(a3)
        check(g["MIMIC"]["denom_tag"] == "MFshap",
              "denominator = most expensive non-SNI comparator, computed")
        p = build(a3 / "t.tex", a3)
        t = p.read_text()
        check("SHAP-on-MF" in t and "denominator =" in t,
              "denominator named in the table body")
        check("30.0" in t, "ratio vs denominator computed (3000/100)")
        check("6,000" in t or "6000" in t, "ratio vs TAP computed (3000/0.5)")
        check("transfer across systems" in " ".join(t.split()),
              "transportability explicitly denied")
        check("transportable" not in t, "no transportability claim")
        import re as _re
        bad = [ln for ln in t.splitlines()
               if "multicolumn" in ln and _re.search(r"\}\}\s*&", ln)]
        check(not bad, "block-header rows carry no extra columns")
        # a missing probe must be disclosed, not silently dropped
        (a3 / "MIMIC_MFperm.json").unlink()
        p2 = build(a3 / "t2.tex", a3)
        check("without a single-thread probe" in " ".join(p2.read_text().split()),
              "unprobed object disclosed in the note")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(_selftest())
    print(f"[OK] wrote {build()}")
    raise SystemExit(0)
