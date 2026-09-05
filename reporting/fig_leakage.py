"""Leakage-axis contrast figure (P5 SS4.1b; P5R-H SS1.4 rework):
detection counts on the two discriminating classes -- interaction
proxies and the zero-increment discrepancy control -- for all six
objects, each count read under the dual-estimand convention of the
Methods. Observed null rates for BOTH highlighted objects are printed
beside their bars; no significance-style number appears anywhere in
the figure. Source: t42_summary.json; every bar height is read, never
typed. Historical class keys are accessed via reporting/termmap.py.

    PYTHONHASHSEED=2025 python reporting/fig_leakage.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.termmap import condition_order                   # noqa: E402

T42 = CODE_ROOT / "results" / "T4_leakage" / "t42_summary.json"
OUT = CODE_ROOT / "reporting" / "out" / "Fig_leakage.pdf"
OBJECTS = ["SNI-D", "P", "MissForest-importance", "SHAP-on-MissForest",
           "Permutation-on-MissForest", "Permutation-on-SNI"]
LABELS = ["SNI D", "TAP", "MF imp.", "SHAP-MF", "Perm-MF", "Perm-SNI"]


def build(out_path: Path = OUT, summary_path: Path = T42) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(summary_path.read_text())
    cnt = {(r["object"], r["condition"]): r for r in d["counts"]
           if r["kind"] == "inj"}
    dc_key = condition_order()[-1]
    inter = [cnt[(o, "interaction")]["detected"] for o in OBJECTS]
    dctrl = [cnt[(o, dc_key)]["detected"] for o in OBJECTS]
    n = cnt[(OBJECTS[0], "interaction")]["n"]
    # Wilson 95% intervals from the summary's own fields (review SS7-14:
    # n = 6 is small; the bars must carry their uncertainty), plus the
    # probe's above-nominal null FPR printed beside its bars.
    iv = {c: {o: (cnt[(o, c)]["wilson_lo"] * n, cnt[(o, c)]["wilson_hi"] * n)
              for o in OBJECTS} for c in ("interaction", dc_key)}
    nulls = d.get("null_exact_binomial", {})

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, vals, cond, title in (
            (axes[0], inter, "interaction",
             f"Interaction proxies (detected / {n})"),
            (axes[1], dctrl, dc_key,
             f"Discrepancy control (flags / {n})")):
        # highlight: SNI-D dark red; Perm-on-SNI dark blue; rest gray
        colors = ["#b2182b" if o == "SNI-D" else
                  "#2166ac" if o == "Permutation-on-SNI" else "#9e9e9e"
                  for o in OBJECTS]
        ax.bar(range(len(OBJECTS)), vals, color=colors)
        errs = np.array([[v - iv[cond][o][0], iv[cond][o][1] - v]
                         for o, v in zip(OBJECTS, vals)]).T
        ax.errorbar(range(len(OBJECTS)), vals, yerr=errs, fmt="none",
                    ecolor="#37474f", elinewidth=1.0, capsize=2.5)
        ax.set_xticks(range(len(OBJECTS)))
        ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=7)
        # The n and the interval convention live in the caption; repeating
        # them in both panel titles made the two titles collide.
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, n + 0.6)
        ax.set_yticks(range(0, n + 1, 2))
        for i, (o, v) in enumerate(zip(OBJECTS, vals)):
            ax.text(i, iv[cond][o][1] + 0.12, str(v), ha="center",
                    fontsize=7)
        for obj, col in (("SNI-D", "#b2182b"),
                         ("Permutation-on-SNI", "#2166ac")):
            nb = nulls.get(obj)
            if nb:
                j = OBJECTS.index(obj)
                ax.text(j, -1.95,
                        f"null {nb['detected']}/{nb['n']}",
                        ha="center", fontsize=6, color=col,
                        clip_on=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("count", fontsize=8)
    fig.suptitle("", fontsize=1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight",
                metadata={"CreationDate": None,
                          "Subject": f"generator: reporting/fig_leakage.py; "
                                     f"input: {summary_path}; commit: "
                                     f"{runconfig.git_commit()}"})
    plt.close(fig)
    return out_path


def _selftest() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    counts = []
    for o in OBJECTS:
        for c in ("interaction", condition_order()[-1]):
            det = 0 if (o, c) == ("SNI-D", "interaction") else 6
            counts.append({"object": o, "condition": c, "kind": "inj",
                           "detected": det, "n": 6,
                           "wilson_lo": max(0.0, det / 6 - 0.15),
                           "wilson_hi": min(1.0, det / 6 + 0.15)})
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "s.json"
        sp.write_text(json.dumps({"counts": counts}))
        out = build(Path(td) / "f.pdf", sp)
        check(out.exists() and out.stat().st_size > 5000,
              "figure written, non-trivial size")
        # RENDERED geometry: panel titles must not collide. A text gate
        # cannot see two titles overstriking each other, which is exactly
        # how the reworked (longer) titles first shipped.
        import subprocess as _sp
        lay = _sp.run(["pdftotext", "-layout", str(out), "-"],
                      capture_output=True, text=True).stdout
        head = [ln for ln in lay.splitlines() if "Interaction proxies" in ln]
        check(bool(head), "panel titles present in the rendered layout")
        if head:
            check("Discrepancy" in head[0]
                  and head[0].index("Discrepancy")
                  > head[0].index("Interaction proxies") + 30,
                  "the two panel titles sit apart, not overstruck")
            check("Wilson" not in head[0],
                  "interval convention is in the caption, not duplicated "
                  "into both titles")
        # missing cell must raise
        sp.write_text(json.dumps({"counts": counts[1:]}))
        try:
            build(Path(td) / "f2.pdf", sp)
            check(False, "missing cell must raise")
        except KeyError:
            check(True, "missing cell raises")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build()
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
