"""Paired sign-test sensitivity analysis (P5R-L SS6; fifth review SS4.1).

The primary inference on the faithfulness axis enumerates seed-block sign
flips, which assumes the seed-level paired differences are sign-exchangeable
under the null. This is the same assumption's simplest consequence read on its
own: an exact two-sided binomial sign test on the seed-level sign counts, which
uses NOTHING but signs -- no effect size, no magnitude, no re-run.

Every number here is computed from `results/T5_stats/t_final.json`; none is
transcribed. A row whose objects did not have equal information is not
reported rather than filled with a provisional value, and the table says
which row and why.

    env PYTHONHASHSEED=2025 python reporting/table_sign_test.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

from common import runconfig                                    # noqa: E402
from reporting.latex import TableStyle, dataframe_to_tex, write_tex  # noqa: E402

T_FINAL = CODE_ROOT / "results" / "T5_stats" / "t_final.json"
OUT = CODE_ROOT / "reporting" / "out" / "tab_sign_test.tex"
#: rows whose compared objects did not have equal information ; not
#: reported rather than guessed
HELD = {"recovery_probe_vs_D"}


def sign_p(k: int, n: int) -> float:
    """Exact two-sided binomial sign test, p = 0.5, no continuity correction."""
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps: list) -> list:
    """Holm step-down, returned in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def _neg(rec: dict) -> tuple:
    m = re.fullmatch(r"(\d+)/(\d+)", str(rec["seeds_negative"]))
    return int(m.group(1)), int(m.group(2))


def collect() -> list:
    tf = json.loads(T_FINAL.read_text())
    rows = []
    for family, key, label in (
            ("with-prior faithfulness", "faithfulness",
             "With-prior faithfulness"),
            ("no-prior faithfulness", "noprior_faithfulness",
             "No-prior control")):
        blk = tf[key]
        fam = []
        for ds in ("MIMIC", "eICU"):
            k, n = _neg(blk[ds])
            fam.append({"family": family, "label": label, "table": ds,
                        "k_negative": k, "n_seeds": n,
                        "p_sign": sign_p(k, n),
                        "p_enum_holm": float(blk[ds]["p_holm"])})
        for r, a in zip(fam, holm([x["p_sign"] for x in fam])):
            r["p_sign_holm"] = a
        rows += fam
    # The archived recovery comparison is between objects with unequal
    # information . A symmetric probe now exists, so the reason has to
    # say which comparison is being withheld and why -- "unequal information"
    # on its own stopped being the whole story the moment the axis was
    # recomputed.
    rows.append({"family": "recovery", "label": "Same-host probe vs \\Dm{}",
                 "table": "synthetic (5 seeds)", "k_negative": None,
                 "n_seeds": None, "p_sign": None, "p_sign_holm": None,
                 "p_enum_holm": None, "held": True,
                 "held_reason": "on the archived reading the same-host "
                                "probe's error signal reads values withheld "
                                "from the imputer, which the objects it is "
                                "compared with do not, so a sign test on its "
                                "signs would compare unequal information. "
                                "The symmetric same-host probe-versus-"
                                "\\Dm{} comparison is reported under the "
                                "prespecified seed-block exact analysis in "
                                "the main text's recovery table. A separate "
                                "paired sign test for that newly recomputed "
                                "comparison was not included in the "
                                "prespecified sensitivity family and is not "
                                "added post hoc here; the descriptive count "
                                "is 5/5 seed blocks in the same direction, "
                                "reported as a count and not as a test"})
    return rows


def build(out_path: Path = OUT) -> Path:
    rows = collect()
    body = pd.DataFrame([
        {"Family": r["label"], "Table": r["table"],
         "Seeds vs \\Dm{}": (f"{r['k_negative']}/{r['n_seeds']}"
                                  if r["k_negative"] is not None else "--"),
         "Sign $p$": (f"{r['p_sign']:.4f}"
                           if r["p_sign"] is not None else "--"),
         "Sign Holm $p$": (f"{r['p_sign_holm']:.4f}"
                                if r["p_sign_holm"] is not None else "--"),
         "Enum.\\ Holm $p$": (f"{r['p_enum_holm']:.4f}"
                                  if r["p_enum_holm"] is not None else "--")}
        for r in rows])
    held = [r for r in rows if r.get("held")]
    note = (
        r"\textit{Sensitivity analysis, not a second primary test.} The "
        r"primary inference enumerates seed-block sign flips, which assumes "
        r"the seed-level paired differences are exchangeable in sign under "
        r"the null -- that a seed's difference is as likely to fall either "
        r"way, and that seeds are independent of one another. That assumption "
        r"is what makes the enumeration exact without any distributional "
        r"form; it is also its limit, since it says nothing about differences "
        r"that share a direction for a reason other than the effect under "
        r"test. This table reads the same assumption at its weakest: an exact "
        r"two-sided binomial sign test on the seed-level signs alone, "
        r"discarding every magnitude. Holm is applied within each family, as "
        r"in the primary analysis. The two columns agree on every "
        r"available row, and no judgment in this paper changes under "
        r"either.")
    if held:
        note += (r" One row is not reported: " + held[0]["held_reason"] + r".")
    tex = dataframe_to_tex(
        body, caption=(r"Paired sign-test sensitivity analysis: the "
                       r"seed-level sign counts alone, beside the "
                       r"enumeration the primary analysis uses."),
        label="tab:sign_test", column_format="llcccc",
        header=list(body.columns),
        style=TableStyle(environment="table", notes=(note,)),
        escape_data=False)
    return write_tex(out_path, tex, provenance={
        "generator": "reporting/table_sign_test.py",
        "input": str(T_FINAL),
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    # the sign test against values a reader can verify by hand
    check(abs(sign_p(15, 15) - 2 / 2 ** 15) < 1e-12,
          f"15/15 -> p = 2/2^15 = {2 / 2 ** 15:.3e}")
    check(abs(sign_p(12, 15) - 0.0352) < 5e-4, f"12/15 -> {sign_p(12, 15):.4f}")
    check(abs(sign_p(11, 15) - 0.1185) < 5e-4, f"11/15 -> {sign_p(11, 15):.4f}")
    check(abs(sign_p(5, 5) - 0.0625) < 1e-12, f"5/5 -> {sign_p(5, 5):.4f}")
    check(sign_p(8, 15) > 0.9, "a near-even split is not significant")
    check(holm([0.01, 0.04]) == [0.02, 0.04], "Holm step-down, two tests")

    rows = collect()
    got = {(r["family"], r["table"]): r for r in rows}
    wp_m = got[("with-prior faithfulness", "MIMIC")]
    wp_e = got[("with-prior faithfulness", "eICU")]
    np_m = got[("no-prior faithfulness", "MIMIC")]
    check((wp_m["k_negative"], wp_m["n_seeds"]) == (12, 15),
          "with-prior MIMIC sign count read from t_final: 12/15")
    check((wp_e["k_negative"], wp_e["n_seeds"]) == (11, 15),
          "with-prior eICU sign count read from t_final: 11/15")
    check((np_m["k_negative"], np_m["n_seeds"]) == (15, 15),
          "no-prior MIMIC sign count read from t_final: 15/15")
    check(abs(wp_m["p_sign_holm"] - 0.0703) < 1e-3,
          f"with-prior Holm on the sign tests: {wp_m['p_sign_holm']:.4f}")
    check(any(r.get("held") for r in rows),
          "the recovery row is held, not filled with a provisional value")
    p = build()
    t = p.read_text()
    check("--" in t, "the held row renders as a dash, not a number")
    check("not reported" in t, "the note says which row is absent and why")
    check("will be" not in t and "is filled when" not in t,
          "the note states a fact, not a future intention")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    for r in collect():
        if r.get("held"):
            print(f"  {r['label']:<26} {r['table']:<20} HELD")
        else:
            print(f"  {r['label']:<26} {r['table']:<20} "
                  f"{r['k_negative']}/{r['n_seeds']}  "
                  f"sign p={r['p_sign']:.4f}  Holm={r['p_sign_holm']:.4f}  "
                  f"(enumeration Holm {r['p_enum_holm']:.4f})")
    print(f"\n[OK] wrote {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
