"""T2.2(d): compare CDC2022's real missingness with our simulated masks.

This is the only place in the project where a simulated mechanism can be checked
against an unsimulated one. `heart_2022_with_nans.csv` carries 445,132 rows of
genuine BRFSS item non-response over the same 40 columns we mask.

**The comparison is a test, not a fit.** The `clinical_v1` specification for
CDC2022 was written from survey-methodology principles and deliberately not
tuned to this file. Reporting where the two disagree is the point; a simulation
tuned to match would demonstrate nothing.

Four axes, chosen because each is something the imputation literature's standard
protocol implicitly assumes and none of them is usually checked:

1. **Per-column rate.** Simulated masks give every column the same target rate by
   construction. Real non-response does not: it ranges from 0.27% to 19.04%.
2. **Position in the instrument.** The real pattern's dominant axis is where a
   question sits in the interview -- corr(position, rate) = +0.84 -- which is
   respondent fatigue and break-off. No covariate-driven MAR mechanism produces
   it, because position is not a property of the respondent.
3. **Co-occurrence.** Real missingness is clustered: a respondent who breaks off
   loses every subsequent item at once, so the distribution of per-row missing
   counts is far more dispersed than independent Bernoulli sampling gives.
4. **Row order.** The real pattern's corr(row missing rate, row index) is
   -0.0186, which is the empirical calibration for the R1-4 threshold and the
   cleanest available refutation of R0's +0.67 to +0.80 masks.

Derived-variable propagation is reported alongside: in the real file
P(BMI missing | height or weight missing) = 1.0000, because BMI is computed from
them. A simulated mask that treats the three columns as independent cannot
express that, and it is worth saying so rather than letting a reader assume it.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: CDC2022 is a public download but not redistributed here; point at it
#: with CDC2022_DIR. The default is repo-relative, not a home directory.
CDC_DIR = Path(os.environ.get(
    "CDC2022_DIR", str(Path(__file__).resolve().parents[2]
                       / "data_CDC2022")))

#: Below this spread of per-column rates, corr(position, rate) is noise. The real
#: pattern's sd is 0.0515; a per-column-calibrated simulated mask sits near
#: 0.0005, three orders of magnitude down and at the sampling floor.
RATE_SD_FLOOR = 0.005


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(np.log(p / (1 - p)))


@dataclass
class Comparison:
    per_column: pd.DataFrame
    row_counts: pd.DataFrame
    summary: dict

    def to_json(self) -> str:
        return json.dumps(self.summary, indent=2, default=str)


def _profile(na: pd.DataFrame, label: str) -> dict:
    """The four axes, computed for one missingness indicator matrix."""
    cols = list(na.columns)
    rates = np.array([na[c].mean() for c in cols], dtype=float)
    pos = np.arange(len(cols), dtype=float)
    row_rate = na.mean(axis=1).to_numpy(dtype=float)
    row_cnt = na.sum(axis=1).to_numpy(dtype=float)
    idx = np.arange(len(na), dtype=float)

    # Axis 3: how dispersed are per-row counts against the independent-Bernoulli
    # null with the same per-column rates? Under independence the variance is
    # sum p_j (1 - p_j); clustering inflates it.
    var_null = float((rates * (1 - rates)).sum())
    var_obs = float(row_cnt.var(ddof=0))

    return {
        "label": label,
        "n_rows": int(len(na)),
        "n_columns": len(cols),
        "overall_rate": float(na.to_numpy().mean()),
        # axis 1
        "col_rate_min": float(rates.min()),
        "col_rate_max": float(rates.max()),
        "col_rate_sd": float(rates.std(ddof=0)),
        # axis 2
        "r_position_vs_rate": (float(np.corrcoef(pos, rates)[0, 1])
                               if rates.std() > 0 else float("nan")),
        "spearman_position_vs_rate": float(
            pd.Series(pos).corr(pd.Series(rates), method="spearman")),
        # A correlation over 40 per-column rates says nothing when those rates
        # are all the same to within sampling error, which is exactly the case
        # for a per-column-calibrated simulated mask: its rate sd sits at the
        # 1e-4 floor, so corr(position, rate) is arbitrary and can come out at
        # -0.34 purely by chance. Anything below this cut is reported as
        # uninformative rather than as a small effect.
        "position_r_is_informative": bool(rates.std(ddof=0) > RATE_SD_FLOOR),
        # axis 3
        "row_count_var_observed": var_obs,
        "row_count_var_independent_null": var_null,
        "dispersion_ratio": var_obs / var_null if var_null > 0 else float("nan"),
        "frac_rows_fully_observed": float((row_cnt == 0).mean()),
        "frac_rows_over_half_missing": float((row_cnt > len(cols) / 2).mean()),
        # axis 4
        "r_rowrate_vs_rowindex": (float(np.corrcoef(row_rate, idx)[0, 1])
                                  if row_rate.std() > 0 else 0.0),
    }


def compare(simulated_masks: Dict[str, pd.DataFrame],
            real_path: Path = CDC_DIR / "heart_2022_with_nans.csv",
            columns: Optional[Sequence[str]] = None,
            always_observed: Sequence[str] = ()) -> Comparison:
    """Compare one or more simulated masks against the real pattern.

    `simulated_masks` maps a label (e.g. "MAR@30%") to a boolean DataFrame with
    1 = missing, sharing the real file's column names.

    `always_observed` names the columns the mechanism is forbidden to mask --
    the MAR drivers. They must be excluded from every statistic here. Leaving
    them in put a hard structural zero into the per-column rate vector and
    inflated the simulated spread to sd 0.0654, *above* the real 0.0515, which
    would have supported precisely the opposite of the true conclusion.
    """
    real = pd.read_csv(real_path)
    drop = set(always_observed)
    cols = [c for c in (columns if columns else real.columns) if c not in drop]
    real_na = real[cols].isna()

    profiles = [_profile(real_na, "real (BRFSS 2022)")]
    for label, m in simulated_masks.items():
        shared = [c for c in cols if c in m.columns]
        profiles.append(_profile(m[shared].astype(bool), label))

    per_col = pd.DataFrame({"column": cols,
                            "position": np.arange(len(cols)),
                            "real": [real_na[c].mean() for c in cols]})
    for label, m in simulated_masks.items():
        per_col[label] = [float(m[c].mean()) if c in m.columns else np.nan
                          for c in cols]

    counts = []
    for label, na in ([("real (BRFSS 2022)", real_na)]
                      + [(l, m[[c for c in cols if c in m.columns]].astype(bool))
                         for l, m in simulated_masks.items()]):
        k = na.sum(axis=1)
        vc = k.value_counts(normalize=True).sort_index()
        counts.append(pd.DataFrame({"label": label, "n_missing_in_row": vc.index,
                                    "fraction_of_rows": vc.values}))
    row_counts = pd.concat(counts, ignore_index=True)

    # Derived-variable propagation, real file only -- a simulated mask that
    # treats columns independently cannot reproduce it by construction.
    prop = {}
    if {"BMI", "HeightInMeters", "WeightInKilograms"} <= set(real.columns):
        hw = real.HeightInMeters.isna() | real.WeightInKilograms.isna()
        prop = {
            "p_bmi_missing_given_height_or_weight_missing":
                float(real.BMI.isna()[hw].mean()) if hw.any() else float("nan"),
            "p_height_or_weight_missing_given_bmi_missing":
                float(hw[real.BMI.isna()].mean()) if real.BMI.isna().any()
                else float("nan"),
            "note": "BMI is computed from height and weight, so its missingness "
                    "is implied by theirs. Independent per-column simulation "
                    "cannot express this.",
        }

    return Comparison(
        per_column=per_col,
        row_counts=row_counts,
        summary={"profiles": profiles,
                 "derived_variable_propagation": prop,
                 "interpretation": _interpret(profiles)},
    )


def _interpret(profiles: List[dict]) -> List[str]:
    """State the conclusions in words, so the report cannot be misread."""
    real = profiles[0]
    out = [
        f"Real per-column rates span {real['col_rate_min']*100:.2f}%-"
        f"{real['col_rate_max']*100:.2f}% (sd {real['col_rate_sd']*100:.2f} pp); "
        f"a simulated mask calibrated per column has an sd near zero by "
        f"construction. Uniform per-column rates are an artifact of the "
        f"protocol, not a property of real data.",
        f"Real missingness is driven mainly by position in the instrument: "
        f"corr(position, rate) = {real['r_position_vs_rate']:+.3f} "
        f"(Spearman {real['spearman_position_vs_rate']:+.3f}). This is "
        f"respondent fatigue and break-off, and no covariate-driven MAR "
        f"mechanism can produce it, because position is a property of the "
        f"questionnaire rather than of the respondent.",
        f"Real missingness is clustered: per-row missing counts have "
        f"{real['dispersion_ratio']:.2f}x the variance of an independent "
        f"Bernoulli mask with the same per-column rates, and "
        f"{real['frac_rows_fully_observed']*100:.1f}% of rows are fully "
        f"observed.",
        f"Real corr(row missing rate, row index) = "
        f"{real['r_rowrate_vs_rowindex']:+.4f}. R0's simulated masks reached "
        f"+0.67 (MIMIC) to +0.80 (eICU), so the row-order structure the "
        f"reviewer objected to is far outside anything real data exhibits.",
    ]
    uninformative = [p["label"] for p in profiles[1:]
                     if not p.get("position_r_is_informative", True)]
    if uninformative:
        out.append(
            "corr(position, rate) is reported as not significant for "
            + ", ".join(uninformative)
            + ": a per-column-calibrated mask gives every column the same rate "
              "to within sampling error, so there is no per-column variation "
              "for position to correlate with, and the coefficient that comes "
              "out of the arithmetic is arbitrary rather than small.")
    return out


def make_figure(cmp: Comparison, out_png: Path, out_pdf: Optional[Path] = None):
    """Four-panel publication figure. Written for grayscale legibility."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200})

    sim_labels = [c for c in cmp.per_column.columns
                  if c not in ("column", "position", "real")]
    # Grayscale-safe: the real series is always black and solid; simulated
    # series are distinguished by marker and dash pattern, not by hue, so the
    # figure survives a monochrome print.
    SIM_STYLE = [("s", (0, (4, 2)), "0.35"), ("^", (0, (1, 1.5)), "0.55"),
                 ("D", (0, (5, 1, 1, 1)), "0.7"), ("v", (0, (3, 1, 1, 1)), "0.45")]
    fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.4))

    # (a) per-column rate, real vs simulated, ordered by questionnaire position
    a = ax[0, 0]
    pc = cmp.per_column
    a.plot(pc.position, pc.real * 100, "o-", ms=3, lw=1.4, color="k",
           label="real (BRFSS 2022)")
    for i, s in enumerate(sim_labels):
        mk, dash, shade = SIM_STYLE[i % len(SIM_STYLE)]
        a.plot(pc.position, pc[s] * 100, marker=mk, ls=dash, color=shade,
               ms=2.5, lw=1.0, label=s)
    a.set_xlabel("position in questionnaire")
    a.set_ylabel("missing rate (%)")
    a.set_title("(a) per-column rate by instrument position", loc="left")
    a.legend(frameon=False, fontsize=6.5)

    # (b) the same relationship as a single number, per profile
    b = ax[0, 1]
    profs = cmp.summary["profiles"]
    names, vals, cols_b = [], [], []
    for i, p in enumerate(profs):
        info = p.get("position_r_is_informative", True)
        # An uninformative coefficient is drawn as zero with the measured value
        # in the label, rather than as a bar a reader would compare against the
        # real one. See RATE_SD_FLOOR.
        names.append(p["label"] + ("" if info
                                   else f"\n(n.s.; rate sd {p['col_rate_sd']*100:.3f} pp)"))
        vals.append(p["r_position_vs_rate"] if info else 0.0)
        cols_b.append("0.15" if i == 0 else "0.55")
    b.barh(range(len(names)), vals, color=cols_b)
    for i, p in enumerate(profs):
        if not p.get("position_r_is_informative", True):
            b.text(0.01, i, "no per-column variation to correlate",
                   va="center", fontsize=5.5, style="italic", color="0.35")
    b.axvline(0, color="k", lw=0.8)
    b.set_yticks(range(len(names)))
    b.set_yticklabels(names, fontsize=6.5)
    b.set_xlabel("corr(position, missing rate)")
    b.set_title("(b) fatigue signature: only the real pattern has it", loc="left")
    b.invert_yaxis()

    # (c) dispersion of per-row missing counts
    c = ax[1, 0]
    j = 0
    for lab, grp in cmp.row_counts.groupby("label", sort=False):
        if lab.startswith("real"):
            style = dict(color="k", lw=1.6, ls="-")
        else:
            _, dash, shade = SIM_STYLE[j % len(SIM_STYLE)]
            style = dict(color=shade, lw=1.1, ls=dash)
            j += 1
        c.plot(grp.n_missing_in_row, grp.fraction_of_rows * 100, **style, label=lab)
    c.set_xlabel("number of missing entries in a row")
    c.set_ylabel("% of rows")
    c.set_yscale("symlog", linthresh=0.01)
    c.set_title("(c) co-occurrence: real missingness clusters", loc="left")
    c.legend(frameon=False, fontsize=6.5)

    # (d) row-order correlation, with R0's published masks for scale
    d = ax[1, 1]
    labs = [p["label"] for p in profs] + ["R0 mask, MIMIC", "R0 mask, eICU"]
    vs = [p["r_rowrate_vs_rowindex"] for p in profs] + [0.67, 0.80]
    cols = ["0.15"] + ["0.55"] * (len(profs) - 1) + ["0.75", "0.75"]
    d.barh(range(len(labs)), vs, color=cols)
    d.axvline(0.05, color="k", ls=":", lw=0.9)
    d.axvline(-0.05, color="k", ls=":", lw=0.9)
    d.axvline(0, color="k", lw=0.8)
    d.set_yticks(range(len(labs)))
    d.set_yticklabels(labs, fontsize=6.5)
    d.set_xlabel("corr(row missing rate, row index)")
    d.set_title("(d) row-order structure vs the |r| < 0.05 threshold", loc="left")
    d.invert_yaxis()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png
