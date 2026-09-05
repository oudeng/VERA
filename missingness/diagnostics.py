"""T2.2(c): the six-item mask diagnostic report.

This is the evidence for reviewer point R1-4. R0's MAR masks used the record
index as their driver, and the resulting masks correlate with row order at
+0.67 (MIMIC) to +0.80 (eICU): rows near the top of the file are about 7% missing
and rows near the bottom about 49%, in a table nominally at 30%. The reviewer
asked for that to be *replaced*, not supplemented, so the replacement has to be
demonstrated rather than asserted.

Six items, each answering a question a sceptical reader would ask:

1. Does the mask still encode row order?         corr(row missing rate, row index)
2. If so, is that the mechanism or the table?    corr(driver, row index)
3. Did every column hit its target rate?         per-column |achieved - target|
4. Is per-column heterogeneity real?             per-column sensitivity to each driver
5. Is the mechanism the same at 10/30/50%?       driver log-odds contrast across rates
6. Are drivers observable, as MAR requires?      driver missing count under every mechanism

Item 5 exists because of B49: R0's mean-linear rescale made the three rates three
different mechanisms rather than three strengths of one, so the multi-rate figure
was not varying what it appeared to vary.

Item 2 exists because of B51: three of the derived tables arrive sorted, and
AutoMPG is sorted by `model_year`, which is itself a legitimate MAR driver. A
non-zero correlation there is a property of the table, not a defect of the
mechanism, and the two must be distinguishable without re-running anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Acceptance threshold for item 1, inherited from T1.5 and calibrated against
#: real data in P2: the CDC2022 `heart_2022_with_nans` table carries a genuine,
#: unsimulated missingness pattern over 445,132 rows, and its corr(row missing
#: rate, row index) is **-0.0186**. So a real pattern sits comfortably inside
#: this threshold while R0's simulated masks sat at +0.67 to +0.80.
ROW_INDEX_THRESHOLD = 0.05

#: Acceptance threshold for item 3.
RATE_TOLERANCE = 0.01


def null_sigma(n: int) -> float:
    """Standard deviation of corr(row rate, row index) under exchangeability.

    Row order carries no information under the null, so the correlation is
    asymptotically N(0, 1/(n-1)). Reporting this alongside the coefficient keeps
    the threshold honest: |r| < 0.05 is a 4.5 sigma test at n=8000 but only a
    1-2 sigma test at n=392, and a reader is entitled to know which.
    """
    return float(1.0 / np.sqrt(max(n - 1, 1)))


def row_index_correlation(mask: pd.DataFrame, cols: Sequence[str]) -> float:
    rate = mask[list(cols)].mean(axis=1).to_numpy(dtype=float)
    idx = np.arange(len(rate), dtype=float)
    if rate.std() == 0:
        return 0.0
    return float(np.corrcoef(rate, idx)[0, 1])


def driver_sensitivity(mask: pd.DataFrame, table: pd.DataFrame,
                       drivers: Sequence[str], targets: Sequence[str]) -> pd.DataFrame:
    """Point-biserial correlation between each driver and each column's mask.

    Item 4. Under R0's row-broadcast MAR every column shared one propensity
    vector, so this matrix had identical rows by construction. Genuine per-column
    heterogeneity shows up as a spread of values within a driver's column,
    including sign changes where the configuration assigned opposite betas.
    """
    rows = []
    for t in targets:
        m = mask[t].to_numpy(dtype=float)
        rec = {"column": t}
        for d in drivers:
            if d not in table.columns:
                rec[d] = np.nan
                continue
            x = pd.to_numeric(table[d], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x)
            rec[d] = (float(np.corrcoef(x[ok], m[ok])[0, 1])
                      if ok.sum() > 2 and np.nanstd(x[ok]) > 0 and m[ok].std() > 0
                      else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def logodds_contrast(mask: pd.DataFrame, table: pd.DataFrame, driver: str,
                     targets: Sequence[str], q: float = 0.1) -> float:
    """Log-odds of being missing in the driver's top decile vs its bottom decile.

    Item 5. Under a true logistic mechanism with a fixed coefficient, changing
    the target rate only shifts the intercept, so this contrast stays put.

    It is a monotone summary, not an exact recovery of the coefficient: it takes
    the logit of a mean rather than the mean of logits, so the value does not
    equal ``beta * (z_hi - z_lo)``. Calibrated on synthetic data with beta = 1.5
    and n = 20,000 it returns 5.51 / 5.12 / 5.13 at rates 0.1 / 0.3 / 0.5 -- a
    spread of 7.5%, which is the sampling floor. R0's masks returned 2.19 / 2.70
    / 4.06, a spread of 63%. The statistic is therefore used as a discriminator
    between "one mechanism at three strengths" and "three different mechanisms"
    (finding B49), not as an estimator of beta.
    """
    if driver not in table.columns:
        return float("nan")
    x = pd.to_numeric(table[driver], errors="coerce").to_numpy(dtype=float)
    r = mask[list(targets)].mean(axis=1).to_numpy(dtype=float)
    ok = np.isfinite(x)
    x, r = x[ok], r[ok]
    lo, hi = np.quantile(x, q), np.quantile(x, 1 - q)
    # A near-constant driver makes the two deciles the same set of rows, and the
    # statistic then compares a group with itself and returns something close to
    # zero that looks like a measurement. NHANES's fasting_state_std is 95.9 %
    # ones after the complete-case step, so q10 == q90 == 1 and the "contrast"
    # of -0.15 it produced was an artifact, not a weak effect. Undefined is the
    # honest answer.
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float("nan")
    a, b = r[x <= lo], r[x >= hi]
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    return _logit(float(b.mean())) - _logit(float(a.mean()))


@dataclass
class MaskBundle:
    """One generated (dataset, mechanism, rate) triple, loaded from disk."""
    dataset: str
    mechanism: str
    rate: float
    mask: pd.DataFrame
    table: pd.DataFrame
    meta: dict

    @property
    def drivers(self) -> List[str]:
        """Every driver the mechanism used. Empty under MCAR, which has none."""
        union = ((self.meta.get("spec", {}) or {}).get("mar", {}) or {}
                 ).get("driver_union")
        if union:
            return list(union)
        pc = self.meta.get("per_column_spec", {}) or {}
        out: List[str] = []
        for spec in pc.values():
            for d in (spec or {}).get("drivers", []) or []:
                if d not in out:
                    out.append(d)
        return out

    @property
    def targets(self) -> List[str]:
        """The columns the mechanism was allowed to mask.

        These live under `meta["spec"]["target_columns"]`. Reading them from the
        top level of `meta` -- as the first version did -- silently returned an
        empty list, the caller fell back to "every column but ID", and the
        always-observed drivers were then judged against a target rate they were
        never meant to meet. Item 3 duly reported a rate error equal to the full
        target rate for exactly those columns.
        """
        spec = self.meta.get("spec", {}) or {}
        cols = spec.get("target_columns") or self.meta.get("target_columns")
        if cols:
            return list(cols)
        observed = set(spec.get("observed_columns") or [])
        ident = {c for c in self.table.columns if c.upper() == "ID"}
        return [c for c in self.table.columns if c not in observed | ident]


def load_bundle(mask_dir: Path, dataset: str, mechanism: str, rate: float,
                table_path: Path) -> MaskBundle:
    """Load one generated triple.

    `generate_and_write` nests its output under `<outdir>/<dataset>/`, and the
    rate tag is zero-padded to two digits (`_rate_tag`, generator.py:443). Both
    layouts are accepted so a hand-assembled flat directory still works.
    """
    stem = f"{dataset}_{mechanism}_{int(round(rate * 100)):02d}per"
    for base in (mask_dir / dataset, mask_dir):
        meta_p, mask_p = base / f"{stem}_meta.json", base / f"{stem}_mask.npy"
        if meta_p.exists() and mask_p.exists():
            break
    else:
        raise FileNotFoundError(f"{stem} not found under {mask_dir}")

    meta = json.loads(meta_p.read_text())
    arr = np.load(mask_p).astype(bool)
    table = pd.read_csv(table_path)
    mask = pd.DataFrame(arr, columns=list(table.columns))
    return MaskBundle(dataset, mechanism, rate, mask, table, meta)


def diagnose_bundle(b: MaskBundle) -> dict:
    n = len(b.mask)
    sigma = null_sigma(n)
    targets = b.targets or [c for c in b.table.columns if c != "ID"]
    drivers = b.drivers

    r_row = row_index_correlation(b.mask, targets)

    # Item 3, computed from the mask rather than read out of the metadata. The
    # first version looked for `per_column_spec[c]["abs_rate_error"]`, which does
    # not exist -- the generator records rates under `rates.per_column_missing
    # _rate` -- so every value came back NaN and the criterion silently checked
    # nothing.
    rate_err = {c: abs(float(b.mask[c].mean()) - float(b.rate)) for c in targets}
    worst = max(rate_err.values()) if rate_err else float("nan")
    # The generator records its own verdict; disagreeing with it would mean one
    # of the two is wrong, so it is surfaced rather than trusted silently.
    gen_outside = (b.meta.get("rates", {}) or {}).get("columns_outside_tolerance")

    drv_obs = {d: int(b.mask[d].sum()) for d in drivers if d in b.mask.columns}

    out = {
        "dataset": b.dataset, "mechanism": b.mechanism, "rate": b.rate, "n": n,
        # item 1
        "row_index_r": r_row,
        "row_index_null_sigma": sigma,
        "row_index_sigmas": abs(r_row) / sigma if sigma else float("nan"),
        # The fixed 0.05 criterion inherited from T1.5. Kept unchanged so the
        # threshold cannot be accused of moving once the results are in.
        "row_index_pass": abs(r_row) < ROW_INDEX_THRESHOLD,
        # ...and the same question asked in a way that respects the sample size.
        # At n = 392 the null sd is 0.0506, so 0.05 is a 1-sigma test and a mask
        # drawn from a genuinely null mechanism fails it about a third of the
        # time; at n = 8000 the same threshold is 4.5 sigma. Both columns are
        # reported and the report says which is which.
        "row_index_pass_noise_adjusted": abs(r_row) < max(ROW_INDEX_THRESHOLD,
                                                          3.0 * sigma),
        # item 2
        "driver_vs_rowindex": (b.meta.get("row_index_diagnostics", {})
                               .get("pearson_r_driver_vs_rowindex", {})),
        # item 3
        "max_abs_rate_error": worst,
        "n_columns_out_of_tolerance": int(sum(v > RATE_TOLERANCE
                                              for v in rate_err.values())),
        "n_target_columns": len(targets),
        "rate_pass": bool(worst <= RATE_TOLERANCE) if rate_err else None,
        "generator_columns_outside_tolerance": gen_outside,
        "rate_verdict_agrees_with_generator": (
            None if gen_outside is None
            else (len(gen_outside) == 0) == (worst <= RATE_TOLERANCE)),
        # item 6
        "driver_missing_counts": drv_obs,
        "drivers_fully_observed": all(v == 0 for v in drv_obs.values()),
        "drivers": drivers,
    }
    return out


def diagnose_grid(mask_dir: Path, tables: Dict[str, Path],
                  mechanisms: Sequence[str] = ("MCAR", "MAR", "MNAR"),
                  rates: Sequence[float] = (0.1, 0.3, 0.5)) -> dict:
    """Run all six items over a whole grid and assemble the report tables."""
    rows, sens, contrast = [], [], []
    for ds, tpath in tables.items():
        for mech in mechanisms:
            for rate in rates:
                try:
                    b = load_bundle(mask_dir, ds, mech, rate, tpath)
                except FileNotFoundError:
                    continue
                rows.append(diagnose_bundle(b))

                targets = b.targets or [c for c in b.table.columns if c != "ID"]
                if b.drivers:
                    s = driver_sensitivity(b.mask, b.table, b.drivers, targets)
                    s.insert(0, "rate", rate)
                    s.insert(0, "mechanism", mech)
                    s.insert(0, "dataset", ds)
                    sens.append(s)
                    # Restrict each driver to the columns that actually declare
                    # it. Averaging over every target column instead dilutes a
                    # strong driver into noise: NHANES's fasting_state_std
                    # governs 2 of 11 columns with a log-odds contrast near 4,
                    # and spreading it over all 11 returned a table-level
                    # contrast of -0.045 -- a value so close to zero that the
                    # relative spread against it was meaningless.
                    pc = b.meta.get("per_column_spec", {}) or {}
                    for d in b.drivers:
                        used = [c for c in targets
                                if d in ((pc.get(c) or {}).get("drivers") or [])]
                        if not used:
                            continue
                        contrast.append({
                            "dataset": ds, "mechanism": mech, "rate": rate,
                            "driver": d, "n": len(b.mask),
                            "n_columns_driven": len(used),
                            "logodds_contrast": logodds_contrast(b.mask, b.table,
                                                                 d, used),
                        })

    summary = pd.DataFrame(rows)
    sensitivity = pd.concat(sens, ignore_index=True) if sens else pd.DataFrame()
    contrasts = pd.DataFrame(contrast)

    # Item 5 verdict: for a fixed (dataset, mechanism, driver), how much does the
    # contrast move across the three rates? A true logistic mechanism holds it
    # constant, so the spread is the statistic of interest.
    stability = pd.DataFrame()
    if not contrasts.empty:
        g = contrasts.groupby(["dataset", "mechanism", "driver"])
        stability = g["logodds_contrast"].agg(["min", "max", "mean", "std"]).reset_index()
        stability["n"] = g["n"].first().to_numpy()
        stability["spread"] = stability["max"] - stability["min"]
        stability["rel_spread"] = (stability["spread"]
                                   / stability["mean"].abs().replace(0, np.nan))
        # The spread has to be read against the sample size. The statistic is a
        # logit of a decile mean, so its sampling error scales roughly as
        # 1/sqrt(n * q); at n = 20,000 the floor is 7.5% (measured on synthetic
        # data), and it grows as sqrt(20000/n). A dataset of 392 rows therefore
        # has a floor near 54%, and a rel_spread of that order is noise, not
        # evidence of B49.
        stability["noise_floor_rel_spread"] = (
            0.075 * np.sqrt(20000.0 / stability["n"].clip(lower=1)))
        # The relative spread divides by the mean, so it blows up whenever the
        # mean contrast is near zero -- which happens for a genuinely weak
        # driver, not for a broken one. The verdict therefore requires BOTH a
        # relative spread above the noise floor AND a mean large enough for the
        # ratio to mean anything. B49's signature was a 63 % spread on a mean of
        # 3.0; a 150 % spread on a mean of 0.04 is arithmetic, not evidence.
        MIN_MEAN_CONTRAST = 0.5
        stability["mean_is_interpretable"] = (stability["mean"].abs()
                                              >= MIN_MEAN_CONTRAST)
        stability["exceeds_noise_floor"] = (
            (stability["rel_spread"] > stability["noise_floor_rel_spread"])
            & stability["mean_is_interpretable"])

    return {"summary": summary, "sensitivity": sensitivity,
            "contrasts": contrasts, "rate_stability": stability}
