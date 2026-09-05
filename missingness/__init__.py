"""Missing-mechanism simulator for code_SNI (task T1.5, reviewer point R1-4).

A rewrite of ``project_sni_R0/sni/utility_missing_data_gen_v1/missing_data_generator.py``
(815 lines) that keeps R0's strict-MAR guarantee and its rate bookkeeping while
replacing the propensity model. The correct pieces were ported from R0's own
synthetic generator, ``project_sni_R0/sni/scripts/synth_generate_s5.py:270-345``,
which already did per-column heterogeneous slopes and logit-space bisection —
the main simulator simply never used them.

What changed, and why
---------------------
========  ==================================================================
finding   fix
========  ==================================================================
R1-4      ``configs/missingness.yaml`` declares clinically-defensible drivers
          per dataset and per target column. R0 used ``ID`` — a record
          counter — as the sole MAR driver for all six datasets, so the
          per-row missing rate correlated with row position at r = +0.67
          (MIMIC) to +0.80 (eICU). See :mod:`missingness.spec`.
--        ``_mar_propensity``'s ``np.repeat`` row broadcast
          (``missing_data_generator.py:467``) replaced by an independent
          logistic per target column. See :func:`missingness.propensity.mar_propensity`.
--        the mean-linear rescale ``p * (rate/mean_p)``
          (``missing_data_generator.py:460-464``) replaced by logit-space
          bisection on the intercept. See :func:`missingness.calibration.solve_intercept`.
B45       one shared ``Generator`` consumed in column order replaced by
          name-keyed independent streams. See :mod:`missingness.rng`.
B46       zero-variance categorical columns (eICU ``vasopressor_use_std``)
          degenerate to a constant target-rate propensity instead of
          ``rate * 1.5`` for every row.
B39       per-column rate calibration, so no column overshoots its target
          (R0's MIMIC ``ALARM`` reached 37.8% against a 30% target).
B38/E4    every written mask is reloaded through ``common.masks.load_and_verify``
          and asserted against ``X_missing.isna()``, in memory and after the
          CSV round-trip.
========  ==================================================================

Nothing here generates masks for the real datasets. Driver selection is a
first-author decision pending task T1.6; ``configs/missingness.yaml`` ships a
schema, a labeled ``_PLACEHOLDER`` example per dataset, and a
``record_index_ID`` profile that reproduces R0 bit-for-bit for comparison.

Quick use::

    from missingness import generate_from_config
    res = generate_from_config(df, "eICU", "MAR", 0.3, profile="clinical_v1_PLACEHOLDER")
    res.row_index_correlation()   # ~0
"""

from .calibration import (  # noqa: F401
    calibrate_column_to_count,
    enforce_min_missing,
    fit_logistic_propensity,
    sigmoid,
    solve_intercept,
    standardize,
)
from .generator import (  # noqa: F401
    MissingnessResult,
    generate,
    generate_and_write,
    generate_from_config,
)
from .propensity import (  # noqa: F401
    build_propensity_matrix,
    mar_propensity,
    mcar_propensity,
    mnar_propensity,
)
from .rng import StreamRegistry, stable_key  # noqa: F401
from .spec import (  # noqa: F401
    MARColumnSpec,
    MissingnessSpec,
    MNARColumnSpec,
    dataset_schema,
    load_config,
    resolve,
    schema_from_frame,
)

__all__ = [
    "MissingnessResult", "MissingnessSpec", "MARColumnSpec", "MNARColumnSpec",
    "StreamRegistry", "build_propensity_matrix", "calibrate_column_to_count",
    "dataset_schema", "enforce_min_missing", "fit_logistic_propensity", "generate",
    "generate_and_write", "generate_from_config", "load_config", "mar_propensity",
    "mcar_propensity", "mnar_propensity", "resolve", "schema_from_frame", "sigmoid",
    "solve_intercept", "stable_key", "standardize",
]
