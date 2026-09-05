"""code_SNI.baselines -- the eight reference imputers, de-leaked (P1 T1.4).

Primary interface::

    from baselines import build_baseline_imputer
    from baselines.schema import DataSchema

    schema = DataSchema.from_yaml("configs/datasets.yaml", "MIMIC")
    imp = build_baseline_imputer("MICE", schema.categorical_vars, schema.continuous_vars, seed=1)
    X_imputed = imp.impute(X_missing, schema)          # X_complete is NOT a parameter

Fit/transform (reviewer R1-3), for MeanMode / KNN / MICE / MissForest / HyperImpute::

    imp.fit(X_train_missing, schema)
    X_test_imputed = imp.transform(X_test_missing)

GAIN / MIWAE / TabCSDI deliberately do not implement fit/transform; see
``registry.FIT_TRANSFORM_ADJUDICATION`` and ``registry.FOLD_INDEPENDENT_PROTOCOL``.
"""

from .registry import (  # noqa: F401
    BaseBaseline,
    FIT_TRANSFORM_ADJUDICATION,
    FOLD_INDEPENDENT_PROTOCOL,
    ORACLE_USAGE_R0,
    build_baseline_imputer,
    list_baselines,
)
from .schema import DataSchema, ObservedStats  # noqa: F401
