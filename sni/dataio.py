from __future__ import annotations

"""Data I/O helpers.

These helpers enforce a **stable schema** for mixed-type tabular CSVs.

Motivation
----------
CSV files that contain missing values will often coerce integer columns into
``float64`` (e.g., ``82`` becomes ``82.0``). This is especially problematic for
integer-coded categorical variables: downstream models may predict labels as
integers/strings, and assigning them back into a float column triggers pandas
dtype warnings (and will become an error in future pandas versions).

This module keeps categorical columns as ``Int64`` (nullable integer) whenever
possible, and casts the *_missing.csv table to match the schema derived from
the corresponding *_complete.csv table.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


Schema = Dict[str, str]

def _strip_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with whitespace-trimmed column names.

    This prevents brittle failures when upstream CSV headers accidentally include
    trailing spaces (e.g., ``'ConcreteCS '``).
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    # Guard against accidental duplicates after stripping.
    if len(set(out.columns)) != len(out.columns):
        raise ValueError(
            "Duplicate column names after stripping whitespace. "
            "Please sanitize the dataset headers."
        )
    return out


def _clean_var_list(vars: List[str]) -> List[str]:
    """Clean a list of variable names.

    - drop None/NaN/empty tokens
    - strip leading/trailing whitespace
    - drop the literal token 'nan'
    """
    out: List[str] = []
    for v in vars:
        if v is None:
            continue
        # In case callers pass NaN values
        if isinstance(v, float) and np.isnan(v):
            continue
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            continue
        out.append(s)
    return out




def _is_int_like(series: pd.Series, *, tol: float = 1e-6) -> bool:
    """Return True if values are (almost) all integers."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return False
    frac = (s - np.round(s)).abs()
    return float((frac < tol).mean()) > 0.99


def infer_schema_from_complete(
    df_complete: pd.DataFrame,
    *,
    categorical_vars: List[str],
    continuous_vars: List[str],
) -> Schema:
    """Infer target dtypes from *_complete.csv + variable lists.

    Rules:
      - continuous vars -> float64
      - categorical vars -> Int64 if numeric & integer-like, else category
    """
    schema: Schema = {}

    for c in continuous_vars:
        schema[c] = "float64"

    for c in categorical_vars:
        if c not in df_complete.columns:
            continue
        dt = df_complete[c].dtype
        if pd.api.types.is_integer_dtype(dt):
            schema[c] = "Int64"
        elif pd.api.types.is_float_dtype(dt) and _is_int_like(df_complete[c]):
            schema[c] = "Int64"
        else:
            # fallback for non-numeric categories
            schema[c] = "category"

    return schema


def cast_dataframe_to_schema(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Cast a dataframe in-place to the provided schema.

    The function is *robust* to common CSV artifacts, such as integer columns
    serialized as floats ("82.0") and empty strings.
    """
    out = df.copy()
    for col, dtype in schema.items():
        if col not in out.columns:
            continue

        if dtype == "Int64":
            num = pd.to_numeric(out[col], errors="coerce")
            # round to nearest integer (safe for integer-coded categories)
            num = np.rint(num)
            out[col] = pd.Series(num, index=out.index).astype("Int64")

        elif dtype in {"float64", "Float64"}:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

        elif dtype == "category":
            # keep NaNs
            out[col] = out[col].astype("category")

        else:
            out[col] = out[col].astype(dtype)

    return out


def load_complete_and_missing(
    *,
    input_complete: str,
    input_missing: str,
    categorical_vars: List[str],
    continuous_vars: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Schema]:
    """Load complete/missing CSVs and cast missing -> schema(complete).

    This helper enforces:
      1) **Column-name stability** via whitespace stripping.
      2) **Row alignment** between complete/missing tables when an ``ID`` column
         exists in both files.
      3) **Type stability**: categorical integer-coded columns remain ``Int64``.

    Notes
    -----
    We read *_missing.csv with ``dtype=str`` to avoid pandas inferring ``float64``
    for integer-coded categorical columns.
    """
    categorical_vars = _clean_var_list(categorical_vars)
    continuous_vars = _clean_var_list(continuous_vars)
    all_vars = list(categorical_vars) + list(continuous_vars)

    # Read raw tables (strip header whitespace early).
    df_complete_raw = _strip_df_columns(pd.read_csv(input_complete))
    df_missing_raw = _strip_df_columns(pd.read_csv(input_missing, dtype=str))

    # ---------------------------------------------------------------------
    # Optional but recommended: align rows by an explicit ID column.
    # This prevents silent metric corruption if a missing CSV is shuffled.
    # ---------------------------------------------------------------------
    if "ID" in df_complete_raw.columns and "ID" in df_missing_raw.columns:
        idc = "ID"
        # ID must be fully observed; otherwise alignment becomes ambiguous.
        if df_complete_raw[idc].isna().any():
            raise ValueError(f"Missing ID values in complete CSV: {input_complete}")
        if df_missing_raw[idc].isna().any():
            raise ValueError(
                f"Missing ID values in missing CSV: {input_missing}. "
                "This usually happens if the ID column was accidentally masked during "
                "missing-data generation. Regenerate with --exclude-cols ID."
            )

        # Normalize to string for stable matching.
        df_complete_raw[idc] = df_complete_raw[idc].astype(str)
        df_missing_raw[idc] = df_missing_raw[idc].astype(str)

        if df_complete_raw[idc].duplicated().any():
            raise ValueError(f"Duplicate {idc} values in complete CSV: {input_complete}")
        if df_missing_raw[idc].duplicated().any():
            raise ValueError(f"Duplicate {idc} values in missing CSV: {input_missing}")

        df_complete_raw = df_complete_raw.set_index(idc)
        df_missing_raw = df_missing_raw.set_index(idc)

        ids_complete = set(df_complete_raw.index)
        ids_missing = set(df_missing_raw.index)
        if ids_complete != ids_missing:
            missing_ids = sorted(list(ids_complete - ids_missing))[:10]
            extra_ids = sorted(list(ids_missing - ids_complete))[:10]
            raise ValueError(
                "ID mismatch between complete and missing CSVs. "
                f"Missing in missing: {missing_ids}; Extra in missing: {extra_ids}."
            )

        # Reorder missing to match complete.
        df_missing_raw = df_missing_raw.loc[df_complete_raw.index]

        # Restore ID as a normal column (so downstream selection can keep/drop it).
        df_complete_raw = df_complete_raw.reset_index()
        df_missing_raw = df_missing_raw.reset_index()

    df_complete = df_complete_raw
    df_missing = df_missing_raw

    # Column existence check
    missing_in_complete = [c for c in all_vars if c not in df_complete.columns]
    if missing_in_complete:
        raise KeyError(
            f"Columns {missing_in_complete} not found in complete CSV: {input_complete}. "
            f"Available columns: {list(df_complete.columns)}"
        )

    missing_in_missing = [c for c in all_vars if c not in df_missing.columns]
    if missing_in_missing:
        raise KeyError(
            f"Columns {missing_in_missing} not found in missing CSV: {input_missing}. "
            f"Available columns: {list(df_missing.columns)}"
        )

    # Keep only configured variables (drop ID unless explicitly listed).
    df_complete = df_complete[all_vars]
    df_missing = df_missing[all_vars]

    # Infer schema from complete, then cast both frames accordingly.
    schema = infer_schema_from_complete(
        df_complete, categorical_vars=categorical_vars, continuous_vars=continuous_vars
    )
    df_complete = cast_dataframe_to_schema(df_complete, schema)
    df_missing = cast_dataframe_to_schema(df_missing, schema)

    return df_complete, df_missing, schema
