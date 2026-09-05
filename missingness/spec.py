"""Declarative specification of a missingness mechanism (engineering principle E1).

The spec is resolved from ``configs/missingness.yaml`` plus the column roles in
``configs/datasets.yaml``, and it is what gets written verbatim into
``meta.json``. Nothing about a generated mask should require reading the code to
understand: drivers, coefficients, fitted intercepts, modes and seeds are all
here.

Resolution order (later wins)::

    configs/missingness.yaml : defaults
      <- profiles[<profile>].defaults
      <- profiles[<profile>].datasets[<DS>].common
      <- profiles[<profile>].datasets[<DS>][<MECHANISM>]
      <- explicit keyword arguments to `resolve`
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

_CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MISSINGNESS_CONFIG = _CODE_ROOT / "configs" / "missingness.yaml"
DEFAULT_DATASETS_CONFIG = _CODE_ROOT / "configs" / "datasets.yaml"

MECHANISMS = ("MCAR", "MAR", "MNAR")

#: MAR propensity construction modes.
MAR_MODES = (
    "per_column",            # R1 default: independent propensity per target column
    "row_broadcast_legacy",  # R0 behavior, reproduced for comparison only
)

#: MNAR propensity modes for continuous columns.
MNAR_CONT_MODES = ("logit", "quantile_steps")

#: MNAR propensity modes for categorical columns.
MNAR_CAT_MODES = ("ordinal", "semantic_groups", "random_split")

#: Per-column rate calibration policies.
RATE_CALIBRATION = ("per_column", "table", "none")

#: Generation engines. ``legacy_R0`` bypasses the rewrite entirely.
IMPLEMENTATIONS = ("rewrite", "legacy_R0")

#: Row-order policies.
ROW_ORDER_MODES = ("as_is", "shuffle")

#: Coefficient may be a number (applied to the standardized driver) or a mapping
#: from category level to an additive logit offset.
Coefficient = Union[float, Mapping[Any, float]]


# ---------------------------------------------------------------------------
# dataset schema (read-only view of configs/datasets.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnMeta:
    name: str
    type: str            # continuous | categorical | integer_index
    role: str            # imputable | identifier
    levels: Optional[List[Any]] = None   # declared ordinal order, if any
    issues: List[str] = field(default_factory=list)

    @property
    def is_categorical(self) -> bool:
        return self.type == "categorical"

    @property
    def is_identifier(self) -> bool:
        return self.role == "identifier" or self.type == "integer_index"


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    columns: Dict[str, ColumnMeta]
    identifier_column: Optional[str]
    downstream_target: Optional[str]

    def order(self) -> List[str]:
        return list(self.columns.keys())

    def continuous(self) -> List[str]:
        return [c.name for c in self.columns.values() if c.type == "continuous"]

    def categorical(self) -> List[str]:
        return [c.name for c in self.columns.values() if c.type == "categorical"]

    def imputable(self) -> List[str]:
        return [c.name for c in self.columns.values() if c.role == "imputable"]


def load_datasets_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return yaml.safe_load(Path(path or DEFAULT_DATASETS_CONFIG).read_text())


def dataset_schema(name: str, *, path: Optional[Path] = None) -> DatasetSchema:
    """Build a :class:`DatasetSchema` from ``configs/datasets.yaml``.

    That file is the single source of truth for column roles and types (E1); this
    module never re-infers types from dtypes the way R0's
    ``missing_data_generator.infer_column_types`` (:108-193) did.
    """
    cfg = load_datasets_config(path)
    try:
        ds = cfg["datasets"][name]
    except KeyError as exc:  # pragma: no cover - config error path
        raise KeyError(f"dataset {name!r} not declared in {path or DEFAULT_DATASETS_CONFIG}") from exc
    cols: Dict[str, ColumnMeta] = {}
    for cname, c in ds["columns"].items():
        cols[cname] = ColumnMeta(
            name=cname,
            type=str(c.get("type", "continuous")),
            role=str(c.get("role", "imputable")),
            levels=list(c["levels"]) if c.get("levels") is not None else None,
            issues=list(c.get("issues", []) or []),
        )
    return DatasetSchema(
        name=name,
        columns=cols,
        identifier_column=ds.get("identifier_column"),
        downstream_target=ds.get("downstream_target"),
    )


def schema_from_frame(df, *, categorical: Sequence[str] = (), identifier: Optional[str] = None) -> DatasetSchema:
    """Ad-hoc schema for synthetic frames in tests. Not used for real datasets."""
    cat = set(categorical)
    cols: Dict[str, ColumnMeta] = {}
    for c in df.columns:
        if identifier is not None and c == identifier:
            cols[c] = ColumnMeta(name=c, type="integer_index", role="identifier")
        elif c in cat:
            cols[c] = ColumnMeta(name=c, type="categorical", role="imputable")
        else:
            cols[c] = ColumnMeta(name=c, type="continuous", role="imputable")
    return DatasetSchema(name="_adhoc", columns=cols, identifier_column=identifier, downstream_target=None)


# ---------------------------------------------------------------------------
# per-column mechanism specs
# ---------------------------------------------------------------------------

@dataclass
class MARColumnSpec:
    """MAR spec for ONE target column.

    ``drivers`` and ``coefficients`` are per column. This is the structural fix
    for the ``np.repeat`` broadcast at ``missing_data_generator.py:467``, which
    forced a single row-level propensity onto every column.
    """
    drivers: List[str]
    coefficients: Dict[str, Coefficient] = field(default_factory=dict)
    rate: Optional[float] = None          # per-column target rate override

    def coefficient_for(self, driver: str) -> Optional[Coefficient]:
        return self.coefficients.get(driver)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drivers": list(self.drivers),
            "coefficients": {k: (dict(v) if isinstance(v, Mapping) else float(v))
                             for k, v in self.coefficients.items()},
            "rate": self.rate,
        }


@dataclass
class MNARColumnSpec:
    """MNAR spec for ONE column."""
    mode: Optional[str] = None                 # None -> mechanism default for the column's type
    coefficient: Optional[float] = None        # slope on the standardized own-value
    groups: Optional[Dict[str, List[Any]]] = None       # semantic_groups: name -> levels
    group_offsets: Optional[Dict[str, float]] = None    # semantic_groups: name -> logit offset
    level_order: Optional[List[Any]] = None    # ordinal: explicit level ordering
    rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "coefficient": self.coefficient,
            "groups": self.groups,
            "group_offsets": self.group_offsets,
            "level_order": self.level_order,
            "rate": self.rate,
        }


@dataclass
class MissingnessSpec:
    """Fully resolved instruction for generating one mask."""

    dataset: str
    mechanism: str
    rate: float
    seed: int

    schema: DatasetSchema

    # columns that must never be masked: identifier + declared + every MAR driver
    always_observed: List[str] = field(default_factory=list)

    # engineering knobs
    tolerance: float = 0.01
    min_missing_per_col: int = 1
    rate_calibration: str = "per_column"

    #: "rewrite" (the T1.5 engine) or "legacy_R0" (verbatim port, see
    #: :mod:`missingness.legacy`). Only the ``record_index_ID`` profile sets the
    #: latter, and it exists solely to regenerate R0's masks bit-for-bit.
    implementation: str = "rewrite"
    #: Column typing used by the legacy engine when it must match R0 exactly.
    legacy_column_types: Optional[Dict[str, List[str]]] = None

    #: Row-order policy: ``as_is`` (default, no behavior change) or ``shuffle``.
    #:
    #: T1.5 measurement: the correlation between per-row missing rate and row
    #: index is a joint property of (mechanism, driver, *row order of the source
    #: table*), not of the simulator alone. Three of the six R0 derived tables
    #: arrive sorted — AutoMPG by ``model_year`` (r = +0.997 with row index),
    #: MIMIC by something tracking ``ALARM`` (+0.598) / ``ABP`` (-0.442),
    #: Concrete by something tracking ``ConcreteCS`` (-0.311); eICU, NHANES and
    #: ComCri are effectively unordered (max |r| ~ 0.06). A perfectly defensible
    #: clinical MAR driver that happens to be the sort key will therefore still
    #: produce a row-index correlation.
    #:
    #: ``shuffle`` permutes the rows before masking and records the permutation,
    #: which makes row position carry zero information by construction. It is
    #: OFF by default because reordering the shipped derived tables is a data-layer
    #: decision for the first author (T1.6), not a simulator decision.
    row_order_mode: str = "as_is"
    row_order_seed: Optional[int] = None

    # MAR
    mar_mode: str = "per_column"
    mar_default: Optional[MARColumnSpec] = None
    mar_columns: Dict[str, MARColumnSpec] = field(default_factory=dict)
    mar_coefficient_default: str = "auto"      # "auto" | "config"
    mar_auto_strength: float = 1.5
    mar_auto_min_abs: float = 0.2
    mar_logistic_scale: float = 1.0            # legacy mode only

    # MNAR
    mnar_continuous_mode: str = "logit"
    mnar_categorical_mode: str = "ordinal"
    mnar_strength: float = 1.5
    mnar_min_abs: float = 0.2
    mnar_columns: Dict[str, MNARColumnSpec] = field(default_factory=dict)
    mnar_quantile_steps: Dict[str, float] = field(default_factory=dict)

    # provenance
    profile: str = "unnamed"
    profile_status: str = "unspecified"
    config_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    #: One sentence of clinical or domain justification for this mechanism,
    #: written to be printable in the supplementary material. Required by P2
    #: T2.2(a): the point is that the missingness specification becomes an
    #: auditable, publishable artifact in its own right, which is the same
    #: property the paper claims for the imputer. A mechanism nobody can justify
    #: in words is exactly what reviewer point R1-4 objected to.
    rationale: Optional[str] = None

    # ------------------------------------------------------------------
    def validate(self) -> "MissingnessSpec":
        if self.implementation not in IMPLEMENTATIONS:
            raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {self.implementation!r}")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}, got {self.mechanism!r}")
        if not (0.0 < self.rate < 1.0):
            raise ValueError(f"rate must be in (0,1), got {self.rate!r}")
        if self.mar_mode not in MAR_MODES:
            raise ValueError(f"mar.mode must be one of {MAR_MODES}, got {self.mar_mode!r}")
        if self.mnar_continuous_mode not in MNAR_CONT_MODES:
            raise ValueError(f"mnar.continuous_mode must be one of {MNAR_CONT_MODES}")
        if self.mnar_categorical_mode not in MNAR_CAT_MODES:
            raise ValueError(f"mnar.categorical_mode must be one of {MNAR_CAT_MODES}")
        if self.rate_calibration not in RATE_CALIBRATION:
            raise ValueError(f"rate_calibration must be one of {RATE_CALIBRATION}")
        if self.row_order_mode not in ROW_ORDER_MODES:
            raise ValueError(f"row_order.mode must be one of {ROW_ORDER_MODES}")

        known = set(self.schema.columns)
        unknown = [c for c in self.always_observed if c not in known]
        if unknown:
            raise ValueError(f"always_observed refers to unknown columns: {unknown}")

        if self.mechanism == "MAR":
            if self.mar_default is None and not self.mar_columns:
                raise ValueError(
                    "MAR requires either a table-level `default.drivers` or per-column "
                    "`columns.<name>.drivers` in configs/missingness.yaml. "
                    "R0 fell back to `col_types.continuous[:2]` "
                    "(missing_data_generator.py:693); silent fallbacks are not allowed here."
                )
            for cname, cs in list(self.mar_columns.items()) + (
                [("<default>", self.mar_default)] if self.mar_default else []
            ):
                bad = [d for d in cs.drivers if d not in known]
                if bad:
                    raise ValueError(f"MAR spec for {cname!r} names unknown drivers: {bad}")
                if not cs.drivers:
                    raise ValueError(f"MAR spec for {cname!r} has no drivers")
        return self

    # ------------------------------------------------------------------
    def driver_union(self) -> List[str]:
        """Every column used as a MAR driver anywhere in this spec."""
        if self.mechanism != "MAR":
            return []
        out: List[str] = []
        specs = list(self.mar_columns.values())
        if self.mar_default is not None:
            specs.append(self.mar_default)
        for cs in specs:
            for d in cs.drivers:
                if d not in out:
                    out.append(d)
        return out

    def observed_columns(self) -> List[str]:
        """Columns kept fully observed under this spec, in table order.

        Includes the identifier, anything the config declared ``always_observed``,
        and — critically — every MAR driver. Keeping drivers observed is the
        strict-MAR guarantee R0 implemented at
        ``missing_data_generator.py:638-654``; it is preserved verbatim, and here
        it is additionally applied under MCAR and MNAR so that a dataset's driver
        set is the same set of columns under all three mechanisms and the masks
        stay comparable.
        """
        keep = set(self.always_observed) | set(self.driver_union())
        if self.schema.identifier_column:
            keep.add(self.schema.identifier_column)
        for c in self.schema.columns.values():
            if c.is_identifier:
                keep.add(c.name)
        return [c for c in self.schema.order() if c in keep]

    def target_columns(self) -> List[str]:
        obs = set(self.observed_columns())
        return [c for c in self.schema.order() if c not in obs]

    def mar_for(self, column: str) -> MARColumnSpec:
        cs = self.mar_columns.get(column)
        if cs is not None:
            return cs
        if self.mar_default is None:  # pragma: no cover - blocked by validate()
            raise KeyError(f"no MAR spec for column {column!r} and no table default")
        return self.mar_default

    def mnar_for(self, column: str) -> MNARColumnSpec:
        cs = self.mnar_columns.get(column) or MNARColumnSpec()
        if cs.mode is not None:
            return cs
        meta = self.schema.columns[column]
        default_mode = self.mnar_categorical_mode if meta.is_categorical else self.mnar_continuous_mode
        return MNARColumnSpec(
            mode=default_mode,
            coefficient=cs.coefficient,
            groups=cs.groups,
            group_offsets=cs.group_offsets,
            level_order=cs.level_order or meta.levels,
            rate=cs.rate,
        )

    def rate_for(self, column: str) -> float:
        if self.mechanism == "MAR":
            cs = self.mar_columns.get(column)
            if cs is not None and cs.rate is not None:
                return float(cs.rate)
        elif self.mechanism == "MNAR":
            cs = self.mnar_columns.get(column)
            if cs is not None and cs.rate is not None:
                return float(cs.rate)
        return float(self.rate)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dataset": self.dataset,
            "mechanism": self.mechanism,
            "target_rate": float(self.rate),
            "seed": int(self.seed),
            "profile": self.profile,
            "profile_status": self.profile_status,
            "rationale": self.rationale,
            "config_path": self.config_path,
            "tolerance": float(self.tolerance),
            "min_missing_per_col": int(self.min_missing_per_col),
            "rate_calibration": self.rate_calibration,
            "implementation": self.implementation,
            "row_order": {"mode": self.row_order_mode, "seed": self.row_order_seed},
            "always_observed_declared": list(self.always_observed),
            "observed_columns": self.observed_columns(),
            "target_columns": self.target_columns(),
            "notes": list(self.notes),
        }
        if self.mechanism == "MAR":
            d["mar"] = {
                "mode": self.mar_mode,
                "coefficient_default": self.mar_coefficient_default,
                "auto_strength": self.mar_auto_strength,
                "auto_min_abs": self.mar_auto_min_abs,
                "logistic_scale": self.mar_logistic_scale,
                "driver_union": self.driver_union(),
                "default": self.mar_default.to_dict() if self.mar_default else None,
                "columns": {k: v.to_dict() for k, v in self.mar_columns.items()},
            }
        if self.mechanism == "MNAR":
            d["mnar"] = {
                "continuous_mode": self.mnar_continuous_mode,
                "categorical_mode": self.mnar_categorical_mode,
                "strength": self.mnar_strength,
                "min_abs": self.mnar_min_abs,
                "quantile_steps": dict(self.mnar_quantile_steps),
                "columns": {k: v.to_dict() for k, v in self.mnar_columns.items()},
            }
        return d


# ---------------------------------------------------------------------------
# YAML resolution
# ---------------------------------------------------------------------------

def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(copy.deepcopy(base))
    for k, v in (over or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or DEFAULT_MISSINGNESS_CONFIG)
    cfg = yaml.safe_load(p.read_text())
    cfg["_path"] = str(p)
    return cfg


def _mar_column_spec(raw: Mapping[str, Any]) -> MARColumnSpec:
    return MARColumnSpec(
        drivers=list(raw.get("drivers", []) or []),
        coefficients=dict(raw.get("coefficients", {}) or {}),
        rate=raw.get("rate"),
    )


def _mnar_column_spec(raw: Mapping[str, Any]) -> MNARColumnSpec:
    return MNARColumnSpec(
        mode=raw.get("mode"),
        coefficient=raw.get("coefficient"),
        groups=dict(raw["groups"]) if raw.get("groups") else None,
        group_offsets=dict(raw["group_offsets"]) if raw.get("group_offsets") else None,
        level_order=list(raw["level_order"]) if raw.get("level_order") else None,
        rate=raw.get("rate"),
    )


def resolve(
    dataset: str,
    mechanism: str,
    rate: float,
    *,
    profile: str,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
    schema: Optional[DatasetSchema] = None,
    datasets_config_path: Optional[Path] = None,
    seed: Optional[int] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> MissingnessSpec:
    """Resolve ``configs/missingness.yaml`` into a :class:`MissingnessSpec`."""
    cfg = dict(config) if config is not None else load_config(config_path)
    mechanism = mechanism.strip().upper()

    profiles = cfg.get("profiles", {}) or {}
    if profile not in profiles:
        raise KeyError(f"profile {profile!r} not in {sorted(profiles)}")
    prof = profiles[profile] or {}

    merged = _deep_merge(cfg.get("defaults", {}) or {}, prof.get("defaults", {}) or {})
    ds_block = ((prof.get("datasets", {}) or {}).get(dataset, {}) or {})
    merged = _deep_merge(merged, ds_block.get("common", {}) or {})
    merged = _deep_merge(merged, ds_block.get(mechanism, {}) or {})
    merged = _deep_merge(merged, overrides or {})

    schema = schema or dataset_schema(dataset, path=datasets_config_path)

    mar_raw = merged.get("mar", {}) or {}
    mnar_raw = merged.get("mnar", {}) or {}

    mar_default = None
    if mar_raw.get("default"):
        mar_default = _mar_column_spec(mar_raw["default"])
    mar_columns = {k: _mar_column_spec(v) for k, v in (mar_raw.get("columns", {}) or {}).items()}
    mnar_columns = {k: _mnar_column_spec(v) for k, v in (mnar_raw.get("columns", {}) or {}).items()}

    always_obs = list(merged.get("always_observed", []) or [])
    notes = list(merged.get("notes", []) or [])
    # Forward fix (P5R-B SS5-B2, 2026-08-28): any FUTURE mask generation
    # keeps the dataset's declared downstream target fully observed, even
    # where a mechanism stanza still lists it (the stanza stays as the
    # historical record of the executed configuration; frozen mask files
    # are untouched). R0 reproduction profiles (implementation: legacy_R0)
    # are exempt -- they exist to reproduce the historical behavior
    # exactly. The exclusion is recorded in the spec's notes, which every
    # generated mask embeds in its meta.
    _tgt = getattr(schema, "downstream_target", None)
    if (_tgt and str(merged.get("implementation", "rewrite")) != "legacy_R0"
            and _tgt not in always_obs):
        always_obs.append(_tgt)
        notes.append(f"downstream target {_tgt!r} auto-excluded from masking "
                     f"at spec resolution (P5R-B SS5-B2 forward fix)")

    spec = MissingnessSpec(
        dataset=dataset,
        mechanism=mechanism,
        rate=float(rate),
        seed=int(seed if seed is not None else merged.get("seed", 2025)),
        schema=schema,
        always_observed=always_obs,
        tolerance=float(merged.get("tolerance", 0.01)),
        min_missing_per_col=int(merged.get("min_missing_per_col", 1)),
        rate_calibration=str(merged.get("rate_calibration", "per_column")),
        implementation=str(merged.get("implementation", "rewrite")),
        legacy_column_types=(dict(merged["legacy_column_types"])
                             if merged.get("legacy_column_types") else None),
        row_order_mode=str((merged.get("row_order") or {}).get("mode", "as_is")),
        row_order_seed=(merged.get("row_order") or {}).get("seed"),
        mar_mode=str(mar_raw.get("mode", "per_column")),
        mar_default=mar_default,
        mar_columns=mar_columns,
        mar_coefficient_default=str(mar_raw.get("coefficient_default", "auto")),
        mar_auto_strength=float(mar_raw.get("auto_strength", 1.5)),
        mar_auto_min_abs=float(mar_raw.get("auto_min_abs", 0.2)),
        mar_logistic_scale=float(mar_raw.get("logistic_scale", 1.0)),
        mnar_continuous_mode=str(mnar_raw.get("continuous_mode", "logit")),
        mnar_categorical_mode=str(mnar_raw.get("categorical_mode", "ordinal")),
        mnar_strength=float(mnar_raw.get("strength", 1.5)),
        mnar_min_abs=float(mnar_raw.get("min_abs", 0.2)),
        mnar_columns=mnar_columns,
        mnar_quantile_steps=dict(mnar_raw.get("quantile_steps", {}) or {}),
        profile=profile,
        profile_status=str(prof.get("status", "unspecified")),
        # From the *merged* block, not from the top-level config: a rationale is
        # written per (dataset, mechanism) and may also be inherited from that
        # dataset's `common`. Reading it off `cfg` silently returned None for
        # every dataset, which the T2.2(a) validation caught.
        rationale=merged.get("rationale"),
        config_path=cfg.get("_path"),
        notes=notes,
    )
    return spec.validate()
