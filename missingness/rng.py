"""Independent, name-keyed random streams for mask generation (P0 finding B45).

Why this module exists
----------------------
R0's ``missing_data_generator.generate_missing_dataset`` created exactly one
generator (``missing_data_generator.py:617``) and threaded it through every
stage: the MNAR categorical "high-missing" draw
(``missing_data_generator.py:541``), the whole-table Bernoulli sample
(``missing_data_generator.py:718``), the min-per-column tie-break
(``missing_data_generator.py:316``) and the rate calibration
(``missing_data_generator.py:383``, ``:386``, ``:402``, ``:405``).

Because a *single* stream is consumed in column order, the number of random
values drawn before column *j* depends on how many categorical columns precede
it and on how many of them needed calibration. Adding one categorical column to
a dataset therefore re-rolls every subsequent column's mask. That is finding B45,
and it makes ablations of the form "same mask, one column added/removed"
impossible to interpret.

The fix
-------
Every (purpose, column) pair gets its own :class:`numpy.random.Generator`, seeded
from a :class:`numpy.random.SeedSequence` whose entropy is
``[root_seed, blake2b(namespace, purpose, *parts)]``.

Note that ``SeedSequence(seed).spawn(k)`` — the obvious alternative — is *not*
used, because ``spawn`` is **positional**: the i-th spawned child depends on i,
so it reintroduces exactly the ordering coupling we are removing. Keying on the
column *name* instead makes a stream a pure function of what it is for, so:

* adding, removing or reordering columns leaves every other column's draws
  bit-identical;
* the propensity draw, the Bernoulli draw, the min-per-column tie-break and the
  rate calibration of one column cannot perturb each other;
* the whole assignment is reproducible from the integers recorded in
  ``meta.json`` (see :meth:`StreamRegistry.describe`).

``hashlib.blake2b`` rather than :func:`hash` because Python's string hash is
salted per process (``PYTHONHASHSEED``) and would not be reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

#: Stream purposes. Each is an independent family of generators.
PURPOSES = (
    "propensity",   # drawing per-column coefficients / category offsets
    "bernoulli",    # the actual mask draw
    "min_per_col",  # tie-break noise when forcing a minimum count per column
    "calibration",  # per-column rate calibration
    "table",        # whole-table operations (legacy table-level calibration)
)

_SEP = "\x1f"


def stable_key(*parts: Any) -> int:
    """A reproducible 64-bit integer derived from arbitrary string-able parts.

    Deliberately not :func:`hash` (salted per process) and not :func:`zlib.crc32`
    (32 bits, and structured collisions on short ASCII names are easy to find).
    """
    payload = _SEP.join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


class StreamRegistry:
    """Hands out independent generators keyed by ``(purpose, *parts)``.

    Parameters
    ----------
    seed
        Root seed. Two registries with the same seed and namespace hand out
        bit-identical streams.
    namespace
        Extra discriminators mixed into every key — in practice
        ``(dataset, mechanism, rate_tag)`` — so that the MCAR and MAR masks of
        the same dataset at the same rate do not share Bernoulli draws.
    """

    def __init__(self, seed: int, *, namespace: Sequence[Any] = ()) -> None:
        self.seed = int(seed)
        self.namespace: Tuple[Any, ...] = tuple(namespace)
        self._cache: Dict[Tuple[Any, ...], np.random.Generator] = {}
        self._entropy: Dict[Tuple[Any, ...], int] = {}

    def stream(self, purpose: str, *parts: Any) -> np.random.Generator:
        if purpose not in PURPOSES:
            raise ValueError(f"unknown stream purpose {purpose!r}; expected one of {PURPOSES}")
        key = (purpose,) + tuple(parts)
        gen = self._cache.get(key)
        if gen is None:
            ent = stable_key(*self.namespace, purpose, *parts)
            gen = np.random.default_rng(np.random.SeedSequence(entropy=[self.seed, ent]))
            self._cache[key] = gen
            self._entropy[key] = ent
        return gen

    def describe(self) -> Dict[str, Any]:
        """Everything needed to re-derive every stream, for ``meta.json``."""
        streams: List[Dict[str, Any]] = [
            {"purpose": k[0], "key": list(k[1:]), "entropy": int(v)}
            for k, v in sorted(self._entropy.items(), key=lambda kv: (kv[0][0], tuple(map(str, kv[0][1:]))))
        ]
        return {
            "root_seed": self.seed,
            "namespace": [str(x) for x in self.namespace],
            "derivation": (
                "np.random.default_rng(SeedSequence(entropy=[root_seed, "
                "blake2b_64(namespace + (purpose,) + key)]))"
            ),
            "n_streams": len(streams),
            "streams": streams,
        }


def independence_probe(seed: int, columns: Iterable[str], n: int = 8) -> Dict[str, List[float]]:
    """Small helper used by the tests: first ``n`` draws of each column's stream.

    Comparing two probes taken over different column *sets* is the direct check
    that B45 is fixed — the shared column names must produce identical draws.
    """
    reg = StreamRegistry(seed, namespace=("probe",))
    return {c: reg.stream("bernoulli", c).random(n).tolist() for c in columns}
