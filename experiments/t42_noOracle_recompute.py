"""The leakage axis recomputed without the privileged error signal.

T6.1 addendum 2026-08-29d SS4, from the sixth review's fourth finding. The
leakage counts that carry the axis's sharpest same-host contrast were produced
by an ablation whose error signal reads the values withheld from the imputer.
The text downgrade does not wait for this; this is the measurement that would
let the comparison be made on equal terms instead.

It re-runs the campaign's cases with `both_signals=True`, so each freshly
trained host emits BOTH readouts -- oracle and no-oracle -- and the pair is
within-host. Comparing a fresh no-oracle matrix against the ARCHIVED oracle
one would confound the oracle with the training draw; that is the defect the
recovery section of the same addendum exists to remove, and it is not
reintroduced here.

Every write goes under results/T6_symmetry/leakage_noOracle/. The archived
campaign is not touched.

    env PYTHONHASHSEED=2025 python experiments/t42_noOracle_recompute.py \
        --shard 0/1 [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "experiments"))

OUT = CODE_ROOT / "results" / "T6_symmetry" / "leakage_noOracle"


def _contained(p: Path) -> Path:
    root = (CODE_ROOT / "results" / "T6_symmetry").resolve()
    rp = Path(p).resolve()
    if root not in rp.parents and rp != root:
        raise RuntimeError(f"refusing to write outside {root}: {rp}")
    return rp


def cases() -> list:
    from t42_leakage import plan, conf_plan
    seen, out = set(), []
    for r in list(plan()) + list(conf_plan()):
        if r[0] in seen:
            continue
        seen.add(r[0])
        out.append(r)
    return out


def run(shard: str = "0/1", limit: int | None = None, dry: bool = False) -> int:
    from t42_leakage import run_case
    i, nw = (int(x) for x in shard.split("/"))
    allc = cases()
    mine = [c for k, c in enumerate(allc) if k % nw == i]
    if limit:
        mine = mine[:limit]
    _contained(OUT).mkdir(parents=True, exist_ok=True)
    print(f"[plan] {len(allc)} cases total, {len(mine)} on shard {shard}")
    if dry:
        for c in mine[:8]:
            print("   ", c[0])
        return 0
    done, t_start = 0, time.time()
    for tag, ds, seed, kind, cls, rho in mine:
        t0 = time.time()
        w = run_case(tag, ds, seed, kind, cls, rho,
                     out_root=_contained(OUT), both_signals=True)
        done += 1
        el = time.time() - t_start
        print(f"[{done}/{len(mine)}] {tag} "
              f"{'cached' if w is None else f'{time.time()-t0:.0f}s'} "
              f"| elapsed {el/3600:.2f} h", flush=True)
    (_contained(OUT / "recompute_progress.json")).write_text(json.dumps(
        {"shard": shard, "n_cases_total": len(allc), "n_on_shard": len(mine),
         "n_done": done, "elapsed_hours": round((time.time() - t_start) / 3600, 2)},
        indent=1))
    return 0


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    c = cases()
    chk(len(c) > 100, f"the case list is the campaign's ({len(c)} cases)")
    chk(len({x[0] for x in c}) == len(c), "no duplicate tags")

    import inspect
    from t42_leakage import run_case
    sig = inspect.signature(run_case).parameters
    chk("both_signals" in sig and "out_root" in sig,
        "run_case takes both_signals and out_root")

    # the guard must refuse the archived campaign's directory
    try:
        _contained(CODE_ROOT / "results" / "T4_leakage" / "runs")
        chk(False, "the write guard refuses the archived campaign directory")
    except RuntimeError:
        chk(True, "the write guard refuses the archived campaign directory")

    src = (CODE_ROOT / "experiments" / "t42_leakage.py").read_text()
    chk("truth_no = pd.to_numeric(X_final[f]" in src,
        "the no-oracle error target is the host's own completed table")
    chk(src.count("yhp = _predict(Zp)") == 1,
        "both variants score the SAME permuted predictions (one forward pass)")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    return run(a.shard, a.limit, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
