"""Grid-scale sentence for Methods (P5 SS1.2): every numeric component is
derived from run_grid.cells() -- the same enumeration the workers ran --
and asserted against the on-disk artifact census before a word is emitted.

    PYTHONHASHSEED=2025 python reporting/grid_scale.py [--selftest]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
for p in (str(CODE_ROOT), str(CODE_ROOT / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import runconfig                                    # noqa: E402
from reporting.latex import write_tex                           # noqa: E402

GRID = CODE_ROOT / "results" / "P2_main_grid"


def enumerate_grid():
    import run_grid as rg
    ds_cfg = rg._cfg("datasets")["datasets"]
    placement = rg._cfg("scheduling")["method_placement"]
    methods = list(placement)                       # all nine, both queues
    todo = rg.cells(list(ds_cfg), methods, True)
    synth = [c for c in todo if c[2] != "REAL_PATTERN"]
    rp = [c for c in todo if c[2] == "REAL_PATTERN"]
    info = {
        "n_total": len(todo),
        "n_methods": len(methods),
        "datasets": list(ds_cfg),
        "n_datasets": len(ds_cfg),
        "mechs": sorted({c[2] for c in synth}),
        "rates": sorted({c[3] for c in synth}),
        "n_seeds": len({c[4] for c in todo}),
        "slim": dict(rg.SLIM),
        "n_rp": len(rp),
        "rp_dataset": sorted({c[1] for c in rp}),
    }
    return info


def build(out_path: Path) -> Path:
    info = enumerate_grid()
    n_disk = len(list(GRID.glob("*/metrics_summary.json")))
    assert n_disk == info["n_total"], \
        f"enumeration {info['n_total']} != on-disk census {n_disk}"
    assert info["mechs"] == ["MAR", "MCAR", "MNAR"], info["mechs"]
    slim_ds = list(info["slim"])[0]
    slim_rate = info["slim"][slim_ds]["rates"][0]
    assert info["rp_dataset"] == [slim_ds], info["rp_dataset"]
    rates_pct = sorted(int(r * 100) for r in info["rates"])
    ds_names = ", ".join(info["datasets"][:4])
    n_uci = info["n_datasets"] - 4
    body = (
        f"The benchmark grid comprises {info['n_total']:,} runs: "                       # % src: cells() == disk census
        f"{info['n_methods']} imputers across {info['n_datasets']} tables "
        f"({ds_names}, and {n_uci} standard UCI tables), three synthetic "
        f"mechanisms (MCAR/MAR/MNAR) at "
        f"{rates_pct[0]}--{rates_pct[-1]}\\% rates "
        f"({slim_ds} at {int(slim_rate*100)}\\% only), "
        f"{info['n_seeds']} seeds per cell, plus a real-missingness "
        f"condition on {slim_ds} ({info['n_rp']} runs).\n"
        f"% src: run_grid.cells() enumeration, asserted == on-disk census "
        f"({n_disk})\n")
    return write_tex(out_path, body, provenance={
        "generator": "reporting/grid_scale.py",
        "input": f"run_grid.cells() + {GRID}",
        "code_SNI commit": runconfig.git_commit()})


def _selftest() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    info = enumerate_grid()
    check(info["n_total"] == 2565, "enumeration totals 2565 (9 methods)")
    check(info["n_methods"] == 9 and info["n_datasets"] == 7,
          "9 methods, 7 datasets")
    check(info["n_seeds"] == 5 and info["n_rp"] == 45,
          "5 seeds; real-pattern block 45 (9 methods x 5 seeds)")
    check(info["slim"] == {"CDC2022": {"mechanisms": ["MCAR", "MAR"],
                                       "rates": [0.3]}},
          "SLIM clause matches the committed grid design")
    n_disk = len(list(GRID.glob("*/metrics_summary.json")))
    check(n_disk == info["n_total"], "on-disk census equals enumeration")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CODE_ROOT / "reporting" / "out"
                                         / "grid_scale.tex"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    out = build(Path(a.out))
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
