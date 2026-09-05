"""P5R-H SS7.3 (third review P1-3, extended route): full-coverage digest
assertion over every frozen mask file.

Scope: all 63 simulated masks (7 datasets x {MCAR,MAR,MNAR} x
{10,30,50}per) plus the 5 real-pattern masks -- 68 files. First run with
--record writes the manifest (results/T5_stats/mask_digest_manifest.json,
sha256 per file); every later run asserts byte-identity against it. Any
drift is red.

    env PYTHONHASHSEED=2025 python experiments/mask_digest.py --record
    env PYTHONHASHSEED=2025 python experiments/mask_digest.py
    env PYTHONHASHSEED=2025 python experiments/mask_digest.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOTS = [CODE_ROOT / "data" / "masks" / "clinical_v1_shuffled",
         CODE_ROOT / "data" / "masks" / "real_pattern"]
MANIFEST = CODE_ROOT / "results" / "T5_stats" / "mask_digest_manifest.json"
N_EXPECTED_SIM = 63


def collect(roots=None) -> dict:
    out = {}
    for root in (roots or ROOTS):
        for f in sorted(root.glob("*/*_mask.npy")):
            try:
                rel = str(f.relative_to(CODE_ROOT))
            except ValueError:
                rel = f.name
            out[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    got = collect()
    n_sim = sum(1 for k in got if "clinical_v1_shuffled" in k)
    if n_sim != N_EXPECTED_SIM:
        print(f"RED: expected {N_EXPECTED_SIM} simulated masks, found "
              f"{n_sim}", file=sys.stderr)
        return 1
    if a.record:
        if MANIFEST.exists():
            print("REFUSED: manifest exists; delete it explicitly only "
                  "under an adjudicated re-freeze.", file=sys.stderr)
            return 1
        MANIFEST.write_text(json.dumps(
            {"n_files": len(got), "sha256": got}, indent=1))
        print(f"[ok] recorded {len(got)} digests -> {MANIFEST}")
        return 0
    ref = json.loads(MANIFEST.read_text())["sha256"]
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    drift = sorted(k for k in set(ref) & set(got) if ref[k] != got[k])
    ok = not (missing or extra or drift)
    print(json.dumps({"n_checked": len(got), "missing": missing,
                      "extra": extra, "drifted": drift,
                      "verdict": "BYTE-IDENTICAL" if ok else "DRIFT"}))
    return 0 if ok else 1


def _selftest() -> int:
    import tempfile
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "x" / "ds"
        d.mkdir(parents=True)
        f = d / "A_MAR_30per_mask.npy"
        f.write_bytes(b"\x01\x02")
        got = collect([Path(td) / "x"])
        check(len(got) == 1 and next(iter(got.values()))
              == hashlib.sha256(b"\x01\x02").hexdigest(),
              "sha256 collected per file")
        f.write_bytes(b"\x01\x03")
        got2 = collect([Path(td) / "x"])
        check(got != got2, "byte change changes the digest")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
