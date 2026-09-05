"""Post-hoc gate-bound scan (P5R-C SS3.4 / review Q10): softplus of the
saved theta_lambda of EVERY archived trained host must not exceed ln 2 --
the optimization invariant the ESM derives (theta initialized at zero,
gate enters the objective only through the non-negative prior term). The
Those host models were trained before the runtime assertion existed; this
scan is the retroactive verification the paper cites.

    env PYTHONHASHSEED=2025 python experiments/lambda_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
FAITH = CODE_ROOT / "results" / "T3_faithfulness"
OUT = CODE_ROOT / "results" / "T5_stats" / "lambda_check.json"


def main() -> int:
    import torch
    import torch.nn.functional as F
    recs = []
    gmax, gargmax = -1.0, None
    for pt in sorted(FAITH.glob("models_*.pt")):
        states = torch.load(pt, map_location="cpu", weights_only=True)
        for f, sd in states.items():
            if "theta_lambda" not in sd:
                continue
            lmax = float(F.softplus(sd["theta_lambda"]).max())
            recs.append({"file": pt.name, "target": f,
                         "lambda_max": round(lmax, 8)})
            if lmax > gmax:
                gmax, gargmax = lmax, f"{pt.name}/{f}"
    n_viol = sum(1 for r in recs if r["lambda_max"] > math.log(2.0) + 1e-6)
    wp = sorted(r["lambda_max"] for r in recs
                if not r["file"].startswith("models_NP_"))
    def q(v, f):
        return round(v[int(f * (len(v) - 1))], 6)
    out = {"n_models_scanned": len(recs),
           "withprior_lambda_quantiles": {
               "min": wp[0], "q25": q(wp, .25), "median": q(wp, .5),
               "q75": q(wp, .75), "max": wp[-1], "n": len(wp)},
           "n_host_archives": len(set(r["file"] for r in recs)),
           "global_max_lambda": round(gmax, 8),
           "ln2": round(math.log(2.0), 8),
           "margin_to_ln2": round(math.log(2.0) - gmax, 8),
           "argmax": gargmax,
           "n_violations": n_viol,
           "verdict": "INVARIANT HOLDS" if n_viol == 0 else "VIOLATED"}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if n_viol:
        print("LAMBDA INVARIANT VIOLATED -- stop and adjudicate",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
