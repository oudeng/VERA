"""Census of every code path that reads a pre-mask ground-truth table.

Rules: docs/T53_input_correction_rules.md SS5 (the first author's condition 4).
The T5.3 defect was one consumer of the ground-truth table doing something
the others do not. This enumerates every such read, requires each to carry a
DECLARED classification, and refuses any that is unclassified or of the
defect shape.

Classes (declared in docs/ground_truth_consumers.json):

  L1  read only to build the masked table -- complete[feats].mask(mask_df)
  L2  read to obtain the withheld true values for SCORING (RMSE / NRMSE /
      the ablation matrix's error target). Legitimate: that is what a
      benchmark is. Enumerated, not waved through.
  L3  read to construct an injected leakage proxy. Legitimate: that is the
      leakage experiment's premise.
  L4  read for a design pre-check applied symmetrically to every object.
      Reported separately, with whether its prospective specification says so.
  L5  read for column names, dtypes, or the never-masked outcome label.
  P   produces or configures the ground-truth table itself.
  SELF the lineage audit's own machinery.
  X   read to compute an audit object, comparator or reported statistic that
      competes with an object computed from the masked or completed table.
      THIS is the defect shape. Any X fails the census.

    env PYTHONHASHSEED=2025 python experiments/ground_truth_consumer_census.py
    env PYTHONHASHSEED=2025 python experiments/ground_truth_consumer_census.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))

REGISTRY = CODE_ROOT / "docs" / "ground_truth_consumers.json"
OUT = CODE_ROOT / "results" / "T6_lineage" / "consumer_census.json"

#: a read of a pre-mask ground-truth table, in any of its spellings
READ_PAT = re.compile(
    r"""(derived_shuffled|data/derived)\b[^\n]*_complete\.csv"""
    r"""|_complete\.csv""")

#: A path read is not the only way withheld values enter a computation: the
#: frame is passed on, and the line that finally indexes it with the masked
#: rows is where the withheld value is actually used. An adversarial sweep of
#: this tree flagged exactly those lines (the ablation matrix's error target)
#: and they were not visible to READ_PAT alone. They are now scanned for and
#: classified like any other consumer.
USE_PAT = re.compile(
    r"""^\s*truth\s*=.*\bcomplete\w*\b"""
    r"""|=\s*complete\w*\[[^\]]+\]\.to_numpy\(\)\[mi(?:ss|_)"""
    r"""|complete\w*\[f\][^\n]*\[miss\]""")

#: directories whose reads are not part of the scientific pipeline
SKIP_DIRS = {"tests", "reporting/out", "results", "data"}
#: the audit's own machinery necessarily reads the ground-truth table in
#: order to test it; declared here and in the registry, not hidden
SELF = {"experiments/tap_lineage_audit.py",
        "experiments/ground_truth_consumer_census.py"}


def scan() -> list:
    """Every (file, line) in the pipeline that names a ground-truth table."""
    hits = []
    for py in sorted(CODE_ROOT.rglob("*.py")):
        rel = py.relative_to(CODE_ROOT).as_posix()
        if any(rel.startswith(d + "/") or rel == d for d in SKIP_DIRS):
            continue
        for i, line in enumerate(py.read_text(errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if READ_PAT.search(line):
                hits.append({"file": rel, "line": i, "text": stripped[:160],
                             "kind": "path_read"})
            elif USE_PAT.search(line):
                hits.append({"file": rel, "line": i, "text": stripped[:160],
                             "kind": "withheld_value_use"})
    return hits


def run(registry_path: Path = REGISTRY, out_path: Path = OUT) -> dict:
    reg = json.loads(registry_path.read_text())
    declared = {(e["file"], e["line"]): e for e in reg["consumers"]}
    hits = scan()

    classified, unregistered, defect_shape = [], [], []
    for h in hits:
        key = (h["file"], h["line"])
        e = declared.get(key)
        if e is None:
            # a line may have moved; match on file + exact text as a fallback,
            # and say so, rather than silently accepting the drift
            same_file = [d for k, d in declared.items()
                         if k[0] == h["file"] and d.get("text_anchor")
                         and d["text_anchor"] in h["text"]]
            if len(same_file) == 1:
                e = dict(same_file[0])
                e["line_drifted_from"] = e["line"]
                e["line"] = h["line"]
            else:
                unregistered.append(h)
                continue
        rec = {**h, "classification": e["classification"],
               "why": e["why"], "reported_artifact": e.get("reported_artifact"),
               "self_referential": h["file"] in SELF}
        if e.get("line_drifted_from"):
            rec["line_drifted_from"] = e["line_drifted_from"]
        classified.append(rec)
        if e["classification"] == "X":
            defect_shape.append(rec)

    stale = [dict(d) for k, d in declared.items()
             if not any(c["file"] == k[0]
                        and (c["line"] == k[1]
                             or c.get("line_drifted_from") == k[1])
                        for c in classified)]

    census = {
        "rules": "docs/T53_input_correction_rules.md SS5",
        "registry": (str(registry_path.relative_to(CODE_ROOT))
                     if CODE_ROOT in registry_path.parents
                     else str(registry_path)),
        "n_reads_found": len(hits),
        "n_classified": len(classified),
        "counts": {c: sum(1 for x in classified if x["classification"] == c)
                   for c in ("P", "L1", "L2", "L3", "L4", "L5", "SELF", "X")},
        "unregistered": unregistered,
        "defect_shape": defect_shape,
        "stale_registry_entries": stale,
        "consumers": classified,
        "pass": bool(not unregistered and not defect_shape and not stale),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(census, indent=1))
    return census


def _selftest() -> int:
    ok = True

    def check(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    import tempfile
    reg = json.loads(REGISTRY.read_text())
    check(all({"file", "line", "classification", "why"} <= set(e)
              for e in reg["consumers"]),
          "every registry entry carries file, line, classification and why")
    check(all(e["classification"] in reg["classes"]
              for e in reg["consumers"]),
          "every classification is one of the declared classes")

    # an unregistered read must fail the census, or the census proves nothing
    with tempfile.TemporaryDirectory() as td:
        fake = json.loads(REGISTRY.read_text())
        dropped = fake["consumers"].pop()
        p = Path(td) / "reg.json"
        p.write_text(json.dumps(fake))
        c = run(p, Path(td) / "out.json")
        check(not c["pass"] and c["unregistered"],
              f"dropping one entry ({dropped['file']}:{dropped['line']}) "
              f"makes the census fail")

    c = run()
    check(c["pass"], f"live census passes (unregistered={c['unregistered']}, "
                     f"X={c['defect_shape']}, stale={c['stale_registry_entries']})")
    check(c["counts"]["X"] == 0, "zero consumers of the defect shape remain")
    check(c["n_reads_found"] == c["n_classified"],
          "every read found is classified")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return _selftest()
    c = run()
    print(f"reads found: {c['n_reads_found']}   classified: {c['n_classified']}")
    print(f"counts: {c['counts']}")
    for cls in ("X", "L4", "L3", "L2", "L1", "L5", "P", "SELF"):
        rows = [x for x in c["consumers"] if x["classification"] == cls]
        if not rows:
            continue
        print(f"\n--- {cls} ({len(rows)}) ---")
        for r in rows:
            print(f"  {r['file']}:{r['line']}  {r['text'][:88]}")
    if c["unregistered"]:
        print(f"\nUNREGISTERED ({len(c['unregistered'])}):")
        for r in c["unregistered"]:
            print(f"  {r['file']}:{r['line']}  {r['text'][:88]}")
    if c["stale_registry_entries"]:
        print(f"\nSTALE REGISTRY ENTRIES ({len(c['stale_registry_entries'])}):")
        for r in c["stale_registry_entries"]:
            print(f"  {r['file']}:{r['line']}")
    print(f"\nCENSUS: {'PASS' if c['pass'] else 'FAIL'}")
    print(f"wrote {OUT}")
    return 0 if c["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
