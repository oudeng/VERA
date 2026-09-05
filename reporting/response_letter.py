"""The point-by-point response to the R0 reviewers, generated -- not typed.

P6. The letter quotes numbers that also appear in the manuscript, the ESM and
the evidence package. A letter whose numbers were typed by hand is a fourth
place for them to drift, and the one place nobody re-derives before
submission. So every figure here is pulled from the artifact that produced it
and formatted at render time; the prose carries `{{...}}` references, never
digits.

    PYTHONHASHSEED=2025 python reporting/response_letter.py
    PYTHONHASHSEED=2025 python reporting/response_letter.py --check <file>
    PYTHONHASHSEED=2025 python reporting/response_letter.py --selftest

--check re-renders and compares character for character, the same shape
change_summary.py and status_table.py use. It is what makes P6 SS5's
"letter numbers <-> frozen PDF" check a command rather than a proofread.

Page numbers are deliberately absent: the manuscript is not frozen, so the
letter cites \\label names and table/figure numbers, and the page column is
filled only after the freeze.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
#: Delivered into VERA_response/ beside the cover letter: the two go to
#: the editor together, so they live together.
OUT = ROOT / "VERA_response" / "RESPONSE_TO_REVIEWERS_R1.md"

#: Where a `{{key}}` gets its value. Each entry is (artifact, dotted path,
#: format). The artifact is resolved under code_SNI unless it starts with a
#: repository-relative directory that exists at ROOT.
#: The published repository, named once. The cover letter imports these
#: rather than keeping a second copy: two letters and a manuscript citing
#: three commits is how this went wrong once already.
#:
#: The COMMIT is not typed here. This file is itself published, so a literal
#: hash would be the tree stating its own identity -- and it would state the
#: PREVIOUS one, because the hash does not exist until the tree containing
#: this line has been committed. The build writes the release identity to
#: results/ (which is not published) after the tree is frozen; this reads it
#: back. Same rule as the README, which for the same reason cannot state its
#: own commit either.
REPO_RELEASE = CODE_ROOT / "results" / "public_release.json"


def _release() -> dict:
    if not REPO_RELEASE.exists():
        raise FileNotFoundError(
            f"{REPO_RELEASE} does not exist: the letters cite a published "
            f"release, and no release has been frozen. Run "
            f"experiments/build_public_repo.py first.")
    return json.loads(REPO_RELEASE.read_text())


REPO_URL = "https://github.com/oudeng/VERA"
REPO_TAG = _release()["tag"]
REPO_COMMIT = _release()["commit"]

T_FINAL = "results/T5_stats/t_final.json"
LAMBDA = "results/T5_stats/lambda_check.json"

VALUES = {
    # --- the fair same-host pair: the paper's central comparison ---
    "fair_T":        (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.T", "+.3f"),
    "fair_T_full":   (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.T", "+.6f"),
    "fair_lo":       (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.ci95_T.0", "+.3f"),
    "fair_hi":       (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.ci95_T.1", "+.3f"),
    "fair_p":        (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.p_exact", ".4f"),
    "fair_floor":    (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.floor", ".4f"),
    "fair_seeds":    (T_FINAL, "recovery.probe_vs_D_same_host_symmetric.n_seeds", "d"),
    # --- D versus TAP, the comparator the reviewer's item 2 asked for ---
    "dtap_T":        (T_FINAL, "recovery.D_vs_TAP.T", "+.3f"),
    "dtap_lo":       (T_FINAL, "recovery.D_vs_TAP.ci95_T.0", "+.3f"),
    "dtap_hi":       (T_FINAL, "recovery.D_vs_TAP.ci95_T.1", "+.3f"),
    "dtap_p":        (T_FINAL, "recovery.D_vs_TAP.p_exact", ".3f"),
    "dtap_neg":      (T_FINAL, "recovery.D_vs_TAP.seeds_negative", "s"),
    # --- behavioral faithfulness, 15 seeds per table ---
    "faith_mimic_T": (T_FINAL, "faithfulness.MIMIC.T", "+.3f"),
    "faith_mimic_p": (T_FINAL, "faithfulness.MIMIC.p_exact", ".3f"),
    "faith_mimic_holm": (T_FINAL, "faithfulness.MIMIC.p_holm", ".4f"),
    "faith_mimic_n": (T_FINAL, "faithfulness.MIMIC.n_seeds", "d"),
    "faith_eicu_T":  (T_FINAL, "faithfulness.eICU.T", "+.3f"),
    "faith_eicu_p":  (T_FINAL, "faithfulness.eICU.p_exact", ".4f"),
    # --- the no-prior control: the same axes without the association prior ---
    "np_mimic_T":    (T_FINAL, "noprior_faithfulness.MIMIC.T", "+.3f"),
    "np_mimic_p":    (T_FINAL, "noprior_faithfulness.MIMIC.p_exact", ".5f"),
    "np_eicu_T":     (T_FINAL, "noprior_faithfulness.eICU.T", "+.3f"),
    # --- leakage: the permutation null the reviewer asked for ---
    "null_detected": (T_FINAL, "leakage.probe_null_by_batch.original.detected", "d"),
    "null_n":        (T_FINAL, "leakage.probe_null_by_batch.original.n", "d"),
    #: p_geq_k_exact_binomial is deliberately NOT referenced. The fourth
    #: internal review ruled no fixed-alpha binomial test admissible on these
    #: counts -- the nominal level is itself estimated from a finite
    #: calibration sample -- and the terminology registry bans the rendered
    #: string. It caught this letter's first draft.
    # --- host bands under information symmetry ---
    "band_mimic":    (T_FINAL, "scoreboard_desc.host_band_mean_symmetric.MIMIC", ".3f"),
    "band_eicu":     (T_FINAL, "scoreboard_desc.host_band_mean_symmetric.eICU", ".3f"),
    # --- the gate that cannot rise ---
    "ln2":           (LAMBDA, "ln2", ".4f"),
    "lam_max":       (LAMBDA, "withprior_lambda_quantiles.max", ".3f"),
    "lam_min":       (LAMBDA, "withprior_lambda_quantiles.min", ".3f"),
    "lam_median":    (LAMBDA, "withprior_lambda_quantiles.median", ".3f"),
    "lam_models":    (LAMBDA, "n_models_scanned", "d"),
    "lam_viol":      (LAMBDA, "n_violations", "d"),
}


def _load(rel: str) -> dict:
    for base in (CODE_ROOT, ROOT):
        p = base / rel
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(rel)


def _dig(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def values() -> dict:
    """Every referenced number, read from the artifact that produced it."""
    cache, out = {}, {}
    for key, (rel, path, fmt) in VALUES.items():
        if rel not in cache:
            cache[rel] = _load(rel)
        v = _dig(cache[rel], path)
        if fmt == "s":
            out[key] = str(v)
        elif fmt == "d":
            out[key] = f"{int(v):d}"
        else:
            out[key] = format(float(v), fmt)
    return out



#: Where each cited label actually sits in the FROZEN manuscript.
#: P6 SS5 held the page column back until the freeze, because a page number
#: written before it is a number that will be wrong. The manuscript is frozen
#: now, so the labels are resolved -- from paperY_main.aux, which is what
#: pdflatex itself wrote, rather than from anything typed here. A reviewer
#: gets "Sect. 3.2 (p. 4)" instead of a label only we can read.
_KIND = {"sec": "Sect.", "tab": "Table", "fig": "Fig.", "eq": "Eq."}


def _aux_labels(aux: Path) -> dict:
    import re as _re
    return {m.group(1): (m.group(2), m.group(3)) for m in _re.finditer(
        r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}", aux.read_text())}


def locations() -> dict:
    """label -> (number, page), from BOTH documents' own .aux files.

    The ESM was added on adjudication (2026-09-05): the letter was sending
    reviewers to "the Online Resource's cost section", which is a feeling,
    not a location. Its labels are namespaced `esm:` here so a main-text
    label and an ESM label can never be confused for one another -- they
    have different page numbers in different documents.
    """
    sys.path.insert(0, str(CODE_ROOT))
    from experiments.package_layout import paper_file
    aux = paper_file("paperY_main.aux")
    if not aux.exists():
        raise FileNotFoundError(
            "paperY_main.aux is missing: the letter cites section and page "
            "numbers and will not invent them. Compile the manuscript first.")
    out = _aux_labels(aux)
    esm = paper_file("paperY_ESM.aux")
    if not esm.exists():
        raise FileNotFoundError(
            "paperY_ESM.aux is missing: the letter cites Online Resource "
            "sections and pages. Compile the Online Resource first.")
    for k, v in _aux_labels(esm).items():
        out["esm:" + k] = v
    return out


def _resolve_labels(text: str) -> str:
    import re as _re
    loc = locations()
    PAT = r"`((?:esm:)?(?:sec|tab|fig|eq):[A-Za-z0-9_-]+)`"
    used = sorted(set(_re.findall(PAT, text)))
    missing = [u for u in used if u not in loc]
    if missing:
        raise KeyError(
            f"the letter points at labels the manuscript does not define: "
            f"{missing}. A pointer a reviewer cannot follow is worse than no "
            f"pointer; two of these were invented in the first draft.")
    def sub(m):
        k = m.group(1)
        n, pg = loc[k]
        if k.startswith("esm:"):
            #: named as the reviewer sees it -- a separate document with its
            #: own page numbering, so "p. 23" alone would be ambiguous.
            kind = _KIND[k.split(":")[1]]
            return f"Online Resource 1, {kind} {n} (p. {pg})"
        return f"{_KIND[k.split(':')[0]]} {n} (p. {pg})"
    return _re.sub(PAT, sub, text)


def render() -> str:
    body = (Path(__file__).parent / "response_letter_body.md").read_text()
    vals = dict(values())
    vals.update({"repo_url": REPO_URL, "repo_tag": REPO_TAG,
                 "repo_commit": REPO_COMMIT})
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", body)) - set(vals))
    if missing:
        raise KeyError(f"the letter references values with no source: {missing}")
    unused = sorted(set(vals) - set(re.findall(r"\{\{(\w+)\}\}", body))
                    - {"repo_url", "repo_tag", "repo_commit"})
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], body)
    text = _resolve_labels(text)
    #: A digit that reached the prose without going through a reference is
    #: exactly what this generator exists to prevent, so the check is on the
    #: SOURCE, before substitution -- afterwards every number looks typed.
    return text, unused



#: Editorial Manager sometimes makes the marked-up manuscript a REQUIRED
#: upload slot. There is no marked-up copy to give it (see
#: VERA_response/EM_UPLOAD_ORDER.md), so the ruling is to put this letter in
#: that slot with a note on its first page saying what it does instead. The
#: note is prepended by the GENERATOR rather than pasted into a copy: a
#: hand-edited duplicate of a 23,000-character letter is a second thing to
#: keep in step, and it would not be kept in step.
MARKUP_NOTE = """> **Note on this document.** Editorial Manager asked for a marked-up copy of the manuscript. This submission does not include one; this point-by-point response is uploaded in its place, and it carries the same function per item: for every change, it gives the reviewer's words verbatim, what was changed, **the section and page in the revised manuscript where the change lands**, and the evidence behind it. The section and page references are resolved mechanically against the frozen manuscript, not typed. The document is otherwise identical to the response to reviewers submitted with this revision.

---

"""
MARKUP_OUT = ROOT / "VERA_response" / "RESPONSE_TO_REVIEWERS_R1_markup_slot.md"


def render_markup_slot() -> str:
    """The same letter, with the first-page note the ruling requires."""
    text, _ = render()
    i = text.index("\n\n")            # after the H1
    return text[:i + 2] + MARKUP_NOTE + text[i + 2:]


def check(delivered: Path) -> dict:
    text, _ = render()
    have = delivered.read_text() if delivered.exists() else None
    ok = have == text
    return {"pass": ok, "chars": len(text),
            "detail": (f"{delivered.name}: character-identical to what this "
                       f"script renders now ({len(text)} chars)" if ok else
                       f"{delivered.name}: DIFFERS from the rendered letter")}


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    v = values()
    c(len(v) == len(VALUES), f"every declared value resolves ({len(v)})")
    c(v["fair_p"] == v["fair_floor"],
      "the fair pair's p equals its own attainable floor -- the letter must "
      "not read this as a near-miss")
    c(v["fair_T"].startswith("+"), "signed effects render their sign")
    c(float(v["lam_max"]) <= float(v["ln2"]),
      "the scanned gate maximum does not exceed ln 2")
    body = (Path(__file__).parent / "response_letter_body.md")
    c(body.exists(), "the letter body is present")
    if body.exists():
        raw = body.read_text()
        #: The reviewers' own numbers are QUOTATIONS and must stay verbatim --
        #: they are what the reviewer wrote, not what our artifacts say, and
        #: regenerating them would be a quiet edit of someone else's words.
        #: Blockquoted lines are therefore exempt; everywhere else, a typed
        #: decimal is the defect this generator exists to prevent.
        ours = "\n".join(l for l in raw.splitlines()
                          if not l.lstrip().startswith(">"))
        stray = re.findall(r"(?<![\w.=/#§-])\d+\.\d+(?![\w.])", ours)
        c(not stray, f"no decimal is typed outside a quotation: {stray[:6]}")
        quoted = re.findall(r"(?<![\w.=/#-])\d+\.\d+(?![\w.])",
                            "\n".join(l for l in raw.splitlines()
                                       if l.lstrip().startswith(">")))
        c(bool(quoted),
          f"the reviewers' own figures are quoted verbatim ({len(quoted)} found)")
        text, unused = render()
        c("{{" not in text, "every reference was substituted")
        c(not unused, f"no declared value goes unused: {unused}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="FILE")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--values", action="store_true",
                    help="print every reference and the artifact behind it")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.values:
        for k, (rel, path, fmt) in sorted(VALUES.items()):
            print(f"{k:16} {values()[k]:>12}   <- {rel}#{path}")
        return 0
    if a.check:
        r = check(Path(a.check))
        print(("[OK] " if r["pass"] else "[RED] ") + r["detail"])
        return 0 if r["pass"] else 1
    text, _ = render()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"[ok] wrote {OUT} ({len(text)} chars)")
    m = render_markup_slot()
    MARKUP_OUT.write_text(m)
    print(f"[ok] wrote {MARKUP_OUT} ({len(m)} chars) -- the same letter with "
          f"the first-page note, for EM's marked-up slot if it is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
