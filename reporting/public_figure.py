"""Fig. 1 and its caption, for the public repository.

A reader who lands on the repository should be able to see what VERA IS before
deciding whether to read 29 pages. The protocol schematic is the one artifact
that does that, and it is safe to publish: it prints no data number -- gate 12
re-derives that from the delivered PDF every run -- so it cannot drift away
from t_final the way a results figure would.

Nothing here is transcribed. The caption is the manuscript's own \\caption{},
lifted by brace matching, with the macros expanded and the cross-references
RESOLVED FROM THE .aux. A hand-copied caption would be a fourth place for that
text to live and the one place nobody re-reads before submission; a caption
that cited "Sect. 4.4" as a typed string would be wrong the first time a
section moved.

    PYTHONHASHSEED=2025 python reporting/public_figure.py --out <dir>
    PYTHONHASHSEED=2025 python reporting/public_figure.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from experiments.package_layout import paper_file       # noqa: E402

REGISTRY = CODE_ROOT / "docs" / "figure_assets.json"
#: Rendered for inline display only. GitHub will not show a PDF in a README,
#: and a reader who has to download a file to see the diagram will not see the
#: diagram. 200 dpi is legible for the 9 pt axis notes without being large.
PNG_DPI = 200


def _labels() -> dict:
    """Section numbers, read from the .aux the manuscript last produced."""
    aux = paper_file("paperY_main.aux")
    if not aux.exists():
        return {}
    return {m.group(1): m.group(2) for m in
            re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux.read_text())}


def caption() -> str:
    """The Fig. 1 caption, from the manuscript, as plain text."""
    tex = paper_file("paperY_main.tex").read_text()
    i = tex.index("\\label{fig:vera}")
    j = tex.rindex("\\caption{", 0, i)
    k, depth = j + len("\\caption{"), 1
    while depth:
        if tex[k] == "{":
            depth += 1
        elif tex[k] == "}":
            depth -= 1
        k += 1
    body = tex[j + len("\\caption{"):k - 1]

    lab = _labels()
    body = re.sub(r"(?<!\\)%.*", "", body)              # provenance comments
    body = re.sub(r"\\ref\{([^}]+)\}",
                  lambda m: lab.get(m.group(1), "?"), body)
    body = (body.replace("\\VERA{}", "VERA").replace("\\TAP{}", "TAP")
                .replace("\\Dm{}", "**D**")
                .replace("\\VERA", "VERA").replace("\\TAP", "TAP"))
    body = body.replace("Sect.~", "Sect. ").replace("~", " ")
    body = body.replace("---", "\u2014").replace("--", "\u2013")
    body = re.sub(r"\s+", " ", body).strip()
    if "\\" in body:
        left = re.findall(r"\\[A-Za-z]+", body)
        raise ValueError(f"unexpanded TeX left in the caption: {left}")
    return body


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def asset() -> tuple:
    reg = json.loads(REGISTRY.read_text())
    a = next(x for x in reg["assets"] if x["figure"] == "Fig_vera")
    pdf = ROOT / a["path"]
    if not pdf.exists():
        pdf = paper_file(Path(a["path"]).name)
    got = _sha(pdf)
    if got != a["pdf_sha256"]:
        raise ValueError(
            f"the figure about to be published is not the registered asset: "
            f"{got[:12]}.. vs {a['pdf_sha256'][:12]}... Publishing an "
            f"unregistered figure would put a diagram in front of readers "
            f"that no gate has checked.")
    return pdf, a


def build(out: Path) -> dict:
    """Write the PDF, a rendered PNG and the caption into `out`."""
    pdf, a = asset()
    out.mkdir(parents=True, exist_ok=True)
    (out / "Fig_vera.pdf").write_bytes(pdf.read_bytes())
    subprocess.run(["pdftoppm", "-r", str(PNG_DPI), "-png", "-singlefile",
                    str(pdf), str(out / "Fig_vera")], check=True)
    png = out / "Fig_vera.png"
    cap = caption()
    (out / "README.md").write_text(
        "# Fig. 1 — the VERA evaluation protocol\n\n"
        "![The VERA evaluation protocol](Fig_vera.png)\n\n"
        f"**Fig. 1** {cap}\n\n"
        "---\n\n"
        "The caption above is the manuscript's own, extracted from "
        "`paperY_main.tex` with its cross-references resolved; it is not "
        "transcribed. `Fig_vera.pdf` is the vector original as submitted, "
        f"sha256 `{a['pdf_sha256']}`; `Fig_vera.png` is rendered from it at "
        f"{PNG_DPI} dpi for inline display.\n\n"
        "**This figure is not covered by the repository's MIT license.** It is "
        "the authors' own work, provided here so the protocol can be "
        "understood without reading the paper. Rights in it are reserved "
        "pending the journal's publishing agreement.\n")
    return {"pdf_sha256": _sha(out / "Fig_vera.pdf"),
            "png_sha256": _sha(png), "png_bytes": png.stat().st_size,
            "caption_chars": len(cap), "dpi": PNG_DPI}


def _selftest() -> int:
    ok = True

    def c(cond, msg):
        nonlocal ok
        print(("[PASS] " if cond else "[FAIL] ") + msg)
        ok = ok and bool(cond)

    cap = caption()
    c(len(cap) > 800, f"the caption is the manuscript's, whole ({len(cap)} chars)")
    c("\\" not in cap, "every macro and cross-reference is expanded")
    c("%" not in cap, "provenance comments are stripped")
    lab = _labels()
    c(f"Sect. {lab.get('sec:faithfulness')}" in cap,
      f"the cross-reference is RESOLVED, not typed "
      f"(sec:faithfulness -> {lab.get('sec:faithfulness')})")
    c("Sect. ?" not in cap, "no cross-reference failed to resolve")
    c(cap.startswith("The VERA evaluation protocol"),
      "the caption starts where the manuscript's does")
    pdf, a = asset()
    c(_sha(pdf) == a["pdf_sha256"],
      "the figure published is the REGISTERED asset, byte for byte")
    #: The reason this is safe to publish at all.
    txt = subprocess.run(["pdftotext", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    nums = sorted(set(re.findall(r"\d+\.\d+|\d+%|\d+/\d+", txt)))
    c(not nums, f"the figure prints no data number, so it cannot drift from "
                f"t_final: {nums[:5]}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    r = build(Path(a.out))
    print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
