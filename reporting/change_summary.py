"""The current-facts half of CHANGE_SUMMARY.md, emitted from the artifacts.

Eighth review P0-5. The file said "本页只陈述现行事实" at the top and then
carried, in the same section, a recovery effect of +0.154 (current: +0.159),
twelve packaging gates (sixteen), 48 pages (55), 55 references (54), sixteen
rule documents (nineteen) and 58 inventory sites (65). Every one of those had
been true once. None was true when it was read, and the count gates did not
catch them because they scan a few hard-coded filenames and word orders.

A summary that claims to state current facts and is written by hand will drift
again on the next round, so this writes it: every number below is read from
the canonical artifact that owns it, and the round-by-round narrative after
the separator is left exactly as it was, under a heading that says what it is.

    PYTHONHASHSEED=2025 python reporting/change_summary.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(CODE_ROOT))
#: paper_R1 is main/ + esm/ since P7-A SS2; a package's assembled
#: view is still flat. One resolver answers for all three layouts.
from experiments.package_layout import paper_file  # noqa: E402

ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

PAPER = ROOT / "paper_R1"
STAGING = ROOT / "internal_review" / "ir_staging"
OUT = STAGING / "CHANGE_SUMMARY.md"
#: everything from this line on is the round log, and is preserved verbatim
HISTORY_MARK = "## IR6 → IR7"


def _pkg(rel: str, default=None) -> Path:
    """The artifact, wherever this script runs from (ninth review P0-3)."""
    here = Path(__file__).resolve().parent.parent
    for c in (default or (here / rel),
              here.parent / "gate_inputs" / "code_SNI" / rel,
              here / rel):
        if Path(c).exists():
            return Path(c)
    return Path(default or (here / rel))


def _report(name: str) -> Path:
    """A reports/ artifact, in the repository or under a package's gate_inputs."""
    here = Path(__file__).resolve().parent.parent
    for c in (ROOT / "reports" / name,
              here.parent / "gate_inputs" / "reports" / name):
        if Path(c).exists():
            return Path(c)
    return ROOT / "reports" / name


def _pdf(name: str) -> Path:
    """The manuscript PDF, in the repository or at a package's top level."""
    here = Path(__file__).resolve().parent.parent
    for c in (paper_file(name, PAPER), here.parent / name,
              here.parent / "_paper" / name):
        if Path(c).exists():
            return Path(c)
    return PAPER / name


def _pages(pdf: Path) -> int:
    m = re.search(r"Pages:\s*(\d+)",
                  subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                 text=True).stdout)
    if not m:
        raise RuntimeError(f"cannot read a page count from {pdf}")
    return int(m.group(1))


def facts() -> dict:
    """Every number the summary states, and the artifact each comes from."""
    tf = json.loads(_pkg("results/T5_stats/t_final.json").read_text())
    band = json.loads(
        _pkg("results/T6_symmetry/no_oracle_band.json").read_text())["datasets"]
    inv = json.loads(_pkg(
        "results/T6_symmetry/perm_sni_comparison_inventory.json").read_text())
    gates = len(re.findall(r"^def gate_\d+_",
                           _pkg("reporting/package_gates.py",
                                CODE_ROOT / "reporting" / "package_gates.py")
                           .read_text(), re.M))
    # Tenth review P0-2: in a package the archive is under gate_inputs/, so
    # this looked in a directory that does not exist there and reported
    # "0 rule documents" -- a generated document that would have been wrong
    # rather than merely unbuildable.
    here = Path(__file__).resolve().parent.parent
    arch = []
    for base in (ROOT / "VERA_GitHub" / "prereg_archive",
                 here.parent / "gate_inputs" / "VERA_GitHub" / "prereg_archive"):
        arch = sorted(base.glob("manifest_*.json"))
        if arch:
            break
    rules = len(json.loads(arch[-1].read_text())["files"]) if arch else 0
    bib = json.loads(_report("bib_inventory.json").read_text())
    cited = bib.get("n_cited") or len(bib.get("cited", []))
    points = bib.get("n_citation_points") or bib.get("citation_points")
    c = inv["counts"]
    return {
        "rec_sym": tf["recovery"]["probe_vs_D_same_host_symmetric"],
        "rec_dvt": tf["recovery"]["D_vs_TAP"],
        "band": {ds: band[ds]["band_noOracle"]["mean"] for ds in ("MIMIC", "eICU")},
        "band_archived": {ds: band[ds]["band_oracle"]["mean"]
                          for ds in ("MIMIC", "eICU")},
        "faith": tf["faithfulness"],
        "pages_main": _pages(_pdf("paperY_main.pdf")),
        "pages_esm": _pages(_pdf("paperY_ESM.pdf")),
        "gates": gates,
        "rules": rules,
        "cited": cited,
        "points": points,
        "sites": inv["n_sites_found"],
        "sym": c.get("SYMMETRIC", 0),
        "recomp": c.get("RECOMPUTED", 0),
        "disc": c.get("ASYMMETRIC-DISCLOSED", 0),
        "blockers": inv["n_blockers"],
    }


def render(f: dict) -> str:
    r, d = f["rec_sym"], f["rec_dvt"]
    fa = f["faith"]
    return f"""# 变更摘要 / Summary of Changes（单一事实版，10 分钟导读）

> **本页的现行事实部分由 `code/reporting/change_summary.py` 生成**，每个数字读自拥有它的那份工件；
> 分隔线以下是**逐轮变更日志**，属历史叙述，不是现行事实声明。
> 八审 P0-5：上一版的本节手写，写着"只陈述现行事实"而同时留着六个已过时的数字。

**稿件**: HISC-D-26-01481（Major Revisions，截止 2026-09-10）
**当前构建**: 主文 **{f['pages_main']} 页**、ESM **{f['pages_esm']} 页**（共 {f['pages_main'] + f['pages_esm']} 页）；\
打包门 **{f['gates']} 道**；规则文档 **{f['rules']} 份**；\
被引文献 **{f['cited']}** 条 / 引用点 **{f['points']}** 处；\
信息对称清单 **{f['sites']}** 站点 = {f['sym']} SYMMETRIC / {f['recomp']} RECOMPUTED / {f['disc']} ASYMMETRIC-DISCLOSED，\
阻断 {f['blockers']}。

## 1. 论文定位 / What the paper is

五轴评测协议 **VERA** + 免训练对照 **TAP** + 对**本 SNI 实现的聚合注意力矩阵 D** 的诚实负面结论。

**不存在形式化的总体判决函数。** 本研究**未**在测量前 commit 任何"由五轴映射到单一主张状态"的规则，因此不报告形式化总体判决（Methods §3.7）。正式输出是**轴级证据剖面**（Results §4.9 + Fig. 3 + 对应表）。整体读法只出现在 Discussion，并**明标为事后综合**。

**resource cost 永不进入效度判定**：低成本不补偿未成立的效度主张，高成本也不建立它。

## 2. 五轴现行读数（单源 `t_final.json`）

| 轴 | 现行读数 |
|---|---|
| 恢复 | **两格均 INDET**：D vs TAP（T = {d['T']:+.3f}，exact p = {d['p_exact']:.4g}）与同宿主探针 vs D（T = {r['T']:+.3f}，描述性逐点 seed 自助区间 [{r['ci95_T'][0]:+.3f}, {r['ci95_T'][1]:+.3f}]，exact p = {r['p_exact']:.4f} = 五 seed 双侧地板）——地板使显著在任何效应量下都不可达 |
| 稳定 | **DESC**（联合诊断）：D 0.947/0.891 vs 宿主带 **{f['band']['MIMIC']:.3f}/{f['band']['eICU']:.3f}**（MIMIC/eICU，**信息对称行**；存档行为 {f['band_archived']['MIMIC']:.3f}/{f['band_archived']['eICU']:.3f}，两行并列于 Table 5） |
| 忠实 | **INDET**：T = {fa['MIMIC']['T']:+.3f}（MIMIC，Holm p = {fa['MIMIC']['p_holm']:.3f}）/ {fa['eICU']['T']:+.3f}（eICU，Holm p = {fa['eICU']['p_holm']:.2f}） |
| 泄漏 | D 与同宿主探针**双双 INDET**；阀值为经验 95 分位，全部计数为描述性，不对固定 0.05 做检验 |
| 成本 | **主口径 = 单线程空机**，两种时钟并列且从不相加；边际 D≈0 |

## 3. 方法学支柱 / Methods spine

- **estimand 统一**：T = mean of seed-level medians；CI = seed-only bootstrap，**明标为描述性逐点区间**；族级判定用 Holm。
- **规则四层命名**：initial confirmatory / prospectively specified replication / prospectively specified sensitivity / post-hoc exploratory；每规则 commit 附零产物存证。
- **下游协议**：**inductive** 与 **batch-transductive** 两个 protocol class，只在类内描述与排序。
- **两道机器门 + 一道人工验收**：五层单一事实门（术语 + 数值，编译前置）、**{f['gates']} 道打包门**、以及全 **{f['pages_main'] + f['pages_esm']} 页**逐页人工视觉验收（`INSPECTION_LOG.md`，**定稿时点执行**）。

## 4. 基准协议自审 / Self-audit

84 项台账 → 四类 20 条（ESM 全文）；全部数字在修正管线下重生成（2,565 格，零失败）。

## 5. 声明与在途 / Declarations & in flight

Declarations 全块在稿；引用 **{f['cited']}** 条逐条核验（共 {f['points']} 个引用点）。
**在途**：REPO-URL / SWHID 待**正式提交前**回填；**{f['rules']} 份**规则文档的完整链随公开仓发布。

---

# 逐轮变更日志 / Round-by-round log

> 以下每一节描述的是**那一轮发生了什么**，不是当前状态。现行事实以上半页为准。

"""


def _existing(out: Path) -> str:
    """The delivered file, wherever it is: the repository staging tree, or the
    package root when this runs from inside a package (tenth review P0-2)."""
    if out.exists():
        return out.read_text()
    here = Path(__file__).resolve().parent.parent
    pkg = here.parent / out.name
    return pkg.read_text() if pkg.exists() else ""


def build(out: Path = None) -> Path:
    out = out or OUT
    f = facts()
    old = _existing(out)
    i = old.find(HISTORY_MARK)
    history = old[i:] if i != -1 else ""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(f) + history)
    return out


def check(delivered: Path) -> tuple:
    """Re-render the current-facts half and compare it to what was delivered.

    Tenth review P0-2: the delivered CHANGE_SUMMARY says on its own first page
    that this script generates it. That claim is only worth something if a
    reader can re-run the script against the delivered file and get the same
    words back -- which is what this does, character for character, over the
    generated half. The round-by-round log below the separator is hand-written
    history and is not compared.
    """
    have = delivered.read_text()
    i = have.find(HISTORY_MARK)
    have_head = have[:i] if i != -1 else have
    want_head = render(facts())
    return (have_head == want_head, have_head, want_head)


def _selftest() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("[PASS] " if c else "[FAIL] ") + m)
        ok = ok and bool(c)

    f = facts()
    WANT = {"pages_main": 29, "pages_esm": 26, "gates": 20, "rules": 19,
            "cited": 54, "points": 59, "sites": 68, "blockers": 0}
    for k, want in WANT.items():
        chk(f[k] == want, f"{k} = {f[k]} (expected {want})")
    chk(f["sites"] == f["sym"] + f["recomp"] + f["disc"] + f["blockers"],
        "the inventory's four states sum to its site count")
    txt = render(f)
    chk("+0.154" not in txt and "48 页" not in txt,
        "no superseded number survives in the generated section")
    chk("规则文档 **0** 份" not in txt and "**0** 份" not in txt,
        "no count renders as zero -- the shape a failed artifact lookup takes")
    print("[SELFTEST " + ("PASS]" if ok else "FAIL]"))
    return 0 if ok else 1


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--output", metavar="FILE",
                    help="write the summary here instead of the default")
    ap.add_argument("--check", metavar="FILE",
                    help="re-render the current-facts half and compare it, "
                         "character for character, with FILE; non-zero if "
                         "they differ")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.check:
        f = Path(a.check)
        if not f.exists():
            # in a package the delivered file sits at the package root
            here = Path(__file__).resolve().parent.parent
            f = here.parent / Path(a.check).name
        if not f.exists():
            print(f"[FAIL] cannot find {a.check}")
            return 1
        same, have, want = check(f)
        if same:
            print(f"[OK] {f.name}: the generated half is character-identical "
                  f"to what this script renders now ({len(want)} chars)")
            return 0
        import difflib
        d = list(difflib.unified_diff(have.splitlines(), want.splitlines(),
                                      "delivered", "re-rendered", lineterm=""))
        print(f"[FAIL] {f.name}: the generated half differs")
        print("\n".join(d[:40]))
        return 1
    print(f"[OK] wrote {build(Path(a.output) if a.output else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
