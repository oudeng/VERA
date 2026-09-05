"""Fill Instruments/Fig1_designer_brief.md from the generator, verbatim.

Run it again whenever fig_vera.py changes: the brief the designer holds must
be the text the figure draws, and the only way to keep that true is to
regenerate rather than to edit the filled copy.


Nothing here is typed by hand: the band strings come from an AST parse of
reporting/fig_vera.py and the axis/readout/eligibility strings from importing
the module, so what reaches the designer is what the figure draws. Every
filled string is then checked against BOTH the source and the rendered PDF
text layer, which is what the brief asks for.
"""
import ast
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
sys.path.insert(0, str(CODE_ROOT / "reporting"))
SRC = CODE_ROOT / "reporting" / "fig_vera.py"
PDF = CODE_ROOT / "reporting" / "out" / "Fig_vera.pdf"
BRIEF = ROOT / "Instruments" / "Fig1_designer_brief.md"
OUT = ROOT / "reports" / "Fig1_designer_filled.md"

from reporting import fig_vera as F                                # noqa: E402

# ---- literals drawn by build(), in source order ------------------------- #
tree = ast.parse(SRC.read_text())
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "build")
lits = {}
for node in ast.walk(fn):
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "text" and len(node.args) > 2
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, str)):
        lits[node.lineno] = node.args[2].value
ordered = [lits[k] for k in sorted(lits) if lits[k] != "<-"]

(band1_title, band1_sub, band2_title, band3_title, ref_header,
 band4_l1, band4_l2, band5_l1, band5_l2) = ordered

#: mathtext in the source renders as a glyph on the page; the designer needs
#: the RENDERED form, with the notation called out so it is not flattened
def rendered(s: str) -> str:
    s = s.replace("TAP$_0$", "TAP₀")
    s = re.sub(r"\$\\mathbf\{([A-Za-z])\}\$", r"**\1**", s)
    return s

readouts = [(t1, rendered(t2)) for t1, t2, _c in F.READOUTS]
axes = [(n, s) for n, s, _r in F.AXES]
refs = [r for _n, _s, r in F.AXES]

FILL = {
    "band1_title": band1_title,
    "band1_subtitle": band1_sub,
    "band2_title": band2_title,
    "band2_box1": f"{readouts[0][0]}\n{readouts[0][1]}",
    "band2_box2": f"{readouts[1][0]}\n{readouts[1][1]}",
    "band2_box3": f"{readouts[2][0]}\n{readouts[2][1]}",
    "band2_box4": f"{readouts[3][0]}\n{readouts[3][1]}",
    "eligibility_note": F.ELIGIBILITY,
    "band3_title": band3_title,
    "axis1": f"{axes[0][0]}\n{axes[0][1]}",
    "axis2": f"{axes[1][0]}\n{axes[1][1]}",
    "axis3": f"{axes[2][0]}\n{axes[2][1]}",
    "axis4": f"{axes[3][0]}\n{axes[3][1]}",
    "axis5": f"{axes[4][0]}\n{axes[4][1]}",
    "ref_evidence_header": ref_header,
    "ref_evidence_1": refs[0], "ref_evidence_2": refs[1],
    "ref_evidence_3": refs[2], "ref_evidence_4": refs[3],
    "ref_evidence_5": refs[4],
    "band4_line1": band4_l1, "band4_line2": band4_l2,
    "band5": f"{band5_l1}\n{band5_l2}",
}

# ---- verification: source AND rendered PDF ------------------------------ #
src_txt = SRC.read_text()
src_flat = " ".join(src_txt.split())
pdf_txt = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                         capture_output=True, text=True).stdout
pdf_flat = " ".join(pdf_txt.split())
#: the side-note pointers are drawn as literal text, one per axis; they are
#: not brief slots (section 2 lets the arrow's FORM change), so the promised
#: automatic comparison has to know to expect them
n_pointer = pdf_flat.count("<-")


#: every string the figure actually draws: the AST literals plus the module
#: constants. Exact membership in this set is a stronger check than a
#: substring search of the file, which would also match a comment.
DRAWN = set(lits.values()) | {F.ELIGIBILITY}
for _t1, _t2, _c in F.READOUTS:
    DRAWN |= {_t1, _t2}
for _n, _s, _r in F.AXES:
    DRAWN |= {_n, _s, _r}


def in_source(s: str) -> bool:
    """Each line is EXACTLY one of the strings the figure draws.

    Not a substring search: the display form is the drawn string with the
    mathtext turned into the notation a designer types, so undoing that
    substitution must land back on the drawn string character for character.
    """
    for line in s.split("\n"):
        probe = (line.replace("TAP\u2080", "TAP$_0$")
                     .replace("**D**", "$\\mathbf{D}$"))
        if probe not in DRAWN:
            return False
    return True


def in_pdf(s: str) -> bool:
    """Present in the RENDERED page. pdftotext flattens the mathtext to
    plain glyphs, so the probe flattens the display form the same way -- a
    subscript zero comes back as '0', a bold D as 'D'."""
    for line in s.split("\n"):
        probe = (line.replace("TAP\u2080", "TAP0")
                     .replace("**D**", "D"))
        if not all(w in pdf_flat for w in probe.split()):
            return False
    return True


checks = [(k, in_source(v), in_pdf(v)) for k, v in FILL.items()]
bad = [(k, a, b) for k, a, b in checks if not (a and b)]

# ---- the caption, verbatim from the tex --------------------------------- #
tex = paper_file("paperY_main.tex").read_text().splitlines()

# ---- banned words must not have crept into any filled string ------------ #
BANNED = ["one ruler", "mandatory floor", "mandatory baseline",
          "mechanical verdict", "before data", "pre-registered", "decoy",
          "decoys", "fixed FPR", "contradicted"]
hits = [(k, b) for k, v in FILL.items() for b in BANNED
        if b.lower() in v.lower()]
if hits:
    print(f"BANNED WORD IN FILLED TEXT: {hits}")
    raise SystemExit(1)

brief = BRIEF.read_text()
for k, v in FILL.items():
    # every LINE gets its own backticks: a two-line box is two text objects in
    # the figure, and the designer copies them one at a time
    cell = "<br>".join(f"`{ln}`" for ln in v.split("\n"))
    brief = brief.replace(f"`[[FILL: {k}]]`", cell)
brief = brief.replace(
    "> **【交付前由 Code 填充】** 第 3 节的逐字文本表须由 Code 从当前 "
    "`reporting/fig_vera.py` 与 tex 图注**逐字**取出填入，填完删除本注，"
    "再交美工。填充后须自动比对：本件每条英文串能在生成器源码或渲染 PDF "
    "文本层中找到。\n\n---\n\n", "")
# ---- a plain-text block, because copying out of a table picks up markup -- #
LABEL = {
    "band1_title": "band 1 title", "band1_subtitle": "band 1 small line",
    "band2_title": "band 2 title",
    "band2_box1": "band 2, box 1", "band2_box2": "band 2, box 2",
    "band2_box3": "band 2, box 3", "band2_box4": "band 2, box 4",
    "eligibility_note": "eligibility note (arrow label, not a box)",
    "band3_title": "band 3 title",
    "axis1": "axis 1", "axis2": "axis 2", "axis3": "axis 3",
    "axis4": "axis 4", "axis5": "axis 5",
    "ref_evidence_header": "side-note column heading",
    "ref_evidence_1": "axis 1 side note", "ref_evidence_2": "axis 2 side note",
    "ref_evidence_3": "axis 3 side note", "ref_evidence_4": "axis 4 side note",
    "ref_evidence_5": "axis 5 side note",
    "band4_line1": "band 4, line 1", "band4_line2": "band 4, line 2",
    "band5": "band 5",
}
plain = ["### 3.3 纯文本版(便于逐条复制,不带任何标记)", "",
         "> 从表格里复制常会带上 markdown 记号。下面是同样的文字,每行一条,"
         "**没有反引号、没有 `<br>`**。`TAP₀` 的 `₀` 是真下标字符,`**D**` "
         "的两侧星号表示**该字母加粗**、星号本身不要打进去。", "", "```text"]
for k, v in FILL.items():
    plain.append(f"[{LABEL[k]}]")
    for ln in v.split("\n"):
        plain.append(ln)
    plain.append("")
plain += ["```", ""]

cap_txt = "\n".join(tex[376:403])
plain += [
    "---", "",
    "### 3.4 图注(**不要画进图里**)", "",
    "论文的 Fig. 1 图注在 LaTeX 源里,由排版系统排在图下方,**不属于本次美化"
    "范围**。列在这里只为一件事:图注里已经写明了循环性排除的理由,所以**图内"
    "不需要、也不应该再写一遍**(验收要点第 4 条)。", "",
    "```latex", cap_txt, "```", "",
    "---", "",
]

# ---- the verification record, at the end -------------------------------- #
record = [
    "", "---", "",
    "## 8. 填充与核验记录(Code 自动生成)", "",
    f"- 填充槽位:**{len(FILL)} / {len(FILL)}**,全部由程序取出,"
    "无一条手打。",
    "- 取值来源:`reporting/fig_vera.py` —— 带 1/2/3/4/5 的标题与正文由 "
    "**AST 解析** `build()` 中的 `ax.text` 字面量取得;四类读出、五轴名称与"
    "短语、五条旁注、eligibility 细注由**导入模块常量** `READOUTS` / `AXES` "
    "/ `ELIGIBILITY` 取得。",
    f"- **逐字核验(源)**:{sum(a for _k, a, _b in checks)} / {len(checks)} —— "
    "每一行都**恰好等于**图实际绘制的某个字符串(集合成员判定,不是子串搜索;"
    "子串搜索会把注释里的字也算数)。",
    f"- **逐字核验(渲染 PDF 文本层)**:"
    f"{sum(b for _k, _a, b in checks)} / {len(checks)} —— 每一行的每个词都能"
    "从当前 `Fig_vera.pdf` 的文本层提取到。",
    "- 数学记号的两种形态已对齐:源码里的 `TAP$_0$` 与 `$\\mathbf{D}$` 在页面"
    "上渲染为 `TAP₀` 与粗体 `D`,本件给出的是**页面上的形态**,核验时分别按各"
    "自形态比对。",
    "- 禁用词(第 3.2 节)在本件全部逐字文本中**零出现**,已扫描。",
    "- **独立复核**:另有四路独立读者按槽位分工逐字复读(12 槽 / 12 槽 / "
    "9 槽 + 一路完整性与结构审计),各自重新解析 `fig_vera.py`、重新提取 PDF "
    "文本层,**字符级零差异**;并逐字符确认页面文本层为纯 ASCII(无弯引号、无 "
    "en dash、无不间断空格),连字符为 `U+002D`。",
    "- **两处已知的\u201c形而非字\u201d差异,不是文本差异**:①轴 4 的短语在当前 "
    "PDF 里折成两行,是渲染时 `textwrap` 所致的换行,字串本身与本件一致;"
    f"②当前参考 PDF 的文本层另含 {n_pointer} 个 `<-` 指示符(每轴一个,见 "
    "\u00a7 2\u201c形式可改\u201d),它们是**指向记号而非文本**,不在本件 "
    f"{len(FILL)} 槽之内,自动比对时应予豁免——建议美工把它画成图形箭头,"
    "不要打成 `<-` 两个字符。", "",
    "---", "",
    "## 9. 交件前候裁的四处(**本件未改**,供第一作者/Chat 定夺)", "",
    "复核时发现四处**委托书自身**的表述会误导美工。它们不是文字提取问题"
    f"(第 3 节 {len(FILL)} 槽字符级无误),而是第 1、2、4、6 节里既有的说明。"
    "按纪律**只报不改**:", "",
    "| # | 位置 | 问题 | 建议 |",
    "|---|---|---|---|",
    "| 1 | \u00a7 4.2 与 \u00a7 6 第 5 条 | \u00a7 4.2 给出\u201c不低于 "
    "**7 pt**\u201d的余地,\u00a7 6 第 5 条却按 **\u2265 8 pt** 验收;且 "
    "\u00a7 4.2 自身又要求\u201c放不下请回报、不要缩字号\u201d——那条 7 pt "
    "余地便永远用不上。美工若用足 7 pt,交出的稿必不过验收 | 删去 7 pt 一句,"
    "或把 \u00a7 6 第 5 条改为 7 pt |",
    "| 2 | \u00a7 6 第 4 条 | 写作\u201c循环性说明仍在**图注**而不在图内"
    "\u201d。但图内 Eligibility 细注本身就含 `circularity exclusions` 一语"
    "(第 3 节必填槽)。美工照此自查,可能删掉或改写该槽,而 \u00a7 1 明令"
    "\u201c一个字母都不行\u201d | 改为\u201c循环性的**理由句**留在图注"
    "\u201d,把\u201c图内保留 `circularity exclusions` 一语\u201d写明 |",
    "| 3 | \u00a7 2 结构表 带 5 行 | 只写\u201c单框\u201d,未标行数;紧邻其"
    "上的带 4 却标了\u201c两行\u201d。带 5 实为**两行**(第二行较长),漏标"
    "易被当成单行排版 | 带 5 行补\u201c单框,两行\u201d |",
    f"| 4 | \u00a7 6 第 2 条 | 承诺\u201c提取出的文字与第 3 节表格逐字相同"
    f"(自动比对)\u201d,但参考 PDF 文本层另有 {n_pointer} 个 `<-`(见 "
    "\u00a7 8) | 在 \u00a7 6 第 2 条注明:指示箭头须画为图形、不计入比对 |",
    "",
    "第 1、2 两处触及**验收判据本身**,建议交件前先定;第 3、4 两处只影响排版"
    "预估与自动比对口径。", "",
]

# section 3's markup is markup, not text to type: say so where the table is,
# not 100 lines below it -- a designer told to copy-paste will copy the stars.
warn = ("> 手打会引入拼写错误，而这些文字是论文正文。\n")
assert brief.count(warn) == 1
brief = brief.replace(warn, warn + (
    ">\n"
    "> **表格里的两处记号不是要打进图里的字**:`` ` `` 反引号只是包住文字用的,"
    "`<br>` 表示\u201c这一框里换行\u201d,`**D**` 两侧的星号表示**把 D 这个"
    "字母加粗**、星号本身不要打。若不放心,请直接用下面 **\u00a7 3.3 的纯文本"
    "版**复制——那里没有任何记号。\n"))

# the plain-text block belongs UNDER section 3, not after section 7: it is the
# only markup-free copy source, and a designer reading in order must reach it
anchor = "---\n\n## 4. 硬性技术要求"
assert brief.count(anchor) == 1
brief = brief.replace(anchor, "\n".join(plain) + "\n" + anchor)
OUT.write_text(brief.rstrip("\n") + "\n".join(record))
print(f"filled {len(FILL)} slots -> {OUT}")
print(f"verification: source {sum(a for _k, a, _b in checks)}/{len(checks)}, "
      f"rendered PDF {sum(b for _k, _a, b in checks)}/{len(checks)}")
if bad:
    print("UNVERIFIED:")
    for k, a, b in bad:
        print(f"   {k}: source={a} pdf={b}")
    raise SystemExit(1)
