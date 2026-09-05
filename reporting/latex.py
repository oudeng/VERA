"""LaTeX emission helpers (engineering principle E5).

R0 produced `.tex` fragments and a human then pasted them into the manuscript,
applying a consistent set of edits by hand every time. Comparing the R0 generator
output with the submitted manuscript shows the edits were systematic:

    generator                          manuscript
    \\begin{table}[t]                   \\begin{table*}[t]      (main text is two-column)
    \\small                             \\scriptsize
    caption "... Best in \\textbf{bold}" caption without that sentence
    \\textbf{} on the best cell          no bold anywhere
    (nothing)                          \\begin{minipage} protocol note

Because the manuscript has zero `\\input{}` commands, none of those edits ever
flowed back into the generator, and the manuscript accumulated hand corrections
the scripts did not know about. The clearest casualty is the Table S3 sign
convention: `06_gen_supp_tables.py:355-360` still states it backwards, the
manuscript was fixed by hand at `ESM:353`, and re-running the generator would have
silently reintroduced the error.

This module applies the final styling at generation time so the emitted file is
what the manuscript uses, via `\\input{}`, with no manual step in between.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

#: Column-name escaping is deliberately NOT automatic — headers routinely contain
#: intentional math such as ``$R^2$`` and ``$\uparrow$``. Escape data cells only.
_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}


#: Suffixes that make a token code rather than an English word.
_CODE_EXT = r"(?:py|ya?ml|json|jsonl|csv|md|tex|txt|sh|cfg|toml)"
#: A candidate run: what an identifier or a path is made of, with the
#: underscore in the shape the escaper above has already given it.
_CODE_TOKEN = re.compile(r"(?:\\_|[A-Za-z0-9./-])+")
#: What qualifies a candidate: an underscore, or a code file suffix.
_IS_CODE = re.compile(r"\\_|\." + _CODE_EXT + r"(?![A-Za-z])")


def codeify(s: str) -> str:
    """Typeset identifier-shaped tokens as code that cannot be broken.

    The seventh review found ``budget_params`` set as ``bud-get_params`` and
    ``registry.py`` split across a line break in the ESM hyperparameter
    table: those cells were escaped but never marked as code, so LaTeX
    typeset them in roman and hyphenated them like English words. Wrapping
    at the call site would fix those two cells; this fixes the point where
    every table cell is written, so a cell that names a file or a parameter
    is code wherever it is written.

    Deliberately narrow: a run qualifies only if it carries an underscore or
    ends in a code suffix, so ``MIMIC-III``, ``80/20`` and ``0.05`` are left
    alone. A cell that already carries LaTeX is left alone entirely.
    """
    if re.search(r"\\(?!_)", s):          # the cell already carries LaTeX
        return s

    def rep(m: "re.Match") -> str:
        tok = m.group(0)
        tail = ""
        while tok and tok[-1] in "./-":     # sentence punctuation, not the name
            tail, tok = tok[-1] + tail, tok[:-1]
        if not tok or not _IS_CODE.search(tok):
            return m.group(0)
        return r"\codeid{" + tok + "}" + tail

    return _CODE_TOKEN.sub(rep, s)


def code_cell(value: object) -> str:
    """Escape a cell and mark the whole of it as code.

    For a column that holds nothing but identifiers. codeify() alone leaves
    such a column mixed -- it marks ``mortality_risk`` and leaves ``mpg``
    in roman, because only the first is identifier-SHAPED -- and a column set
    half in one face and half in another reads as a mistake.
    """
    return r"\codeid{" + "".join(_ESCAPE.get(ch, ch) for ch in
                                 ("" if value is None else str(value))) + "}"


def escape_cell(value: object) -> str:
    s = "" if value is None else str(value)
    return codeify("".join(_ESCAPE.get(ch, ch) for ch in s))


@dataclass
class TableStyle:
    """The styling the manuscript actually uses, applied at generation time."""

    environment: str = "table*"      # main text is two-column; ESM uses "table"
    placement: str = "t"
    size: str = r"\scriptsize"
    centering: bool = True
    booktabs: bool = True
    bold_best: bool = False          # the manuscript carries no bold in these tables
    row_colour: Optional[str] = None  # the manuscript carries no shading either
    notes: Sequence[str] = field(default_factory=tuple)
    notes_width: str = r"\textwidth"
    #: Row compression, for the one shape LaTeX will not warn about: a table
    #: plus its note taller than the text block. There is no Overfull \vbox
    #: for it -- the float simply prints past the bottom margin and the folio
    #: lands inside the last line. reporting/render_checks.py measures it on
    #: the rendered page; this is the knob the measurement calls for.
    row_stretch: Optional[float] = None
    #: Inter-column padding, when a table is a few points too wide for the
    #: measure and the remedy must not be an abbreviation. LaTeX does warn
    #: about this one (Overfull \hbox), so it is caught -- but the default
    #: remedy people reach for is shortening a label back into a key.
    col_sep_pt: Optional[float] = None


#: Vertical air around the rules that bound a table. Both were found by
#: rendering pages and measuring pixels, not by reading LaTeX: at the class's
#: defaults the caption's descenders touched the top rule and the bottom rule
#: touched the note's first line. Changing these changes every table at once,
#: which is the point -- the defect class is "a rule drawn onto text", and it
#: has exactly one emission site.
CAPTION_GAP_PT = 6
#: NOT \belowcaptionskip: sn-jnl boxes the table caption itself
#: (\@tablecaption) and the class's own \vskip never reaches the page --
#: measured 0.5 pt of air under a caption that asked for 7. The air is
#: therefore emitted explicitly, and render_checks.py measures it.
CAPTION_RULE_GAP = rf"\par\vspace{{{CAPTION_GAP_PT}pt}}"
#: Same trap, other end of the table: a bare \vspace after \end{tabular}
#: is issued in horizontal mode, so the skip is DEFERRED past the note and
#: the note lands on the rule. The \par is what makes the air real.
NOTE_GAP_PT = 7
NOTE_RULE_GAP = rf"\par\vspace{{{NOTE_GAP_PT}pt}}"


#: Shapes that mean an internal object reached the typesetter instead of a
#: rendering of it. Every one of these has actually shipped: a Python list of
#: driver names printed with its brackets and quotes, a coefficient dict
#: printed with its braces, a p value printed as "1.2e-04" beside a properly
#: typeset one. None is illegal LaTeX and none is a banned term, so nothing
#: but a reader ever caught them.
_REPR_SHAPES = [
    (re.compile("^\\s*[\\[(]\\s*['\u2019\"]"), "a Python list/tuple repr"),
    (re.compile("[{]\\s*[-+0-9'\u2019\"]"), "a Python dict repr"),
    (re.compile("^\\s*(?:nan|NaN|None|inf|-inf|<[a-z]+ object)"),
     "a missing/None/inf sentinel"),
    (re.compile("\\b\\d(?:\\.\\d+)?e[-+]\\d\\d?\\b"), "machine e-notation"),
    (re.compile("(?<![\\d.])[-\u2212]0\\.0+(?![0-9])"), "a signed zero"),
]


#: How an internal method key is written for a reader. One map, because the
#: same baseline was rendered "MeanMode" in one table and "Mean/Mode" in the
#: table on the facing page.
METHOD_DISPLAY = {"MeanMode": "Mean/Mode"}
#: The synthetic pilot's regime keys, as a reader should meet them. They
#: reached the ESM as raw snake_case -- the same shape as the "P-alone" leak
#: an earlier round fixed.
REGIME_DISPLAY = {"linear_gaussian": "Linear-Gaussian",
                  "nonlinear_mixed": "Nonlinear mixed",
                  "interaction_xor": "Interaction-XOR"}


def display_name(key: str) -> str:
    k = str(key)
    return METHOD_DISPLAY.get(k, REGIME_DISPLAY.get(k, k))


def signed(x: float, nd: int = 3) -> str:
    """A signed number, except that zero has no sign.

    "-0.000" shipped as a confidence bound: a magnitude below the printed
    precision, rendered as if the interval reached below zero. At this
    precision the sign is not determined, so it is not printed.
    """
    t = f"{float(x):+.{nd}f}"
    return f"{0.0:.{nd}f}" if float(t) == 0.0 else t


def math_signed(x: float, nd: int = 3) -> str:
    """A signed number set in math mode, so its sign is a MINUS.

    Outside math, "-0.030" prints a hyphen. The same quantity is set as
    "$-0.030$" everywhere in the running text, so a table that writes it the
    other way puts two different characters on the page for one sign (seventh
    review SS13 on symbol style; found on main p.27 in the eighth-round freeze
    inspection).
    """
    return f"${signed(x, nd)}$"


def refuse_repr(value: str, where: str) -> str:
    """A table cell must be a rendering, never an object's repr."""
    t = str(value)
    # A cell carrying a LaTeX COMMAND is markup the generator wrote on purpose
    # (\multicolumn{7}{l}{...}); its braces are not a dict's. An escaped brace
    # (\{) is not a command, so an escaped dict is still caught.
    is_markup = bool(re.search(r"\\[A-Za-z]", t))
    for rx, what in _REPR_SHAPES:
        if is_markup and "Python" in what:
            continue
        if rx.search(t):
            raise ValueError(
                f"{where}: refusing to typeset {what} -- {t[:80]!r}. Render it "
                f"for a reader (a list as a list of names, a dict as "
                f"'key: value', an exponent as a power of ten, a bound below "
                f"print precision without a minus sign).")
    return t


def _minipage(notes: Sequence[str], width: str) -> List[str]:
    """Protocol / interpretation block, as the manuscript formats it."""
    if not notes:
        return []
    out = [NOTE_RULE_GAP, rf"\begin{{minipage}}{{{width}}}", r"\scriptsize"]
    for n in notes:
        out.append(n + r"\\")
    if out[-1].endswith(r"\\"):
        out[-1] = out[-1][:-2]
    out.append(r"\end{minipage}")
    return out


def dataframe_to_tex(
    df: pd.DataFrame,
    *,
    caption: str,
    label: str,
    column_format: Optional[str] = None,
    header: Optional[Sequence[str]] = None,
    style: Optional[TableStyle] = None,
    escape_data: bool = True,
    midrule_after: Iterable[int] = (),
) -> str:
    """Render a DataFrame as a finished, `\\input`-able table.

    ``header`` overrides the column names and is emitted verbatim, so it may carry
    math. Data cells are escaped unless ``escape_data=False``.
    ``midrule_after`` inserts ``\\midrule`` after the given zero-based row indices,
    which is how the manuscript separates panels.
    """
    st = style or TableStyle()
    ncol = df.shape[1]
    colfmt = column_format or ("l" + "c" * (ncol - 1))
    head = list(header) if header is not None else [str(c) for c in df.columns]
    if len(head) != ncol:
        raise ValueError(f"header has {len(head)} entries but the frame has {ncol} columns")

    if st.environment == "longtable":
        return _longtable(df, caption=caption, label=label, colfmt=colfmt,
                          head=head, st=st, escape_data=escape_data,
                          midrule_after=midrule_after)

    lines = [rf"\begin{{{st.environment}}}[{st.placement}]"]
    if st.centering:
        lines.append(r"\centering")
    if st.row_stretch:
        lines.append(rf"\renewcommand{{\arraystretch}}{{{st.row_stretch}}}")
    if st.col_sep_pt:
        lines.append(rf"\setlength{{\tabcolsep}}{{{st.col_sep_pt}pt}}")
    if st.size:
        lines.append(st.size)
    # The gap between a caption set ABOVE the table and the table's top rule is
    # not the class's default here: with the body at \scriptsize and the
    # caption at its own size, the class's belowcaptionskip leaves the
    # caption's descenders sitting ON the rule -- it reads as a strikethrough,
    # and fourteen ESM tables shipped that way before anyone rendered a page.
    # Set it explicitly, in the emitter, so no table can be laid out without it.
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              CAPTION_RULE_GAP,
              rf"\begin{{tabular}}{{{colfmt}}}"]
    lines.append(r"\toprule" if st.booktabs else r"\hline")
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule" if st.booktabs else r"\hline")

    after = set(midrule_after)
    for i, (_, row) in enumerate(df.iterrows()):
        cells = [refuse_repr(escape_cell(v) if escape_data else str(v),
                             f"{label} column {head[k]!r}")
                 for k, v in enumerate(row)]
        lines.append(" & ".join(cells) + r" \\")
        if i in after:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule" if st.booktabs else r"\hline")
    lines.append(r"\end{tabular}")
    lines += _minipage(st.notes, st.notes_width)
    lines.append(rf"\end{{{st.environment}}}")
    return "\n".join(lines) + "\n"


def _longtable(df, *, caption, label, colfmt, head, st, escape_data,
               midrule_after) -> str:
    """A table too tall for one page: it must break, so it cannot float.

    The 42-row leakage readout is taller than a page; as a float it was
    carried past the section it belongs to, and set in place it overflowed
    the page (fourth internal review SS4.1). A longtable breaks across
    pages and stays in reading order, which is what an ESM needs.
    """
    n = len(head)
    out = [r"\begingroup",
           # longtable's caption box defaults to 4in; match the other tables
           r"\setlength{\LTcapwidth}{\textwidth}"]
    if st.size:
        out.append(st.size)
    # ...and the caption gap goes OUTSIDE the environment: inside a longtable
    # everything between rows must sit in \noalign, and a bare \setlength
    # there is "Misplaced \noalign" -- which is how this was found.
    out.append(rf"\begin{{longtable}}{{{colfmt}}}")
    out += [rf"\caption{{{caption}}}", rf"\label{{{label}}}\\[{CAPTION_GAP_PT}pt]",
            r"\toprule" if st.booktabs else r"\hline",
            " & ".join(head) + r" \\",
            r"\midrule" if st.booktabs else r"\hline",
            r"\endfirsthead",
            rf"\multicolumn{{{n}}}{{l}}{{\emph{{Table \thetable{{}} "
            rf"(continued)}}}}\\[{CAPTION_GAP_PT - 2}pt]",
            r"\toprule" if st.booktabs else r"\hline",
            " & ".join(head) + r" \\",
            r"\midrule" if st.booktabs else r"\hline",
            r"\endhead",
            r"\midrule" if st.booktabs else r"\hline",
            rf"\multicolumn{{{n}}}{{r}}{{\emph{{continued on the next "
            rf"page}}}}\\",
            r"\endfoot",
            r"\bottomrule" if st.booktabs else r"\hline",
            r"\endlastfoot"]
    after = set(midrule_after)
    for i, (_, row) in enumerate(df.iterrows()):
        cells = [refuse_repr(escape_cell(v) if escape_data else str(v),
                             f"{label} column {head[k]!r}")
                 for k, v in enumerate(row)]
        out.append(" & ".join(cells) + r" \\")
        if i in after:
            out.append(r"\midrule")
    out.append(r"\end{longtable}")
    # Outside a float the note block starts a paragraph, so a \textwidth
    # minipage overflows the line by \parindent unless it is suppressed.
    for ln in _minipage(st.notes, st.notes_width):
        if ln.startswith(r"\begin{minipage}"):
            out.append(r"\noindent")
        out.append(ln)
    out.append(r"\endgroup")
    return "\n".join(out) + "\n"


_STRAY_PCT = re.compile(r"(?<!\\)%")


def stray_percents(body: str) -> List[str]:
    """Unescaped ``%`` that is not a whole-line LaTeX comment.

    A bare Python percent format (``f"{x:.1%}"``) emits a literal ``%``, which
    LaTeX reads as a comment: it silently swallows the rest of the source line
    and the numbers vanish from the rendered page while every lexical check on
    the source still passes. That is exactly how the real-pattern coverage
    paragraph lost its rates, sample sizes and coverage in the IR4 package
    (fourth internal review §4.1). Percentages must be written ``\\%``.
    """
    out = []
    for i, line in enumerate(body.splitlines(), 1):
        for m in _STRAY_PCT.finditer(line):
            if line[:m.start()].strip() == "":
                break                      # whole-line comment: intended
            out.append(f"line {i}: {line.strip()[:100]}")
            break
    return out


#: A bare negative decimal: not preceded by a word character (so "4.3-4.4"
#: and "gen-1" are ranges/names, not signs) and not followed by a unit letter
#: (so "\\vspace{-0.5em}" keeps its length).
_BARE_NEG = re.compile(r"(?<![\w${}-])-(\d+\.\d+)(?![0-9]*[A-Za-z])")


def _minus_outside_math(body: str) -> str:
    """Set bare negative decimals in math, leaving math segments untouched."""
    parts, out, in_math = re.split(r"(?<!\\)(\$)", body), [], False
    for tok in parts:
        if tok == "$":
            in_math = not in_math
            out.append(tok)
        else:
            out.append(tok if in_math else _BARE_NEG.sub(r"$-\1$", tok))
    return "".join(out)


def write_tex(path: Path, body: str, *, provenance: Optional[dict] = None,
              allow_inline_percent: bool = False) -> Path:
    """Write a `.tex` fragment with a provenance comment header.

    The header is a LaTeX comment, so it does not render, but it means anyone
    reading the manuscript source can see which script and which commit produced
    the numbers — the traceability R0's evidence map was supposed to provide.

    Refuses to write a fragment containing a mid-line unescaped ``%`` unless
    the caller opts in; see :func:`stray_percents`.
    """
    # A negative number's sign is a MINUS, not a hyphen. Generators wrote
    # plain "-0.030" into cells and notes while the running text set the same
    # quantity as "$-0.030$", so one sign printed as two different characters
    # (seventh review SS13 symbol style; main p.27, ESM tables). Applied only
    # OUTSIDE math (segments between unescaped $), and only to a number that
    # is not part of a range, an identifier or a length.
    body = _minus_outside_math(body)

    # An em dash is "---" in this manuscript. Notes written in the generators
    # spelled some of them "--", which sets an EN dash, and two lengths of dash
    # then appeared in the same table note (seventh review SS13, eighth-round
    # freeze inspection, main p.25). Ranges are written without the spaces
    # ("Interaction--XOR", "0.71--0.80"), so the spaced form is unambiguous.
    body = re.sub(r"(?<=\s)--(?=\s)", "---", body)

    # A control character in a fragment means a LaTeX macro lost its
    # backslash: writing rf"\textit{...}" inside a NON-raw Python string turns
    # the \t into a tab, and the manuscript then renders the literal word
    # "extit". It compiles, no gate notices, and a reader sees it. Cheap to
    # check, and it has happened.
    # A LaTeX line break immediately followed by a letter is the other half of
    # the same escaping bug: writing r"\\\\emph{...}" inside a non-raw Python
    # string DOUBLES the backslash, and the fragment then opens with a line
    # break the surrounding paragraph has no line to end.
    doubled = re.findall(r"\\\\(?=[A-Za-z])", body)
    if doubled:
        raise ValueError(
            f"{path}: a LaTeX macro carries a doubled backslash "
            f"({len(doubled)} occurrence(s)); '\\\\emph' is a line break "
            f"followed by the literal word 'emph', not an emphasis. Write the "
            f"fragment as a raw string, or halve the backslash.")
    lost = sorted({m.group(0) for m in re.finditer(
        r"[\t\r\f\v\b\a]|(?<![\\A-Za-z])(?:extit|extbf|extsc|mph|nderline)"
        r"\{", body)})
    if lost:
        raise ValueError(
            f"{path}: a LaTeX macro looks like it lost its backslash to a "
            f"Python escape (found {lost[:5]}). Write the fragment as a raw "
            f"string, or escape the backslash.")
    if not allow_inline_percent:
        bad = stray_percents(body)
        if bad:
            raise ValueError(
                f"{path}: mid-line unescaped '%' would be read by LaTeX as a "
                f"comment and swallow the rest of the line -- write '\\%'. "
                + "; ".join(bad[:5]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["% GENERATED FILE - do not edit by hand.",
            "% Edit the generator and re-run; the manuscript \\input{}s this file."]
    if provenance:
        for k, v in provenance.items():
            head.append(f"% {k}: {v}")
    path.write_text("\n".join(head) + "\n" + body)
    return path
