"""Disclosure sentences that must read identically wherever they appear.

The information-symmetry disclosure (rules:
`docs/T61_information_symmetry_rules.md` §6; addenda on
`docs/T4F_presentation_rule.md` and `docs/T42_leakage_rules.md`, 2026-08-29)
has to appear on the recovery table, the leakage table, the stability table,
the leakage figure and in the prose of both axes. Writing it once here is the
first author's instruction -- "措辞与 recovery 侧同源,一次写定" -- and it is
also the only way a reader can see that the same fact is being stated in every
place rather than six differently-hedged versions of it.

Generators import these constants; they do not paraphrase them.
"""
from __future__ import annotations

#: The core fact, in the form the recovery and leakage tables both carry.
INFO_ASYMMETRY = (
    r"\textit{Information asymmetry (disclosed):} the same-host permutation "
    r"readout is not computed under the same information as the objects it is "
    r"compared with. Its error signal is measured against the values withheld "
    r"from the imputer, whereas every other object here derives its values "
    r"from the masked table or from an imputer's own completion --- including "
    r"the permutation readout of the comparison host, which is scored against "
    r"that host's own completed values. The asymmetry runs in the direction "
    r"that favors the same-host readout.")

#: The leakage axis adds what the asymmetry forbids being said.
INFO_ASYMMETRY_LEAKAGE = INFO_ASYMMETRY + (
    r" The attention matrix's 0/6 and the same-host readout's 6/6 on the "
    r"interaction class are therefore not stated here as a comparison between "
    r"objects with equal information.")

#: The recovery axis prints BOTH same-host calibers as separate rows, so an
#: unqualified disclosure lands on the symmetric row too and says the false
#: thing about it (eighth review P1-1). Verbatim from that review's SS8.3.
INFO_ASYMMETRY_RECOVERY = (
    r"\textit{Information asymmetry, archived row only:} the archived "
    r"same-host permutation readout used an error signal computed against "
    r"values withheld from the imputer. The symmetric row instead uses the "
    r"same host's own completed table and is the equal-information comparison "
    r"used for the current result.")

#: The stability axis, where the statistic is the matrix's stability rather
#: than its accuracy, so the asymmetry bites less -- stated in that weaker form
#: rather than dropped.
INFO_ASYMMETRY_STABILITY = (
    r"\textit{Information asymmetry, archived row only:} the archived "
    r"same-host permutation row is produced by an ablation whose error signal "
    r"is measured against the values withheld from the imputer, which is not "
    r"true of the objects it is compared with. The symmetric row instead "
    r"takes that signal from the host's own completed table and is the "
    r"equal-information reading. The statistic in this table is the matrix's "
    r"cross-seed stability rather than its accuracy, so the asymmetry bears "
    r"on it less directly than on the recovery and leakage axes; it is "
    r"disclosed here for the same reason.")

#: What the faithfulness reference is, said in the terms a reader needs.
REFERENCE_SEMANTICS = (
    r"The behavioral reference is host behavior measured \emph{against the "
    r"withheld values}: each entry is the degradation of the host's error on "
    r"the masked cells, scored against those cells' true values, when an input "
    r"column is permuted. It is a privileged measurement by construction, "
    r"which is what a reference is for; every object on this axis is scored "
    r"against the same one, so no object is advantaged over another by it.")


def all_disclosures() -> dict:
    """For the single-facts gate and the package README."""
    return {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, str)}
