# Cover letter — revised manuscript HISC-D-26-01481

**To** the Editors, *Health Information Science and Systems*

**Manuscript** HISC-D-26-01481 · **Decision** Major Revisions, 19 July 2026

**Submitted as** *Audit-oriented imputation of incomplete tabular health data using statistical-neural interaction networks*

**Revised title** *Evaluating intrinsic and post-hoc audit artifacts for imputation of mixed-type health data*

---

Dear Editors,

Please find enclosed our revised manuscript, together with a point-by-point response to both reviewers.

We should say at the outset what kind of revision this is, because it is not the usual kind. The reviewers' central objections were, in our judgment, correct. Acting on them required building measurements we had not previously made — and several of those measurements came back against the paper's own claims. We have withdrawn those claims rather than defended them. Two of the original headline arguments no longer appear anywhere in the manuscript, and the paper's central result is now explicitly negative: under prespecified analysis, the aggregated attention readout of our own imputer did not demonstrate an audit advantage over a training-free comparator that costs a fraction of a second to compute.

The title changed for the same reason. The paper is no longer a proposal for an imputer; it is an evaluation of the audit artifact that imputer produces, and it reports that the artifact did not earn the role we had claimed for it.

## What the revision contains

The manuscript now presents **VERA**, a five-axis evaluation protocol (structural recovery, stability alignment, behavioral faithfulness, leakage-risk discrimination, resource cost) whose per-axis decision rule was committed to version control before the corresponding measurement existed, and **TAP**, a prespecified training-free comparator. The protocol is applied first to our own method. Its structure is set out in Fig. 1 (p. 6) and `sec:vera`.

The two reviewer objections that drove the largest changes were R1-1 on the learned gate and R2-2 on the fairness of the recovery comparison. In both cases the work we did to answer the reviewer **strengthened the reviewer's premise rather than ours**: the gate turned out to be bounded above by its own initialization, {{ln2}}, as an optimization invariant of the architecture rather than a learned quantity, so the accuracy-neutrality argument it supported was deleted; and recomputing the recovery comparison under equal information left the difference inconclusive at the design's attainable floor. The full account is in the response letter.

Behavioral faithfulness, the axis the artifact exists for, is reported as measured: the seed-block effect is negative on both tables and does not survive multiplicity correction, which we state as a persistent direction with no evidence of advantage rather than as a demonstrated deficit. Where a reviewer's suggestion could not be carried out as asked, we say so plainly instead of implying that it was: we did not restore the two deep generative baselines to the affected panel, and we explain in the response letter why the original justification for their absence was factually wrong and what we did instead.

Nothing in the paper is a claim we could not check. Every verdict is computed by code from a single evidence source and never typed by hand; the decision rules, their original commit hashes and their zero-artifact attestations travel with the code.

## Editorial independence

As disclosed in our original cover letter, one of the co-authors, Prof. Qun Jin, holds an editorial role with *Health Information Science and Systems*. He has had no involvement in the handling or review of this submission, and we understand the manuscript is being processed independently of him. We restate the disclosure here so that it is on the record for this revision as well as for the original submission. In accordance with the journal's policy, the manuscript's Competing interests declaration reads that the authors declare no competing interests; we are of course happy to adjust that declaration if the editorial office would prefer the relationship stated in the article itself.

## Data and code

The complete evaluation code, the missingness simulator, the imputer under test, and every prospectively specified decision-rule document with its original commit evidence are publicly available at {{repo_url}} (release `{{repo_tag}}`, commit `{{repo_commit}}`), under the MIT license.

Restricted derived tables from MIMIC-IV and eICU are **not** distributed: they are accessible through PhysioNet under data use agreements and require credentialed access. We provide the preprocessing code and feature definitions needed to reconstruct them from an authorized download. Trained model weights and the frozen mask files are likewise not distributed; the masks are reproducible from the released generation configuration and seeds, and verifiable against the released SHA-256 manifest. The private development history is retained and available to the editorial office on request.

## Enclosed

- the revised manuscript ({{main_pages}} pages) and Online Resource 1 ({{esm_pages}} pages);
- a point-by-point response to both reviewers, with the section and page reference for every change resolved against the frozen manuscript.

We are grateful to both reviewers. The revision is long because their objections were substantive, and the paper is more honest than the one they read.

Yours sincerely,

Ou Deng, on behalf of all authors
Graduate School of Human Sciences, Waseda University
dengou@toki.waseda.jp
