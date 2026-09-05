# Response to the reviewers

**Manuscript** HISC-D-26-01481 · **Journal** Health Information Science and Systems · **Decision** Major Revisions, 19 July 2026

**Submitted as** *Audit-oriented imputation of incomplete tabular health data using statistical-neural interaction networks* · **Revised title** *Evaluating intrinsic and post-hoc audit artifacts for imputation of mixed-type health data*. The title changed because the paper's subject did: it is no longer a proposal for an imputer but an evaluation of the audit artifact that imputer produces, and the result is negative.

We thank both reviewers. The revision that follows is larger than a revision usually is, and it goes in one direction: **on most of these points the reviewers were right, and where the evidence we built to answer them came back against us, the claim was withdrawn rather than defended.** Two of the paper's original headline arguments no longer appear. The paper this letter accompanies reports a mixed evidence profile with an explicitly negative central result; it does not argue that the reviewers' concerns were unfounded.

Paths in the **Evidence** lines are relative to the public repository at {{repo_url}} (release `{{repo_tag}}`, commit `{{repo_commit}}`), which holds the code, the prospectively specified rule documents with their original commit evidence, and — under `evidence/` — the aggregate artifacts every number in this letter is read from. Row-level derived tables for MIMIC-IV and eICU are not there and are not ours to distribute; the Online Resource says which are restricted and how to rebuild them. Every pointer below carries a section or float number and a page number. They are not typed: the manuscript is frozen, and the generator that produces this letter reads each label out of `paperY_main.aux` — the file pdflatex itself wrote for the PDF you receive — so no pointer here can disagree with it. The same generator refuses to emit the letter if any label fails to resolve, which is how two pointers that named sections the manuscript does not have were caught before you saw them.

---

## Reviewer #1

### R1-1 — the λ ablation

> The lambda ablation (Table S2) indicates that a fixed λ=0.1 quantitatively surpasses the learned configuration in MIMIC Accuracy and Macro-F1, which somewhat undermines the argument for learnt λ as an accuracy-neutral advantage. The authors recognize this, but it warrants further elaboration in the main article.

**Response.** We did not elaborate the argument. We withdrew it. On re-examining the implementation we found that the learned gate *cannot* behave the way the original claim required: with λ_h = softplus(θ_h) and θ initialized at zero, the gate enters the objective only through the non-negative prior term and never the prediction path, so every per-head gradient is non-negative and θ can only decrease under weight decay. λ_h is therefore bounded above by its own initialization, ln 2 ≈ {{ln2}}, for the whole of training. A scan of {{lam_models}} trained models found a maximum of {{lam_max}} and {{lam_viol}} violations. There is no accuracy-neutral advantage to defend because there is no free parameter: the quantity we had presented as a learned confidence coefficient is a one-sided, self-attenuating term.

Accordingly, R0's Table S2, its companion figure, and both passages that carried the accuracy-neutrality argument were deleted. The manuscript now states the bound as an optimization invariant of this architecture and says plainly that the gate is not an unconstrained confidence diagnostic. We did not run a finer λ grid; no sensitivity analysis was added, and the letter should not be read as claiming one.

The result is one-sided in a further sense we state in the Online Resource: it shows the gate cannot *rise* above initialization, not that it cannot vary within its range. The verification scan is an implementation consistency check consistent with the derivation, not independent validation of it.

**Where it lands.** Introduction, contribution (v) (`sec:intro`); the neutral description of the coefficient in `sec:sni`; the derivation and its scope in `sec:why`, fourth observation; the invariant and the scan in the Online Resource's λ section.

**Evidence.** `evidence/lambda_check.json` in the repository (the scan over {{lam_models}} trained models, its quantiles and its verdict); the derivation and the invariant are set out in the Online Resource's λ section.

---

### R1-2 — GAIN and MIWAE in Panel A

> The exclusion of deep generative baselines (GAIN, MIWAE) from Panel A of Table 5 should either be rectified or more thoroughly justified beyond the rationale of "table range".

**Response.** Rectified — and we must first correct the record. The original rationale was not merely thin; it was untrue. Tracing the R0 pipeline showed that KNN, MICE, GAIN and MIWAE had never been run for that panel at all. The missing entries were not a presentation decision about table range; the experiments did not exist. We regret the original wording, which described an absence of results as an editorial choice.

The panel is rebuilt with the full method set actually executed. The reviewer should read this as a correction rather than as a restoration: GAIN and MIWAE were never in Panel A, and no earlier results for them were withheld.

**Where it lands.** The downstream table and its note (`tab:impute_predict_expanded`, generated); the corrections paragraph in the Online Resource, which records the original rationale and why it was wrong.

**Evidence.** The corrections audit in the Online Resource, which records what the original justification claimed and what the pipeline actually did; the regenerated table and its note in the revised manuscript.

---

### R1-3 — train/validation/test separation

> Train-validation-test separation is insufficiently described. For downstream evaluation, imputation must be fitted entirely inside each training fold; otherwise, information from test rows can influence the completed table.

**Response.** Accepted, and the audit found the problem to be worse than described. In the R0 downstream panel the missingness was generated on the full table and the whole table was imputed before splitting, so test rows did influence the completed table. The protocol has been rewritten and the panel re-run.

We must be precise about how far the fix goes, because it does not go equally far for every method. Four imputers — mean/mode, KNN, MICE and MissForest — are fitted strictly inside the training fold and applied to the test block. Five — GAIN, MIWAE, HyperImpute, TabCSDI and SNI itself — complete the test block jointly, because these methods impute a table rather than learning a transferable per-column map. That is a property of the method class, not an oversight, and it means the two groups are not exchangeable. The manuscript therefore reports downstream results **within protocol class** and never ranks across the two classes. We do not claim a fold-internal guarantee for all nine.

**Where it lands.** The protocol-class definition and the downstream analysis section of the main text (`sec:downstream`); the per-class table and its note; the Online Resource's protocol-class appendix.

**Evidence.** The downstream protocol and the two protocol classes are specified in the Online Resource; `experiments/t44_downstream.py` in the repository is the code that fits and scores them, and the independence argument is stated in Methods.

---

### R1-4 — the MAR simulator

> Replace the record-index MAR simulator with clinically plausible MAR mechanisms based on observed covariates, workflows, measurement frequency, or patient characteristics.

**Response.** Replaced, not supplemented. We first confirmed the defect mechanically: every R0 MAR mask recorded its driver column as the row identifier, which is a strictly increasing counter created during assembly and carries no clinical meaning. The R0 MAR condition was therefore missingness driven by row order.

The simulator now draws its drivers from observed covariates and patient characteristics, and the mechanism is documented per dataset with the driver columns named. We should be candid about coverage against the four families the reviewer listed: observed covariates and patient characteristics are delivered squarely; workflow and measurement-frequency proxies are represented only where the source tables carry a variable that stands in for them, and we say so rather than implying full coverage of all four.

**Where it lands.** The missingness specification section of the Online Resource (generated), which names the driver columns per dataset; the datasets table note in the main text.

**Evidence.** `configs/missingness.yaml` in the repository is the specification the Online Resource's mechanism section is generated from, verbatim; `data_manifests/masks_sha256_manifest.json` is the checksum manifest the regenerated masks are verified against.

---

### R1-5 — statistics, effect sizes and equivalence

> Apply a more suitable multi-method statistical analysis and present effect sizes and confidence intervals. A non-significant Wilcoxon result in comparison to HyperImpute does not establish equivalence.

**Response.** Accepted on both halves, and the equivalence claim was withdrawn rather than supported. The manuscript no longer asserts equivalence with HyperImpute or with any comparator anywhere.

The inference layer was rebuilt around the unit that is actually independent here. Seeds are the inference unit; within a seed, regime-level cells are nested, and treating them as independent was pseudo-replication. Each axis that carries a claim now reports a signed effect, a seed-level bootstrap interval and an exact test by enumeration over seed-block sign flips, with Holm correction across the family. For behavioral faithfulness on MIMIC, for example, the effect is {{faith_mimic_T}} over {{faith_mimic_n}} seeds, exact p = {{faith_mimic_p}}, Holm-adjusted p = {{faith_mimic_holm}}; on eICU, {{faith_eicu_T}} with exact p = {{faith_eicu_p}}.

Two boundaries that this letter must not blur. First, **we did not run an equivalence test.** There is no TOST, no non-inferiority margin and no equivalence verdict in the manuscript; a failure to reject is reported as a failure to reject. Second, the manuscript does not report Friedman tests, rank correlation effect sizes of the rank-biserial family, or critical-difference diagrams; the multi-method layer we built is the one described above, and we name it precisely so the reviewer is not led to expect a different apparatus.

**Where it lands.** The statistical analysis section (`sec:stats`) and the estimand statement it carries; every axis table's note; the Online Resource's inference appendix.

**Evidence.** `evidence/t_final.json` in the repository is the single source every number in the manuscript is read from, including the estimand text itself; `experiments/t_final.py` is the code that builds it.

---

## Reviewer #2

We are grateful for the opening assessment — that the auditability angle is relevant and reasonably positioned, and that the evaluation is thorough. The six points that follow were the ones that reshaped the paper, and we address them without leaning on that opening.

### R2-1 — the auditability case, and the missing comparison

> The novelty depends on auditability, and that case is not really made. SNI does not improve accuracy (rank 4.27/9, behind MissForest and MIWAE, only tied with HyperImpute). D is basically averaged attention, which is not new. The key missing experiment is a direct comparison: why prefer an intrinsic artifact from a mid-pack imputer over a post-hoc explainer (SHAP or permutation importance) run on a stronger imputer like MissForest? In fact Fig. 4 shows D and SHAP disagreeing.

**Response.** We ran the experiment the reviewer identified as missing, we specified it before running it, and it did not come out in our favor. The paper no longer argues that an intrinsic artifact should be preferred to a post-hoc explainer on a stronger imputer.

The comparison is now the paper's central one, and it is made under equal information. A same-host behavioral probe run on the model's own completed table is compared with the attention readout D on the same host, over {{fair_seeds}} seeds: the effect is {{fair_T}} ({{fair_T_full}} unrounded), with a seed-level bootstrap interval of [{{fair_lo}}, {{fair_hi}}] and an exact two-sided p of {{fair_p}} — which is the attainable floor for this design, {{fair_floor}}, so the design cannot produce a smaller p and the result is reported as indeterminate rather than as a win in either direction. Against the prespecified training-free comparator, D's effect is {{dtap_T}} with interval [{{dtap_lo}}, {{dtap_hi}}], exact p = {{dtap_p}}, {{dtap_neg}} seeds negative.

We also correct a comparison that was unfair in our own favor's opposite direction and had to be withdrawn: the probe's error signal reads values that were withheld from the imputer, which no comparator can read. Where that asymmetry is present the readout is reported as a descriptive positive control, not as a comparison on equal information, and the figure and tables say so in their own notes.

The manuscript reports SNI's imputation rank flatly as mid-field and then states that the paper's claims do not rest on it.

**Where it lands.** The recovery axis and the fair-pair readout in `sec:recovery`; the object-by-axis figure and its caption (`fig:vera`); the unequal-information disclosure on the leakage axis in `sec:leakage`; the Discussion's honest-boundaries subsection.

**Evidence.** `evidence/fair_same_host_recovery.json` and the cell-level `evidence/fair_same_host_recovery_cells.csv` in the repository; `experiments/recompute_fair_pair.py`, which a reader can run against them to reproduce the effect, the interval and the exact p from the cells alone.

---

### R2-2 — D under-validated against a cheap prior

> D is under-validated on its own terms. In the synthetic recovery test (Table 2), "Prior-Only" beats full SNI in two of the three regimes. If a cheap marginal prior does the job, the added value of CPFA for auditing is not clear.

**Response.** We ran this properly and the reviewer's premise was strengthened, not weakened. The cheap marginal prior was promoted from an ablation arm into a named, prespecified comparator carried across all five axes at full hyperparameters, and the outcome is that we do not demonstrate an advantage for the attention readout over it: the effect is {{dtap_T}}, interval [{{dtap_lo}}, {{dtap_hi}}], exact p = {{dtap_p}}.

The prior is not incidental to the model, and we report that too: removing the association prior entirely collapses behavioral faithfulness, to {{np_mimic_T}} on MIMIC (exact p = {{np_mimic_p}}) and {{np_eicu_T}} on eICU. So the prior carries real weight in the architecture — which is precisely why the absence of a demonstrated advantage for the attention readout *over the prior alone* is the uncomfortable result rather than a trivial one.

We want to be exact about what that does and does not say. It is not a demonstration that the prior is superior, and the manuscript does not claim the reviewer was mistaken. It is an absence of demonstrated advantage for the more expensive artifact, on this evidence, at this power — which is the conclusion the reviewer's original observation pointed at.

**Where it lands.** The comparator definition in `sec:objects`; the recovery axis table and its note; the profile figure's recovery row.

**Evidence.** The comparator's specification document, with its original commit hash and attestation, in `prereg_archive/`; the recovery cells in `evidence/fair_same_host_recovery_cells.csv` and the recomputation script in `experiments/`.

---

### R2-3 — the proxy-injection case

> The proxy-injection case is anecdotal: one duplicated column, no permutation null, no random-proxy baseline. A duplicated column will obviously attract attention, so this shows sensitvity to an easy case, not detection of realistic leakage.

**Response.** We built every piece of machinery the reviewer asked for — multiple seeds, a permutation null, and a random-proxy control — and the machinery confirmed the diagnosis rather than answering it. The observed null rate is {{null_detected}} of {{null_n}} injections, reported in the manuscript as a calibration diagnostic and not as a detection success. We attach no fixed-alpha test to these counts: the nominal level is itself estimated from a finite calibration sample, so a binomial test against it would be testing an estimate against itself.

The reviewer's central sentence — that a duplicated column demonstrates sensitivity to an easy case rather than detection of realistic leakage — is now supported by our own controls rather than contradicted by them. The leakage axis is reported with that reading, and the challenge design, including the discrepancy control, is documented so that the easy case and the realistic case are distinguished in the reporting rather than pooled.

**Where it lands.** The leakage axis section (`sec:leakage`), including the two estimands and the discrepancy control; the leakage figure and its caption; the challenge-set specification in the Online Resource.

**Evidence.** `evidence/t42_summary.json` in the repository carries the per-object, per-condition counts and the observed null rates; `experiments/t42_leakage.py` is the code that produces them. The null rates are also printed in the figure.

---

### R2-4 — D's stability, and only two datasets

> D's stability drops exactly where auditing matters most (ρ from 0.951 on MIMIC to 0.633 on eICU), and it is only tested on two datasets. Real tables are usually wider, so this is not convincing.

**Response.** The primary response is that the claim this item attacked is gone: the manuscript no longer offers D's cross-seed stability as positive evidence for the artifact, so there is no durability argument left to defend.

We must be careful about one thing here, and we state it rather than let a number do quiet work. The rebuilt eICU table is not the R0 eICU table — its column set differs — so the corrected stability figures are **not** comparable with the R0 pair the reviewer quotes, and we do not present them as an improvement on those numbers. Where stability is reported it is reported against the band measured under information symmetry, which for these tables is {{band_mimic}} and {{band_eicu}}.

The second half of the item we do not answer. The evaluation remains on two clinical tables, and wider tables were not tested. That limitation is stated in the manuscript's boundaries subsection in the reviewer's own terms rather than softened.

**Where it lands.** The stability axis section; the honest-boundaries subsection of the Discussion, which carries the two-table limitation; the axis table notes that name the band and its caliber.

**Evidence.** The stability readouts and both host bands are printed in the Online Resource's stability section and read from `evidence/t_final.json`, not restated.

---

### R2-5 — λ_h weakly supported

> λh is weakly supported: it sits in 0.63-0.68 across all six datasets, and Table S2 shows the learned value is not the best on several metrics. A nearly constant coefficient looks more like low sensitivity than a useful diagnostic. The authors should show λh actually reacts under controlled perturbations, or tone down the claim.

**Response.** The reviewer offered two options. We took the second, and further than it was offered: the diagnostic claim was not toned down, it was withdrawn, and it was replaced by a derivation showing the implemented gate is structurally incapable of the behavior the original claim required. The reviewer's suspicion that near-constancy indicated low sensitivity rather than a useful signal turns out to be correct at the level of the code — see R1-1, which this item shares a target with.

We did not run a controlled-perturbation experiment on λ_h. Across the scanned models the coefficient's median is {{lam_median}} with a range of [{{lam_min}}, {{lam_max}}], all below the ln 2 bound; the manuscript reports this as the signature of a bounded, self-attenuating term rather than as evidence of a diagnostic.

**Where it lands.** Introduction, contribution (v); `sec:why`, fourth observation; the λ section of the Online Resource.

**Evidence.** `evidence/lambda_check.json` in the repository; the runtime assertion that enforces the bound during training ships with the code in `sni/`.

---

### R2-6 — power, dropped panels, and cost

> Some caveats on the stats and cost: with only 12 paired settings the tests are underpowered, so "no difference from HyperImpute" means failure to reject, not equivalence. The MNAR panels drop two baselines, and SNI is also slower than the more accurate MissForest/MICE, so the reader pays more for less fidelity.

**Response.** All three clauses are accepted; none is rebutted.

*Power and equivalence.* The equivalence reading was withdrawn. Where a test does not reject, the manuscript says so and stops. We note that the design's exact enumeration has an attainable floor — for the five-seed central comparison it is {{fair_floor}} — so for that comparison no amount of care in the analysis could produce a smaller p; the limit is the number of independent units, and we report the result as indeterminate.

*Dropped panels.* The corrected benchmark reports the method set that was actually run per condition, and where a method is absent the table says which and why rather than leaving a gap.

*Cost.* Accepted and reported as a cost, on its own axis, with marginal and total accounting kept separate and never offset against the validity axes. The reviewer's summary — more cost for less fidelity — is the reading the cost axis supports, and the manuscript does not argue otherwise.

**Where it lands.** The statistical analysis section and every axis note; the cost axis section and its table; the boundaries subsection.

**Evidence.** `evidence/t_final.json` for every figure quoted; the cost accounting and its window annotations are in the Online Resource's cost section.

---

## The scale of the revision

We summarize what changed, because the list is long and a reader is entitled to see it in one place. This is a factual account, not a claim of merit.

- **Inference rebuilt on the correct unit.** Seeds are the independent unit; regime-level cells nested within a seed are not. Every axis that carries a claim reports a signed effect, a seed-level bootstrap interval and an exact enumeration test, with Holm correction across the family.
- **Information symmetry corrected on three axes.** Where a readout's error signal reads values withheld from the imputer, comparators cannot see the same information. Those comparisons are relabelled as descriptive positive controls and a fair same-host pair was constructed for the central comparison.
- **A prespecified training-free comparator.** The cheap prior became a named comparator carried across all five axes, at full hyperparameters.
- **A lineage audit of that comparator family**, so that what each arm consumes is documented rather than assumed.
- **A baseline family rebuilt** and the downstream protocol re-run under fold-internal fitting, reported within protocol class.
- **Every reference verified** against an authoritative record, with per-citation support records rather than per-reference ones.
- **Fourteen rounds of institutional internal review**, each closed against a machine-checkable gate set that runs inside the delivered evidence package and reports the same verdict in four time zones.

Two of the paper's original headline claims did not survive this process. We report the outcome as it stands, including a central comparison that is indeterminate at its own attainable floor.

---

## A note on what this letter is

Every number above is read at render time from the artifact that produced it — the same single evidence source the manuscript reads — and none is typed into this document. The reviewers' own figures, quoted in the blockquotes, are the exception: those are quotations and are reproduced verbatim. Before submission this letter is regenerated and compared character by character against the file being sent, so that a number cannot drift between the manuscript and the response to the people who asked for it.
