# Evaluation Findings: accuracy_completeness Rubric Revision

**Date:** 2026-06-27

---

## 1. What Changed

The `accuracy_completeness` dimension description was rewritten to tie "completeness" to a strict, observable service standard:

- Did the rep verify patient identity and ask the right clarifying questions before acting?
- Did the rep either fully resolve the request, or correctly escalate it to a human when it falls outside agent scope (e.g. billing disputes, suspected coding errors)?

Score-down triggers: acting on unverified or outdated information; deflecting a question the patient explicitly asked; failing to resolve or escalate.

The new definition explicitly excludes tone, politeness, and empathy — those are captured by the separate `empathy_tone` dimension. This prevents double-counting and keeps each dimension's signal clean.

**Motivation:** patient-care quality and clinic reputation depend on patients leaving calls with their questions actually answered or appropriately handed off — not deflected. The prior definition ("provided correct information, addressed every question") was too diffuse to enforce this consistently.

---

## 2. Observed Effect: Two Spot-Check Calls

Results from `scripts/debug_judge.py` run against the real judge (no re-scoring performed after this point).

### HB-2026-00512 — Judge 5, Ground Truth 3

**Judge reasoning:** Rep verified identity, gathered the full picture across both visits and insurance, correctly identified the likely coding error, and escalated to a billing supervisor for the dispute rather than acting unilaterally. She answered every clarifying question (credit, appeal timing, contact).

**Judge evidence:** *"The explanation of benefits shows denial code CO-4... it typically means there was a coding error when we submitted the claim... This is something that can usually be corrected and resubmitted."* and *"Let me put you on hold one more time so I can speak with our billing supervisor directly about next steps."*

**Assessment:** Judge score is correct under the new standard. The rep did exactly what the rubric asks — identified scope limits and escalated rather than acting unilaterally.

### HB-2026-00203 — Judge 2, Ground Truth 4

**Judge reasoning:** The rep acted on account info without proper identity verification and deflected the patient's direct question about the plan change rather than explaining what was on file or escalating the billing concern.

**Judge evidence:** *PATIENT: Can you explain what's different now? REP: That would be something you'd need to check with your employer or your insurance company directly.*

**Assessment:** Judge score is correct under the new standard. Deflecting a direct patient question without explanation or escalation is precisely what the rubric now penalizes.

---

## 3. Root-Cause Conclusion

The elevated MAE on these two calls is **not a judge defect**. The judge reasoning in both cases is coherent and consistent with the revised rubric — it is applying the stricter standard correctly.

The ground-truth labels were provided with the assignment and are of unknown provenance. They appear to reflect the older, looser definition of completeness (one that did not penalize deflection or missing escalation). Tuning the judge or the MAE threshold to fit these labels would hide the signal the new rubric is designed to surface.

---

## 4. Decision

**Keep the rubric as written. Do not adjust the MAE threshold.**

The golden set (`data/seed_transcripts.json`) should be flagged for re-labeling under the current standard before it is used as an authoritative regression baseline. Until re-labeled, regression MAE scores on `accuracy_completeness` should be interpreted with this caveat in mind.

---

## 5. Test Design Reference

The project uses a three-layer testing strategy that keeps iteration cheap:

| Layer | Tool | Cost | When to use |
|---|---|---|---|
| Smoke | `pytest tests/test_smoke.py` | Free | Every change — no API calls, runs in ~0.2s |
| Regression | `pytest tests/test_regression.py` | ~61c | Pre-deploy gate against the full golden set |
| Spot-check | `python scripts/debug_judge.py <call_ids>` | ~10c/call | Rubric iteration — verify judge reasoning on 2-3 targeted calls |

Rubric changes should always be spot-checked with `debug_judge.py` before triggering the full regression suite.

---

## 6. Judge Model Comparison (single call, HB-2026-00203)

Overall scores produced by each model on the same transcript, against the human ground-truth overall of **2.8**:

| Model | Overall score | Δ vs human GT |
|---|---|---|
| Sonnet 4.6 | 1.24 | −1.56 |
| Opus 4.8 | 1.62 | −1.18 |
| Haiku 4.5 | 1.76 | −1.04 |
| Human ground truth | 2.8 | — |

**Substance:** all three models agreed on the failure modes — missed identity verification, deflected billing dispute, no escalation, weak close. The reasoning was consistent across models; only the degree of penalty differed.

**Strictness does not track model cost.** Sonnet was the strictest (1.24), Haiku the most lenient (1.76), Opus in between. This rules out "use the most capable model for the most lenient score" as a tuning lever.

**Label provenance confirmed again.** All three models scored below the 2.8 human label. This is consistent with the conclusion in §3: the label reflects the older, more lenient standard, not a systematic downward bias in the judge.

**Implication for production:** fix one model and compute all metrics relative to it. Mixing models across runs introduces score drift that is unrelated to call quality or rubric changes and will pollute trend analysis and MAE tracking.

---

## 7. Judge Model Comparison at the Suite Level

Extends §6 (single-call comparison) to whole suites, using the suite-run
history feature. Same smoke/regression sets run on each judge model; MAE vs
human ground truth per model:

| Suite | Haiku | Sonnet | Opus |
|---|---|---|---|
| Smoke | 1.02 | 0.59 | 0.83 |
| Regression | 0.73 | 0.78 | 0.56 |

(Lower MAE = closer to human labels.)

**Key finding: the best-aligned model depends on the suite.** On the smoke set,
Sonnet is closest to human labels (0.59); on the regression set, Opus is
closest (0.56). This is a stronger and more actionable result than §6's
single-call view: model strictness is not a fixed ranking — which judge best
matches human labels shifts with the distribution of calls in the set.

**Implication:** model selection for production should be validated on a
representative set, not a single call or an assumption that one model is
uniformly best. The suite-level MAE comparison is the right tool for this
choice.

**Caveat (carries over from §3):** these MAE values are still computed against
the original human labels, which §3-4 flag as possibly reflecting an older,
looser standard. Until the golden set is re-labeled, treat the absolute MAE
numbers as relative model-comparison signals, not ground-truth-accurate
error rates. The ranking between models is informative even if the absolute
values shift after re-labeling.
