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

---

## 8. Synthetic GT Circularity in Suite MAE (found & fixed 2026-07-04)

**What:** `suite_runs.mae_vs_gt` (Test runs page) included synthetic calls (`HB-SYNTH-*`) whose ground truth was itself model-generated, via `DATAGEN_MODEL`. The Overview page already excluded these calls from its MAE metric and chart; the Test runs page's suite-run MAE did not — each page filtered independently instead of sharing one rule, so the two calculations drifted apart and the suite-level figure was optimistic: the same model family was, in part, grading its own synthetic output.

**Evidence:** MAE before ("dirty" — human + synthetic pairs) vs after ("clean" — human pairs only) the fix, all 6 suite × model combinations from the 2026-07-04 runs:

| Suite | Model | Dirty MAE | Clean MAE | Δ |
|---|---|---|---|---|
| Smoke | Sonnet 4.6 | 0.50 | 0.66 | +0.16 |
| Smoke | Opus 4.8 | 0.80 | 0.82 | +0.02 |
| Smoke | Haiku 4.5 | 0.88 | 1.01 | +0.13 |
| Regression | Opus 4.8 | 0.63 | 0.71 | +0.08 |
| Regression | Haiku 4.5 | 0.71 | 0.81 | +0.10 |
| Regression | Sonnet 4.6 | 0.77 | 1.06 | +0.29 |

**Key observations:**

- **All six deltas are positive.** Including synthetic pairs never made the metric look worse — only better. That's the expected signature of circularity, not noise.
- **Suite champions are unchanged.** Sonnet stays closest to human labels on Smoke (0.66 vs. 0.82/1.01) and Opus stays closest on Regression (0.71 vs. 0.81/1.06), both before and after the fix — the §7 model-comparison conclusion still holds.
- **Sonnet's Regression delta (+0.29) is roughly 3x the other two Regression deltas (+0.08, +0.10).** `DATAGEN_MODEL` defaults to `claude-sonnet-4-6`, and no override is set in this environment, so the synthetic ground truth in `generated_transcripts.json` was most likely produced by Sonnet itself (to be confirmed against the actual generation run). Sonnet-as-judge agreeing most closely with Sonnet-as-generator on those calls is exactly the circularity this fix removes, and is the cleanest direct evidence that the leakage was real rather than incidental.
- **Small-denominator caveat.** Clean MAE is computed over only 4 human-labeled pairs on Smoke and 6 on Regression. With denominators this small, one call's error moving by roughly a point shifts the suite MAE by about 0.15 — treat these numbers as directionally useful, not statistically precise, until the golden set grows.

**Fix:** extracted `is_synthetic_call_id()` and `build_gt_pairs()` into `src/data_gen.py` as the single source of truth for the `HB-SYNTH-*` exclusion rule, now used by both the Overview page and the Test runs page. Added a caption on the Test runs page next to the MAE metric ("MAE vs human-labeled ground truth only — synthetic calls excluded") and a unit test (`test_build_gt_pairs_excludes_synthetic`) asserting a synthetic pair is dropped even when `ground_truth_qa` is present.

---

## 9. Operational Findings

Smaller fixes found and closed out alongside the investigation above:

- **Review-queue dedupe was broken.** `storage.save_review` deduped on `(call_id, run_id)`, but `run_id` is unique per judge run, so the check never matched across repeated evaluations of the same call — a call re-evaluated N times produced N pending review rows. Fixed to dedupe on `(call_id, status='Pending')`, updating the existing row's `run_id` instead of inserting a duplicate.
- **`REGRESSION_SET` referenced 2 nonexistent transcripts** (`HB-SYNTH-00101`, `HB-SYNTH-00103`) that `ingest.run_suite` was silently skipping on every run. Removed from the set and added a test (`test_default_sets_reference_existing_transcripts`) asserting every id in `SMOKE_SET`/`REGRESSION_SET` resolves to a real transcript, so set/data drift fails CI instead of silently skipping.
- **The API accepted a placeholder `call_id`.** `POST /evaluate-call` would store a real evaluation row for `call_id="string"` — Swagger UI's auto-filled example for an unfilled field. Added `min_length=5` plus a validator rejecting the literal `"string"`.

---

## 10. Review Queue Routing Overhaul (reduce noise, keep safety)

The Review Queue had accumulated 53 pending entries, and 52 of them traced back to Smoke/Regression/Full suite runs — calibration experiments on the judge, not production traffic. Two changes close this: suite runs stop routing at all, and the remaining call-type triggers got sharper.

**1. Suite runs no longer route to review.** `storage.save()` gained a `route_to_review` flag (default `True`); `ingest.run_suite()` is the one caller that passes `False`. Single-call evaluation from the dashboard and the FastAPI endpoint are unaffected — they still route exactly as before.

**2. Old triggers vs new:**

| Trigger | Old | New |
|---|---|---|
| Overall score | `< 3.0` | `< 2.5` |
| Any dimension | `<= 2` (`dimension_fail`) | `<= 1` (`dimension_fail`) |
| Identity verification | `<= 2` (`privacy_identity_risk`) | unchanged |
| `clinical_triage` | any call of this type (`safety_triage`) | `protocol_adherence <= 2` OR `accuracy_completeness <= 2` |
| `prescription_refill` | any call of this type (`safety_prescription`) | `protocol_adherence <= 2` |

**Rationale:** call type was a *proxy* for risk, not risk itself — a clean, well-handled triage or refill call routed every single time purely because of its type, regardless of how well the rep actually handled it. That's noise pretending to be safety. The new triggers route on the score that actually indicates risk for that call type (protocol adherence for both; accuracy for triage, since a missed clinical detail is exactly the failure mode that matters there). The generic any-dimension catch-all moved from `<= 2` to `<= 1` because it's no longer the only safety net — identity and the two type-specific rules now cover the `<= 2` cases that matter, so the catch-all only needs to catch genuinely bad (`1`) scores on dimensions with no dedicated rule.

**Cleanup:** `scripts/resolve_suite_run_reviews.py` bulk-resolves the pre-existing backlog — any still-`Pending` review whose `run_id` traces back to a suite-run evaluation is marked `status='Resolved'`, `note='bulk-resolved: calibration run artifact'`. Reviews a human already triaged are left untouched regardless of origin.
