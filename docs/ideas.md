# Project Backlog / Future Improvements

## ✅ Completed — code review fixes (this session)

- **Fixed misleading storage test:** `test_storage_upsert_idempotent` was passing for the wrong reason — `count()` is `COUNT(DISTINCT call_id)`, so two INSERTs of the same call still returned 1. Renamed to `test_storage_accumulates_runs_per_call`; now asserts `runs_for_call()` returns 2 rows (history accumulates), not just the distinct-call count.
- **Added smoke tests for new storage functions:** `save_suite_run`, `runs_for_call`, `all_suite_runs` were entirely untested. Added `test_suite_run_roundtrip` (verifies mean/MAE/counts/JSON round-trip), `test_suite_run_all_failed` (edge case: empty `ok_scores` must not divide by zero, stores `None`), and `test_runs_for_call_returns_all_runs` (two saves → two rows, newest first, dimension columns present).
- **Rubric weights now sum to 100%:** changed from 15/25/25/25/15 (= 105%) to 12/25/25/25/13 (= 100%). Prompt Claude reads no longer contains mathematically inconsistent weights.
- **Overview MAE excludes synthetic GT:** both the overall MAE metric and the per-dimension MAE chart now filter out `HB-SYNTH-*` calls. MAE is computed against human seed labels only; a caption says so. Edge case (no seed calls) handled gracefully.
- **Wired `RUBRIC_VERSION` constant (1.1):** `src/rubric.py` now exports `RUBRIC_VERSION = "1.1"` (with a comment: bump on rubric changes). Both `storage.save()` and `storage.save_suite_run()` default to it. Existing rows keep their stored value; only new runs get `"1.1"`. Rubric changes are now traceable in the data.

## ✅ Recently Completed (not previously tracked)

- **Suite-run history (`suite_runs` table):** each Test-runs page run is persisted as a unit with `suite_name`, `model`, `rubric_version`, outcome counts (`n_ok`/`n_failed`/`n_skipped`), `selected_call_ids` (JSON), `failed_call_ids`, `skipped_call_ids`, `mean_overall`, and `mae_vs_gt`. Shown in a Suite run history view on the Test runs page, Smoke and Regression separated into tabs.
- **Cross-model MAE comparison at suite level:** bar chart of MAE vs ground truth by judge model, one bar per model per suite, averaged across runs. See `docs/eval_findings.md` §7 for findings.
- **Per-page judge model selection:** Test runs page has its own model selectbox (overrides sidebar) so a suite can be run on a different model than the rest of the dashboard without switching context.
- **Local-time timestamp display:** all timestamps stored as UTC in the DB; displayed in America/Denver (Mountain Time, DST-aware via `zoneinfo`) everywhere in the UI.
- **FastAPI evaluator endpoint (`POST /evaluate-call`):** same `evaluate()` + `storage.save()` as the UI; persists to the shared DB; results flow into the review queue automatically; returns `run_id`; auto-docs at `/docs`. Two independent processes (Streamlit :8501 + FastAPI :8000) over shared `src/`.
- **CI/CD via GitHub Actions (`.github/workflows/ci.yml`):** runs the free test tier (smoke + api; regression excluded via `pytest.ini`) on every push/PR to `main`; test deps (`pytest`, `httpx`) added to `requirements.txt`; CI status badge in README.

## UI / Dashboard

- ✅ DONE — **Multi-select call evaluation:** implemented as the Test runs page with per-call checkboxes, pre-filled from named default sets, with run-level progress and summary.
- 🟡 PARTIAL — **Filter calls by scenario / call_type:** calls are grouped by call_type on the Test runs page, but there is no filter control to show/hide a call_type group yet. Grouping is visual only.
- 🟡 PARTIAL — **Named eval suites (smoke / regression / targeted):** Smoke and Regression default sets, manual checkbox adjustment, per-run model selection, and suite-run history are all done. **Still pending:** saving a custom set by name (custom set persistence requires a schema addition and is deferred).

## Eval Methodology

- **Re-label the golden set** under the current rubric (ideally two independent reviewers + measure inter-annotator agreement) before treating seed labels as authoritative. See `docs/eval_findings.md` §3–4 for context.
- ✅ DONE — **Per-dimension MAE vs ground truth** in the dashboard — isolate which dimension diverges, instead of only overall MAE. (Implemented; now also excludes synthetic GT — human labels only.)
- 🟡 PARTIAL — **Rubric versioning:** `RUBRIC_VERSION = "1.1"` is now wired into stored runs (bumped manually when rubric changes). Field exists on both `evaluations` and `suite_runs`. **Still pending:** full UI-based version management / rollback (see Production section).
- ✅ DONE — **Score history / run tracking over time:** storage migrated from UPSERT-per-call to per-run INSERT; every evaluation is now a new timestamped row with `run_id`, `model`, `rubric_version`, and `created_at`. Call detail page shows a Run history table (all runs for a call) with a run selector that loads the chosen run's scores, reasoning, and transcript.
- **Suite-run duration & cost tracking:** record per-suite-run wall-clock duration and an estimated cost (from token usage) so history shows e.g. "regression on Opus took 40s, ~$2". Deferred from the suite_runs work to avoid scope creep; add once token-usage logging exists.

## Synthetic Data

- **Append instead of overwrite:** make `generated_transcripts.json` append (with dedupe by `call_id`) instead of overwriting, to accumulate a versioned synthetic test set across generation runs.
- **Circularity caveat:** synthetic ground truth is self-generated by the model — use synthetic data for scenario coverage, human labels for calibration. Do not treat synthetic GT as a substitute for re-labeled seed data.

## Model / Cost

- ✅ DONE — **Document the judge-model comparison** (Haiku 4.5 vs Sonnet 4.6 vs Opus 4.8): documented at both single-call level (`docs/eval_findings.md` §6, HB-2026-00203) and suite level (§7, smoke/regression sets). Key finding: best-aligned model varies by suite — Sonnet lowest MAE on Smoke, Opus lowest on Regression.

## Tooling / Integration (future, evaluate rationality before building)

- **MCP integration (staged, low priority for now):** expose the eval
  project to Claude Code as a tool via Model Context Protocol, so it can
  query runs and assist review instead of reading files manually.
  PREREQUISITE: this depends on the storage-history change (persist each
  judge run + a human-review queue) already listed under Eval Methodology
  — MCP is only useful once there's structured run history to expose.
  Staged plan:
  1. Persist each judge run as a record (run_id, model, scores, reasoning,
     flagged, flag_reason).
  2. Build a human-review queue triggered by: score below threshold,
     critical issue, low judge confidence, or models-disagree.
  3. Local MCP server exposing read/write functions over that data
     (get_flagged_calls, get_call_details, write_human_review,
     update_golden_set, trigger_regression).
  4. Only later: external MCP (GitHub issues, Jira tickets, CI/CD,
     Postgres) — start local/SQLite first.
  Note: captured to avoid losing the idea; assess whether it's actually
  worth building before committing. MCP is technical breadth, not core to
  the eval methodology — prioritize accordingly.

## Human Review — future enhancements (roadmap)

✅ **v1 Human Review Queue is BUILT (this session):** routing on five triggers (low score, dimension fail, `clinical_triage`, `prescription_refill`, identity-verification failure as privacy proxy), pending/resolved workflow with statuses + notes + editing, and judge context (summary + transcript) in each card. The items below are FUTURE extensions beyond v1.

Planned extensions to the Human Review Queue that are NOT being built in v1.
Ordered by priority / dependency.

### 1. Detection flags in the judge (next major step after v1 queue)

v1 routes calls to human review via proxy triggers: low overall score, a
dimension below threshold, `clinical_triage` call_type, or
`identity_verification <= 2` as a privacy proxy. The next step is to have the
judge explicitly detect and return boolean flags stored alongside the
evaluation, so routing is based on real detection — not proxies.

Flags to add to `TranscriptEvaluation`:

| Flag | Meaning | Priority |
|---|---|---|
| `privacy_violation` | Agent disclosed PHI improperly — before verifying identity, or to an unauthorized party | High |
| `identity_bypass_attempt` | Caller tried social engineering: impersonating another patient, or extracting another patient's data without authorization | **Critical — security issue, not caught by quality scoring** |
| `safety_concern` | Clinical / medical risk present in the call | High |

**What this requires:**
- Change the judge prompt to instruct explicit detection of each flag.
- Extend `TranscriptEvaluation` (Pydantic model + DB schema) to store the boolean fields.
- Update review-queue routing to use the flags directly.
- Test on crafted example transcripts designed to trigger each flag.

---

### 2. Healthcare-specific checks to evaluate (prioritized by harm)

These are candidate checks for the judge rubric or dedicated detection passes.
They overlap partially with the existing rubric (identity verification,
accuracy/escalation, protocol adherence); the **new ground** is: explicit
safety/privacy flagging, red-flag symptom detection, out-of-scope medical
advice, voicemail PHI, and social-engineering detection.

**Privacy / HIPAA compliance**
1. PHI disclosed without identity verification (root HIPAA risk).
2. Social engineering / pretexting to obtain another patient's data.
3. Discussing patient data with an unauthorized third party (e.g. relative without consent).
4. Leaving PHI on voicemail / answering machine (common real-world violation).
5. Confirming patient status itself when that confirmation is disclosure.

**Clinical safety (risk to health / life)**
1. Incorrect urgency triage — emergency symptom treated as routine.
2. Missed red-flag symptoms (chest pain, difficulty breathing, stroke signs, suicidal ideation) without escalation.
3. Medical advice beyond agent scope (agent is not a clinician — no diagnosis or prescribing).
4. Medication errors on refill (wrong drug/dose, drug interactions, ignored allergies or contraindications).

**Action correctness (verify the deed, not just the words)**
1. Agent claimed an action (e.g. "appointment scheduled") that did not actually happen — requires deterministic API/DB validation (separate roadmap item).
2. Incorrect insurance / coverage information given (financial harm to patient).
3. Wrong routing or referral.

**Escalation & boundaries**
1. Failed to escalate when required (out-of-scope billing dispute, clinical question).
2. Mishandled escalation or dropped the call.
3. Did not recognize a vulnerable caller (elderly, in distress, language barrier).

**Documentation / audit**
1. Did not capture required patient consent.
2. No audit trail for critical actions.

**Implementation notes:**
- Assess each check for data availability before building — some require crafted
  test transcripts; action-correctness checks require a real scheduling API/DB.
- Build incrementally after the v1 review queue is stable.
- The flag-based detection (§1 above) is the prerequisite for most of §2.

---

## Protocol checklists (per call_type)

**Concept:** replace the holistic `protocol_adherence` score (currently 1–5 judged by gut feel) with explicit per-call-type checklists the judge verifies step by step, scoring from steps completed. More reproducible (raises inter-annotator agreement), more explainable, and leverages a QA-checklist mindset. Ties into the deterministic-validation roadmap item.

Steps have two weights: **REQUIRED** (protocol broken if missing) and **RECOMMENDED** (nice to have, not a failure).

### Checklist: `appointment_scheduling` (drafted, validated)

**REQUIRED steps:**
1. Introduce self (name + org)
2. Verify identity (full name + DOB) before accessing records
3. Understand request (visit type, provider)
4. Verify insurance
5. Check availability and offer concrete slots
6. Confirm slot back to patient (date, time, provider)
7. State next steps (confirmation method)
8. Close politely

**RECOMMENDED steps:**
- Clarify visit type / duration
- Give pre-visit instructions

**Validated against `HB-2026-00147`:** 7/8 required steps met (step 4 — insurance — was missing). Matches the human ground-truth `protocol_adherence` score of 4. Checklist is realistic.

### Score mapping — DECIDED: hybrid critical/required/recommended

**Decision:** use a hybrid model, not a linear fraction.

Steps are divided into three tiers:
- **CRITICAL** — missing one auto-fails the dimension; caps `protocol_adherence` at 1–2 regardless of other steps. Per-call-type examples: identity verification (all types); recognizing red-flag symptoms in `clinical_triage`; verifying patient before discussing medications in `prescription_refill`.
- **REQUIRED** — if criticals pass, the score derives from the fraction of required steps completed (optionally weighted by step importance).
- **RECOMMENDED** — bonus only; never a penalty.

**Rationale:** linear fraction wrongly treats all misses as equal. In healthcare, missing identity verification is categorically worse than skipping a pre-visit reminder — they belong in different buckets. The critical/auto-fail concept mirrors real contact-center and healthcare QA practice and is consistent with the existing review-routing philosophy (`greeting_identity_verification <= 2` already triggers a `privacy_identity_risk` flag).

**Validated on `HB-2026-00147`:** critical step (identity verification) passed; 1 ordinary required step missed (insurance). Score: 4. Matches human ground-truth `protocol_adherence` label of 4. ✓

### Implementation plan (future, own session)

Build ONE checklist (`appointment_scheduling`) end-to-end first:
1. Prompt change — instruct the judge to check each step explicitly
2. Extend `TranscriptEvaluation` — per-step boolean results alongside the existing score
3. Score mapping — implement one of the options above, gate on seed MAE
4. Tests — unit-test score mapping; smoke-test the judge on HB-2026-00147
5. UI — show checklist step results in the Call detail dimension expander

Validate, then replicate to the remaining call types: `clinical_triage`, `prescription_refill`, `billing_inquiry`, `insurance_verification`.

## Generalization / public version (after panel)

- ✅ **Rebranding done:** all original company name / call-ID prefix references replaced with "HealthBridge" / "HB-" across source, tests, docs, and data files. README was already generic.
- After the panel (if invited), make a **public copy** in a separate repo for broader job search: add a live hosted demo (e.g. Streamlit Cloud). The original private repo stays for the panel.

## Tech debt / Minor cleanup

- **Replace deprecated `@app.on_event('startup')` in `src/api.py`** with FastAPI's new lifespan event handler API. Currently works but emits a DeprecationWarning (visible in CI logs). Low priority — not breaking.

## Production / Operability (future)

- **Rubric versioning with UI-based rollback (production operability):** For
  a production deployment, give the system operator visual access to rubric
  versions through the UI — view the current and past rubric versions, switch
  the active version, and roll back to an earlier one. Rationale: if a rubric
  change degrades evaluation quality in production, the operator can quickly
  revert to a known-good rubric without touching code or redeploying — fast
  incident recovery. Requires: storing rubric version texts (not just a
  version label) with timestamps, an "active version" pointer, and UI to
  view/switch/rollback. Also tie each evaluation + suite run to the rubric
  version active at run time (the rubric_version field exists and is now wired
  to `RUBRIC_VERSION` in `src/rubric.py`, bumped manually on rubric changes —
  UI-based management is the remaining step). Same pattern could extend to the
  golden set (versioned labels + re-labeling history) so MAE is always
  interpreted against a known label version.
  NOTE: this is a production-grade feature. For the current local/demo
  project, git already provides version history and rollback for the rubric
  (it's code) — this UI-based operator workflow only earns its complexity once
  there's a real production deployment. Capture as vision; build only when
  prod exists.
