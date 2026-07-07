# Human Annotation Guidelines (v1.0)

*Reflects rubric v1.1 (`src/rubric.py`). For labeling GPT-generated calls on the Labeling page.*

> **Disclaimer:** These guidelines are intentionally simplified for demonstration purposes. They are not a substitute for a full clinical, legal, or compliance-grade QA program. Reviewers score only observable behaviors — what was actually said or done in the transcript — not assumed intent, unstated policy, or information the call never gave a chance to appear. When the transcript doesn't show something, that is evidence of absence only if the rubric says it was required at that point in the call.

---

## Golden Rules

1. **Score only what is observable.** Judge the transcript in front of you, not what a rep "probably" would have done off-screen, and not your own assumptions about clinic policy.
2. **Unknown policy → "Cannot determine," not "Missing."** If a call never reaches the point where a policy detail (e.g. a specific escalation path) would surface, do not deduct for its absence. A gap only counts against a dimension if the rubric for that dimension required it to be observable by that point in the call.

---

## Per-Dimension Guidelines

### 1. Greeting & Identity Verification — weight 12%

> **Rule (verbatim, rubric v1.1):** Rep introduced themselves, verified patient identity (full name AND date of birth — both are required) before taking any account action, and stated the purpose of the call. Failure to collect date of birth, regardless of how well the rest of the call went, must result in a score of 1 or 2. Partial verification (name only, or DOB collected after account action) scores no higher than 2.

| Score | Anchor |
|---|---|
| 5 | Introduces self, collects full name **and** DOB before any account action, states the purpose of the call. |
| 4 | Same as 5, but the greeting is perfunctory or purpose-of-call is stated a beat late. |
| 3 | Name and DOB both collected before any account action, but rep never introduces themselves or never states the call's purpose. |
| 2 | **Hard ceiling** — DOB collected only *after* an account action already occurred, or only name (no DOB) was ever verified. |
| 1 | No identity verification attempted before an account action, or the step is skipped entirely. |

### 2. Empathy & Tone — weight 25%

> **Rule (verbatim):** Acknowledged the patient's concerns, used empathetic language, and maintained a professional, warm tone throughout the call.

| Score | Anchor |
|---|---|
| 5 | Explicitly acknowledges the patient's concern or emotion; warm and professional throughout. |
| 4 | Warm and professional overall; one missed chance to acknowledge frustration or concern. |
| 3 | Professional but flat — transactional, no explicit acknowledgment of how the patient feels. |
| 2 | At least one noticeably curt, impatient, or dismissive moment. |
| 1 | Rude or dismissive, or repeatedly ignores clear patient distress. |

### 3. Accuracy & Completeness — weight 25%

> **Rule (verbatim):** Did the rep gather the information needed before acting — verifying the patient and asking the right clarifying questions — and either fully resolve the request or correctly escalate it to a human when it falls outside what the agent should handle alone (e.g. a billing dispute or a suspected coding error)? Score down for acting on unverified or outdated info, deflecting a question the patient actually asked, or failing to resolve or escalate. Judge only what information was gathered and whether the task was completed or properly handed off — do NOT score tone, politeness, or empathy here. When an error occurs and is then corrected, distinguish who caught it: a rep who proactively verifies the currency of data and catches their own mistake demonstrates strong completeness (full credit); a correction made only after the patient objects is acceptable but not exemplary — it should score about one point lower — especially when the rep could have confirmed the data's currency before acting. Do not penalize the rep for information the patient never provided that could not reasonably have been on file.

| Score | Anchor |
|---|---|
| 5 | Gathers everything needed, resolves or correctly escalates; if data was stale/wrong, the **rep** catches and corrects it proactively. |
| 4 | Resolves or escalates correctly, but a data error is only caught **after the patient objects** — apply the "one point lower" rule here. |
| 3 | Resolves the request and gathers most needed info, but skips one clarifying question that didn't change the outcome. |
| 2 | Acts on unverified or outdated information, or deflects a question the patient explicitly asked, without escalating. |
| 1 | Fails to resolve or escalate an out-of-scope request; multiple deflected or unresolved questions. |

**Do not** let tone, politeness, or warmth influence this score — that belongs entirely to Empathy & Tone. Do not penalize the rep for information the patient never volunteered that couldn't reasonably have been on file already.

### 4. Protocol Adherence — weight 25%

> **Rule (verbatim):** Followed the appropriate workflow for this call type: identity before any account action; insurance verified before confirming an appointment; clinical concerns triaged appropriately; billing disputes documented.

| Score | Anchor |
|---|---|
| 5 | Every required step for this call type (see table below) happens, in the correct order. |
| 4 | Correct order followed; one non-critical step is slightly out of ideal sequence with no negative consequence. |
| 3 | One required step is missed or out of order, with no safety or compliance consequence. |
| 2 | A required step for this call type is skipped entirely (e.g. insurance not verified before confirming an appointment). |
| 1 | Multiple required steps skipped, or a clinical/billing call handled with no triage/documentation at all. |

### 5. Closing & Next Steps — weight 13%

> **Rule (verbatim):** Summarized the actions taken, confirmed concrete next steps (dates, amounts, follow-ups), and closed the call professionally.

| Score | Anchor |
|---|---|
| 5 | Clear summary of what was done; concrete next steps with specific dates/amounts; professional close. |
| 4 | Summary and next steps given, but one is vague (e.g. "we'll follow up soon," no date). |
| 3 | No explicit summary, but the call still ends professionally and the patient knows what to expect. |
| 2 | Call ends abruptly — no summary, no next steps. |
| 1 | No closing at all, or next steps contradict what was actually discussed. |

---

## Per-Call-Type Expectations

"Required" items missing at the point they should occur justify a score deduction under the relevant dimension. "Preferred" items are best practice but their absence alone should not sink a score below 3–4 on their own. "Optional / later" items should **never** be scored down — they're explicitly allowed to happen outside this call.

| Call type | Required | Preferred | Optional / later |
|---|---|---|---|
| **appointment_scheduling** | Full name + DOB verified before any scheduling action; appointment date/time/provider confirmed back to the patient | Insurance verified before confirming the appointment; reason for visit noted | Address — optional, may be collected via intake forms; accommodation/interpreter needs |
| **billing_inquiry** | Full name + DOB verified before discussing the account or charges; dispute documented if the patient contests a charge | Member ID confirmed — preferred; may be unavailable for new patients; charge explained in plain language | Payment-plan details — can be handled in a follow-up call |
| **clinical_triage** | Full name + DOB verified; symptoms triaged and escalated per severity; **any time-critical red-flag symptom (see below) escalated immediately — not scheduled as a routine appointment** | Symptom onset/duration noted | Non-urgent lifestyle or wellness questions — can be deferred to the visit |
| **insurance_verification** | Full name + DOB verified; current provider and coverage confirmed before any account action that depends on it | Member ID confirmed — preferred; may be unavailable for new patients; copay/coverage details communicated | Address — optional, may be collected via intake forms |
| **prescription_refill** | Full name + DOB verified; prescriber's refill authorization confirmed before submitting to the pharmacy; **self-adjusted dosing, early run-out, or worsening symptoms routed to provider review — not processed as a routine refill** | Last-fill date confirmed; pharmacy on file confirmed | Dosage/side-effect counseling — can be deferred to the pharmacist |

### Safety-Critical Escalation Rules

These two rules override the normal Required/Preferred framing above — get them wrong and `protocol_adherence` (and likely `accuracy_completeness`) should score 1–2 regardless of how well the rest of the call went.

- **clinical_triage:** Time-critical symptoms — **stroke signs, chest pain with shortness of breath, suicidal ideation** — require immediate escalation. Scheduling a routine appointment in response to any of these is **not** an appropriate resolution, no matter how politely or efficiently it's done.
- **prescription_refill:** Self-adjusted dosing, an early run-out, or worsening symptoms mentioned during the call must be routed to provider review — processing these as a routine refill (even a fast, friendly one) is a protocol failure, not a completeness bonus.

---

## Evidence Notes Format

The Labeling page captures evidence as three fields — **Observed** (required), **Concern** (optional), **Impact** (optional) — combined on save into a single `reviewer_notes` string, one line per section, empty optional sections omitted:

```
Observed: <what actually happened, ideally with timestamps>
Concern: <gap or issue noticed, if any>
Impact: <why it matters for the patient or clinic, if any>
```

A clean call legitimately has no Concern or Impact — leave those blank rather than writing filler text.

**Worked example** (`appointment_scheduling`):

```
Observed: Rep verified full name and DOB at 00:12 before pulling up the chart, then offered two appointment slots.
Concern: The confirmed appointment date was never restated back to the patient before the call ended.
Impact: The patient may leave the call unsure of the exact date and miss the appointment.
```

---

## Ambiguous Cases Log

Fill this in during labeling whenever a call doesn't clearly map to a rule above. One row per open question — resolve it, then keep the row as precedent for future labelers.

| call_id | dimension | question | resolution |
|---|---|---|---|
| | | | |
