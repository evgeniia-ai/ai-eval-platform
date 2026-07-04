"""Synthetic transcript generation via a scenario matrix of call types, quality levels, and edge cases.

A matrix of (call_type, quality target, scenario) specs that
deliberately covers edge cases — missing identity verification, angry patients,
wrong-order protocol, clinical red flags, etc. Each spec is realized by Claude into
a realistic timestamped transcript plus human-style ground-truth QA scores, so the
generated set can also serve as an eval set for the QA engine itself.
"""

from __future__ import annotations

import json
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from . import config
from .models import CallType, Score, Utterance
from .rubric import rubric_for_prompt

SYNTHETIC_ID_PREFIX = "HB-SYNTH"


def is_synthetic_call_id(call_id: str) -> bool:
    """True if `call_id` was minted by this module — synthetic ground truth, not human."""
    return call_id.startswith(SYNTHETIC_ID_PREFIX)


def build_gt_pairs(
    results: list[tuple[str, float, float | None]]
) -> list[tuple[float, float]]:
    """Filter (call_id, predicted, ground_truth) triples down to human-labeled pairs.

    Synthetic ground truth is excluded to avoid circular calibration — the same
    model family would be grading its own generated output.
    """
    return [
        (predicted, gt)
        for call_id, predicted, gt in results
        if gt is not None and not is_synthetic_call_id(call_id)
    ]


class GeneratedGroundTruth(BaseModel):
    greeting_identity_verification: Score
    empathy_tone: Score
    accuracy_completeness: Score
    protocol_adherence: Score
    closing_next_steps: Score
    reviewer_notes: str


class GeneratedCall(BaseModel):
    duration_seconds: int = Field(description="Plausible call length, 120-900 seconds.")
    patient_satisfaction_score: float = Field(description="1.0-5.0, consistent with the call.")
    transcript: list[Utterance] = Field(description="8-16 realistic timestamped utterances.")
    ground_truth_qa: GeneratedGroundTruth


# A scenario matrix designed for coverage and edge cases, not just happy paths.
SPECS: list[dict] = [
    {"call_type": "appointment_scheduling", "quality": "excellent", "scenario": "routine follow-up, textbook handling"},
    {"call_type": "appointment_scheduling", "quality": "poor", "scenario": "rep skips identity verification and books anyway"},
    {"call_type": "appointment_scheduling", "quality": "medium", "scenario": "rep verifies identity but forgets to confirm the date back"},
    {"call_type": "insurance_verification", "quality": "good", "scenario": "patient confused about a plan change; rep explains patiently"},
    {"call_type": "insurance_verification", "quality": "poor", "scenario": "rep is dismissive and verifies insurance before identity (wrong order)"},
    {"call_type": "insurance_verification", "quality": "excellent", "scenario": "complex coordination of benefits, handled thoroughly"},
    {"call_type": "billing_inquiry", "quality": "excellent", "scenario": "upset patient de-escalated, payment plan offered proactively"},
    {"call_type": "billing_inquiry", "quality": "medium", "scenario": "correct info but rushed; no summary of next steps"},
    {"call_type": "billing_inquiry", "quality": "poor", "scenario": "rep gives incorrect balance and never verifies identity"},
    {"call_type": "prescription_refill", "quality": "good", "scenario": "refill on a controlled substance; rep follows protocol"},
    {"call_type": "prescription_refill", "quality": "poor", "scenario": "rep promises a refill they cannot authorize"},
    {"call_type": "prescription_refill", "quality": "medium", "scenario": "handled fine but cold, transactional tone"},
    {"call_type": "clinical_triage", "quality": "excellent", "scenario": "chest pain — rep correctly escalates to 911 / nurse line"},
    {"call_type": "clinical_triage", "quality": "poor", "scenario": "patient describes red-flag symptoms; rep fails to escalate"},
    {"call_type": "clinical_triage", "quality": "good", "scenario": "mild symptoms, appropriate self-care guidance and follow-up"},
    {"call_type": "appointment_scheduling", "quality": "medium", "scenario": "patient reschedules twice; rep slightly impatient but accurate"},
    {"call_type": "billing_inquiry", "quality": "good", "scenario": "duplicate charge dispute, documented and escalated correctly"},
    {"call_type": "insurance_verification", "quality": "medium", "scenario": "rep verifies identity but gives an incomplete coverage answer"},
    {"call_type": "billing_inquiry", "quality": "excellent", "scenario": "rep verifies identity, gathers full info, explains charges clearly, AND correctly escalates a coding-error dispute to billing — textbook completeness"},
    {"call_type": "billing_inquiry", "quality": "poor", "scenario": "rep verifies identity and is polite but fails to escalate a clear coding-error dispute — tells the patient to contact their insurer themselves"},
    {"call_type": "prescription_refill", "quality": "medium", "scenario": "rep initially sends the refill to the wrong pharmacy on outdated info; patient catches it and rep corrects it — initial accuracy error that was recovered"},
]


def _build_prompt(spec: dict) -> str:
    return f"""You are generating realistic training data for a healthcare contact-center QA system.

Produce ONE call transcript between a HealthBridge service representative and a patient.

CALL TYPE: {spec['call_type']}
TARGET QUALITY: {spec['quality']}
SCENARIO: {spec['scenario']}

Requirements:
- Make it realistic and specific (names, dates, dollar amounts, medications as appropriate).
- Timestamps ascending in MM:SS format starting near 00:00.
- The call quality must genuinely reflect the target and scenario — a "poor" call should
  contain the described failure, a "excellent" call should model best practice.
- Provide human-reviewer ground-truth scores (1-5) for each rubric dimension that an honest
  QA reviewer would assign to THIS transcript, plus brief reviewer notes.

RUBRIC (for scoring the ground truth)
{rubric_for_prompt()}

Return the required structured format."""


def generate_one(spec: dict, call_index: int, rep_id: str) -> dict:
    response = config.client().messages.parse(
        model=config.datagen_model(),
        max_tokens=2500,
        system="You generate realistic, varied, and honestly-labeled healthcare call transcripts.",
        messages=[{"role": "user", "content": _build_prompt(spec)}],
        output_format=GeneratedCall,
    )
    gen = response.parsed_output
    gt = gen.ground_truth_qa
    dim_scores = gt.model_dump(exclude={"reviewer_notes"})
    from .rubric import weighted_overall
    return {
        "call_id": f"{SYNTHETIC_ID_PREFIX}-{call_index:05d}",
        "call_type": spec["call_type"],
        "duration_seconds": gen.duration_seconds,
        "rep_id": rep_id,
        "patient_satisfaction_score": gen.patient_satisfaction_score,
        "ground_truth_qa": {
            "overall_score": weighted_overall(dim_scores),
            "reviewer_notes": gt.reviewer_notes,
            "dimension_scores": dim_scores,
        },
        "transcript": [u.model_dump() for u in gen.transcript],
        "_scenario": spec["scenario"],
    }


# Spread synthetic calls across a small pool of reps so the coaching loop has
# multiple calls per rep to aggregate over.
REP_POOL = ["REP-0042", "REP-0015", "REP-0078", "REP-0091", "REP-0103"]


def generate(n: int | None = None, progress=None) -> list[dict]:
    specs = SPECS if n is None else SPECS[:n]
    out: list[dict] = []
    for i, spec in enumerate(specs):
        rep = REP_POOL[i % len(REP_POOL)]
        try:
            call = generate_one(spec, 100 + i, rep)
            out.append(call)
            if progress:
                progress(i + 1, len(specs), call["call_id"])
        except anthropic.APIError as e:
            if progress:
                progress(i + 1, len(specs), f"FAILED: {e}")
    return out


def generate_indices(indices: list[int], progress=None) -> list[dict]:
    """Generate only the SPECS entries at the given indices."""
    out: list[dict] = []
    for done, idx in enumerate(indices):
        spec = SPECS[idx]
        rep = REP_POOL[idx % len(REP_POOL)]
        try:
            call = generate_one(spec, 100 + idx, rep)
            out.append(call)
            if progress:
                progress(done + 1, len(indices), call["call_id"])
        except anthropic.APIError as e:
            if progress:
                progress(done + 1, len(indices), f"FAILED: {e}")
    return out


def save(calls: list[dict], path: str | None = None) -> str:
    path = path or config.GENERATED_PATH
    with open(path, "w") as f:
        json.dump(calls, f, indent=2)
    return path
