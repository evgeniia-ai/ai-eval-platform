"""Routing logic for the Human Review Queue.

needs_review is a pure function: no DB access, no I/O, fully testable.
It returns the list of reason strings that should trigger human review.
An empty list means no review is needed.
"""

from __future__ import annotations

from .qa_engine import Evaluation

# Lowered from 3.0 — the old threshold flagged plenty of merely-mediocre calls,
# not just risky ones. See docs/eval_findings.md for the routing-noise rationale.
_LOW_SCORE_THRESHOLD = 2.5
# Generic "any dimension" catch-all — tightened from 2 to 1. A single dimension
# at 2 is common and often unremarkable; type-specific and identity risk below
# now cover the "2" case with more targeted conditions instead.
_DIM_FAIL_MAX = 1
# Threshold for identity privacy risk and the type-specific safety risks below.
_RISK_DIM_MAX = 2
_IDENTITY_DIM = "greeting_identity_verification"


def needs_review(evaluation: Evaluation) -> list[str]:
    """Return reason strings for human review; empty list means no review needed."""
    reasons: list[str] = []
    dim_scores = evaluation.dimension_scores()

    if evaluation.overall_score < _LOW_SCORE_THRESHOLD:
        reasons.append("low_score")

    if any(v <= _DIM_FAIL_MAX for v in dim_scores.values()):
        reasons.append("dimension_fail")

    if dim_scores.get(_IDENTITY_DIM, 99) <= _RISK_DIM_MAX:
        reasons.append("privacy_identity_risk")

    # Type-specific safety risk — no longer an unconditional call_type trigger.
    # A clean clinical_triage or prescription_refill call no longer routes by
    # type alone; it routes when the type-relevant dimension shows real risk.
    if evaluation.call_type == "clinical_triage" and (
        dim_scores.get("protocol_adherence", 99) <= _RISK_DIM_MAX
        or dim_scores.get("accuracy_completeness", 99) <= _RISK_DIM_MAX
    ):
        reasons.append("safety_triage")

    if (
        evaluation.call_type == "prescription_refill"
        and dim_scores.get("protocol_adherence", 99) <= _RISK_DIM_MAX
    ):
        reasons.append("safety_prescription")

    return reasons
