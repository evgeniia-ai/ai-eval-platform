"""Routing logic for the Human Review Queue.

needs_review is a pure function: no DB access, no I/O, fully testable.
It returns the list of reason strings that should trigger human review.
An empty list means no review is needed.
"""

from __future__ import annotations

from .qa_engine import Evaluation

_LOW_SCORE_THRESHOLD = 3.0
_DIM_FAIL_MAX = 2
_IDENTITY_DIM = "greeting_identity_verification"


def needs_review(evaluation: Evaluation) -> list[str]:
    """Return reason strings for human review; empty list means no review needed."""
    reasons: list[str] = []
    dim_scores = evaluation.dimension_scores()

    if evaluation.overall_score < _LOW_SCORE_THRESHOLD:
        reasons.append("low_score")

    if any(v <= _DIM_FAIL_MAX for v in dim_scores.values()):
        reasons.append("dimension_fail")

    if evaluation.call_type == "clinical_triage":
        reasons.append("safety_triage")

    if evaluation.call_type == "prescription_refill":
        reasons.append("safety_prescription")

    if dim_scores.get(_IDENTITY_DIM, 99) <= _DIM_FAIL_MAX:
        reasons.append("privacy_identity_risk")

    return reasons
