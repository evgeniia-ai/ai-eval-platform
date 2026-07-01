"""FastAPI evaluator endpoint — evaluates transcripts and persists results to the shared DB."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import storage
from .models import CallType, Transcript, Utterance
from .qa_engine import QAError, evaluate
from .review import needs_review

app = FastAPI(title="QA Evaluator API")


@app.on_event("startup")
def _startup() -> None:
    storage.init()


class EvaluateRequest(BaseModel):
    call_id: str
    call_type: CallType
    transcript: list[Utterance]
    rep_id: str = "unknown"
    duration_seconds: int = 0
    patient_satisfaction_score: Optional[float] = None


class EvaluateResponse(BaseModel):
    call_id: str
    overall_score: float
    status: str
    review_required: bool
    reasons: list[str]
    dimension_scores: dict[str, int]
    run_id: int


@app.post("/evaluate-call", response_model=EvaluateResponse)
def evaluate_call(request: EvaluateRequest) -> EvaluateResponse:
    transcript = Transcript(
        call_id=request.call_id,
        call_type=request.call_type,
        duration_seconds=request.duration_seconds,
        rep_id=request.rep_id,
        patient_satisfaction_score=request.patient_satisfaction_score,
        transcript=request.transcript,
    )

    try:
        evaluation = evaluate(transcript)
    except QAError as e:
        raise HTTPException(status_code=422, detail=str(e))

    run_id = storage.save(evaluation, transcript=transcript)

    reasons = needs_review(evaluation)
    return EvaluateResponse(
        call_id=evaluation.call_id,
        overall_score=evaluation.overall_score,
        status="FAIL" if reasons else "PASS",
        review_required=bool(reasons),
        reasons=reasons,
        dimension_scores=evaluation.dimension_scores(),
        run_id=run_id,
    )
