"""Smoke tests for POST /evaluate-call. No live API calls — evaluate() is mocked."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src import storage
from src.models import DimensionEvaluation, TranscriptEvaluation
from src.qa_engine import Evaluation

_DIM3 = DimensionEvaluation(
    score=3,
    evidence="n/a",
    reasoning="n/a",
    suggestion="n/a",
)
_DETAIL3 = TranscriptEvaluation(
    greeting_identity_verification=_DIM3,
    empathy_tone=_DIM3,
    accuracy_completeness=_DIM3,
    protocol_adherence=_DIM3,
    closing_next_steps=_DIM3,
    summary="Test call.",
    top_strengths=[],
    top_improvements=[],
)


def _fake_evaluate(transcript) -> Evaluation:
    return Evaluation(
        call_id=transcript.call_id,
        rep_id=transcript.rep_id,
        call_type=transcript.call_type,
        overall_score=3.0,
        detail=_DETAIL3,
    )


_BASE_PAYLOAD = {
    "call_id": "TEST-001",
    "call_type": "billing_inquiry",
    "transcript": [
        {"timestamp": "00:00", "speaker": "rep", "text": "Hello, how can I help?"},
        {"timestamp": "00:05", "speaker": "patient", "text": "I have a billing question."},
    ],
}

client = TestClient(app)


def test_clean_billing_call_returns_pass(tmp_db):
    with patch("src.api.evaluate", side_effect=_fake_evaluate):
        resp = client.post("/evaluate-call", json=_BASE_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "PASS"
    assert data["review_required"] is False
    assert data["reasons"] == []
    assert data["call_id"] == "TEST-001"
    assert data["overall_score"] == 3.0
    assert set(data["dimension_scores"].keys()) == {
        "greeting_identity_verification",
        "empathy_tone",
        "accuracy_completeness",
        "protocol_adherence",
        "closing_next_steps",
    }
    assert isinstance(data["run_id"], int)


def test_clinical_triage_returns_fail(tmp_db):
    payload = {**_BASE_PAYLOAD, "call_id": "TEST-002", "call_type": "clinical_triage"}
    with patch("src.api.evaluate", side_effect=_fake_evaluate):
        resp = client.post("/evaluate-call", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAIL"
    assert data["review_required"] is True
    assert "safety_triage" in data["reasons"]


def test_malformed_request_returns_422():
    # Missing required fields: call_type and transcript.
    resp = client.post("/evaluate-call", json={"call_id": "X"})
    assert resp.status_code == 422


def test_evaluation_persisted_after_post(tmp_db):
    payload = {**_BASE_PAYLOAD, "call_id": "TEST-PERSIST"}
    with patch("src.api.evaluate", side_effect=_fake_evaluate):
        resp = client.post("/evaluate-call", json=payload)
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    row = storage.get_evaluation("TEST-PERSIST")
    assert row is not None
    assert row["call_id"] == "TEST-PERSIST"
    assert row["overall_score"] == 3.0
    assert row["run_id"] == run_id


def test_placeholder_call_id_rejected():
    # "string" is Swagger UI's auto-filled example for a bare `str` field — a
    # request that never replaced it should be rejected, not stored as a real call.
    payload = {**_BASE_PAYLOAD, "call_id": "string"}
    resp = client.post("/evaluate-call", json=payload)
    assert resp.status_code == 422


def test_too_short_call_id_rejected():
    payload = {**_BASE_PAYLOAD, "call_id": "abcd"}  # 4 chars, below min_length=5
    resp = client.post("/evaluate-call", json=payload)
    assert resp.status_code == 422


def test_clinical_triage_creates_pending_review(tmp_db):
    payload = {**_BASE_PAYLOAD, "call_id": "TEST-REVIEW", "call_type": "clinical_triage"}
    with patch("src.api.evaluate", side_effect=_fake_evaluate):
        resp = client.post("/evaluate-call", json=payload)
    assert resp.status_code == 200

    pending = storage.pending_reviews()
    call_ids = [r["call_id"] for r in pending]
    assert "TEST-REVIEW" in call_ids
