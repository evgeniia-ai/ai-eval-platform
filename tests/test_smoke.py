"""Layer 1: free, deterministic smoke tests. Zero API calls.

The judge (src.qa_engine.evaluate) is replaced by _fake_evaluate, which
returns a valid Evaluation dataclass built from real Pydantic models — no
MagicMock — so storage.save's model_dump_json() calls work without patching.
"""

import json
from unittest.mock import patch

import pytest

from src import config, ingest, storage
from src.models import DimensionEvaluation, Transcript, TranscriptEvaluation
from src.qa_engine import Evaluation, QAError
from src.review import needs_review
from src.rubric import DIMENSION_KEYS, weighted_overall

_SEED_COUNT = 7

# A single DimensionEvaluation reused for all five dimensions in the fake.
# score must be a Literal[1,2,3,4,5] int — Pydantic validates this at construction.
_DIM = DimensionEvaluation(
    score=3,
    evidence="Rep said 'Thank you for calling HealthBridge.'",
    reasoning="Adequate but not exemplary.",
    suggestion="Verify date of birth before taking account action.",
)
_DETAIL = TranscriptEvaluation(
    greeting_identity_verification=_DIM,
    empathy_tone=_DIM,
    accuracy_completeness=_DIM,
    protocol_adherence=_DIM,
    closing_next_steps=_DIM,
    summary="A competent call with room for improvement.",
    top_strengths=["Professional greeting"],
    top_improvements=["Identity verification completeness"],
)


def _fake_evaluate(transcript) -> Evaluation:
    """Mirrors the real evaluate() signature; returns a fixed score-3 result."""
    return Evaluation(
        call_id=transcript.call_id,
        rep_id=transcript.rep_id,
        call_type=transcript.call_type,
        overall_score=3.0,
        detail=_DETAIL,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_seed_transcripts():
    transcripts = ingest.load_transcripts(include_generated=False)
    assert len(transcripts) == _SEED_COUNT
    for t in transcripts:
        assert t.call_id
        assert len(t.transcript) > 0


def test_default_sets_reference_existing_transcripts():
    # A call_id in SMOKE_SET/REGRESSION_SET with no matching transcript on disk
    # is silently skipped by ingest.run_suite (see "skipped_call_ids") instead of
    # failing — catch that drift here so it fails loudly in CI.
    from src import labeling
    from src.test_sets import FULL_SET, REGRESSION_SET, SMOKE_SET

    seed_gen_ids = {t.call_id for t in ingest.load_transcripts(include_generated=True)}
    missing_smoke = SMOKE_SET - seed_gen_ids
    missing_regression = REGRESSION_SET - seed_gen_ids
    assert not missing_smoke, f"SMOKE_SET references missing call_ids: {missing_smoke}"
    assert not missing_regression, f"REGRESSION_SET references missing call_ids: {missing_regression}"

    # FULL_SET can reference GPT-generated calls too (data/gpt_transcripts.json),
    # which ingest.load_transcripts() doesn't know about.
    all_ids = seed_gen_ids | {c["call_id"] for c in labeling.load_calls()}
    missing_full = FULL_SET - all_ids
    assert not missing_full, f"FULL_SET references missing call_ids: {missing_full}"


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def test_export_demo_produces_suite_runs_reviews_and_stats(tmp_db):
    import json as _json
    from collections import Counter

    from scripts.export_demo import export

    storage.init()
    t = ingest.load_transcripts(include_generated=False)[0]
    # Evaluation first so its created_at precedes every suite run below —
    # each suite run's _matching_evaluation lookup requires created_at <= suite's.
    storage.save(_fake_evaluate(t), transcript=t, model="claude-sonnet-4-6")

    for i in range(7):  # Smoke/Regression must never be exported at all, however many exist
        storage.save_suite_run(
            suite_name="Smoke" if i % 2 == 0 else "Regression",
            selected_call_ids=[t.call_id],
            failed_call_ids=[],
            skipped_call_ids=[],
            ok_scores=[4.0],
            gt_pairs=[],
            model="claude-sonnet-4-6",
        )
    for _ in range(4):  # one more than N_FULL_RUNS, to verify only the latest 3 are kept
        storage.save_suite_run(
            suite_name="Full",
            selected_call_ids=[t.call_id],
            failed_call_ids=[],
            skipped_call_ids=[],
            ok_scores=[4.0],
            gt_pairs=[],
            model="claude-sonnet-4-6",  # matches the evaluation saved above, for _matching_evaluation
        )

    # A low overall score trips needs_review -> a human_reviews row to export.
    low_ev = _fake_evaluate(t)
    low_ev.overall_score = 2.0
    storage.save(low_ev, transcript=t, model="claude-sonnet-4-6")

    data = export()
    _json.dumps(data)  # must be JSON-serializable

    # Smoke/Regression are stale pre-guidelines runs — never exported, only Full is.
    assert Counter(r["suite_name"] for r in data["suite_runs"]) == {"Full": 3}
    assert data["evaluations"]
    assert all(e["call_id"] == t.call_id for e in data["evaluations"])

    assert len(data["human_reviews"]) == 1
    assert data["human_reviews"][0]["call_id"] == t.call_id

    stats = data["labeling_stats"]
    assert stats.keys() == {"n_labeled", "n_seed_labeled", "n_gpt_labeled", "n_holdout"}
    assert stats["n_labeled"] == stats["n_seed_labeled"] + stats["n_gpt_labeled"]


def test_labeling_stats_counts_seed_and_gpt_labeled(
    tmp_seed_path, tmp_gpt_path, tmp_holdout_path
):
    from scripts.export_demo import _labeling_stats

    seed_calls = [
        {
            "call_id": "HB-2026-S1",
            "call_type": "billing_inquiry",
            "duration_seconds": 60,
            "rep_id": "REP-1",
            "ground_truth_qa": {
                "overall_score": 4.0,
                "reviewer_notes": "n/a",
                "dimension_scores": {"greeting_identity_verification": 4},
            },
            "transcript": [{"timestamp": "00:00:00", "speaker": "rep", "text": "hi"}],
        },
        {
            "call_id": "HB-2026-S2",
            "call_type": "billing_inquiry",
            "duration_seconds": 60,
            "rep_id": "REP-1",
            "transcript": [{"timestamp": "00:00:00", "speaker": "rep", "text": "hi"}],
        },
    ]
    with open(config.SEED_PATH, "w", encoding="utf-8") as f:
        json.dump(seed_calls, f)

    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([
            {
                "call_id": "HB-GPT-0001",
                "call_type": "billing_inquiry",
                "duration_seconds": 60,
                "rep_id": "REP-GPT-01",
                "ground_truth_qa": {
                    "overall_score": 4.0,
                    "reviewer_notes": "Observed: fine.",
                    "dimension_scores": {"greeting_identity_verification": 4},
                },
                "transcript": [{"timestamp": "00:00:00", "speaker": "rep", "text": "hi"}],
            },
        ], f)

    with open(config.HOLDOUT_PATH, "w", encoding="utf-8") as f:
        json.dump(["HB-GPT-9999"], f)

    stats = _labeling_stats()
    assert stats == {
        "n_labeled": 2,
        "n_seed_labeled": 1,
        "n_gpt_labeled": 1,
        "n_holdout": 1,
    }


def test_app_imports_cleanly_in_demo_mode():
    import os
    import pathlib
    import subprocess
    import sys

    env = {**os.environ, "DEMO_MODE": "true"}
    result = subprocess.run(
        [sys.executable, "-c", "import src.app"],
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_run_pipeline_smoke(tmp_db):
    with patch("src.ingest.evaluate", side_effect=_fake_evaluate):
        ok, fail = ingest.run(include_generated=False)
    assert (ok, fail) == (_SEED_COUNT, 0)


def _risky_fake_evaluate(transcript) -> Evaluation:
    """A fake judge result that WOULD trigger review under needs_review() —
    used to prove ingest.run_suite's route_to_review=False actually suppresses it."""
    risky_dim = DimensionEvaluation(score=1, evidence="n/a", reasoning="n/a", suggestion="n/a")
    detail = TranscriptEvaluation(
        greeting_identity_verification=risky_dim,
        empathy_tone=_DIM,
        accuracy_completeness=_DIM,
        protocol_adherence=_DIM,
        closing_next_steps=_DIM,
        summary="Risky call.",
        top_strengths=[],
        top_improvements=[],
    )
    return Evaluation(
        call_id=transcript.call_id,
        rep_id=transcript.rep_id,
        call_type=transcript.call_type,
        overall_score=1.5,
        detail=detail,
    )


def test_run_suite_creates_zero_reviews(tmp_db):
    storage.init()
    t = ingest.load_transcripts(include_generated=False)[0]
    transcript_map = {t.call_id: t}

    with patch("src.ingest.evaluate", side_effect=_risky_fake_evaluate):
        result = ingest.run_suite(
            call_ids=[t.call_id],
            transcript_map=transcript_map,
            suite_name="Smoke",
            model="claude-sonnet-4-6",
        )

    assert result["ok_scores"] == [1.5]
    # Sanity check: this evaluation would trigger review outside a suite run...
    assert needs_review(_risky_fake_evaluate(t)) != []
    # ...but ingest.run_suite disables routing, so zero reviews were created.
    assert storage.pending_reviews() == []


def test_storage_roundtrip(tmp_db):
    t = ingest.load_transcripts(include_generated=False)[0]
    ev = _fake_evaluate(t)
    storage.init()
    storage.save(ev, transcript=t)

    row = storage.get_evaluation(t.call_id)
    assert row is not None
    scores = row["detail"].dimension_scores()
    for key in DIMENSION_KEYS:
        assert scores[key] == 3


def test_storage_accumulates_runs_per_call(tmp_db):
    # Each save creates a NEW row (per-run INSERT, not UPSERT).
    # count() uses COUNT(DISTINCT call_id), so two saves of the same call
    # still shows 1 distinct call — but runs_for_call should return 2 rows.
    t = ingest.load_transcripts(include_generated=False)[0]
    ev = _fake_evaluate(t)
    storage.init()
    storage.save(ev, transcript=t)
    storage.save(ev, transcript=t)
    assert storage.count() == 1                              # 1 distinct call
    assert len(storage.runs_for_call(t.call_id)) == 2       # 2 run rows accumulated


def test_weighted_overall_deterministic():
    for score in (1, 3, 5):
        result = weighted_overall({k: score for k in DIMENSION_KEYS})
        assert result == float(score), f"uniform score {score}: expected {score}, got {result}"


def test_evaluate_empty_transcript_raises():
    from src.qa_engine import evaluate

    t = Transcript(
        call_id="test-empty",
        call_type="billing_inquiry",
        duration_seconds=10,
        rep_id="REP-TEST",
        transcript=[],
    )
    with pytest.raises(QAError):
        evaluate(t)


def test_evaluate_malformed_dict_raises():
    from src.qa_engine import evaluate

    with pytest.raises(QAError):
        evaluate({"garbage": True})


def test_build_gt_pairs_excludes_synthetic():
    from src.data_gen import build_gt_pairs

    results = [
        ("HB-2026-00147", 4.0, 4.0),        # human — kept
        ("HB-2026-00203", 3.0, 3.5),        # human — kept
        ("HB-SYNTH-00118", 5.0, 4.5),       # synthetic GT present — excluded
    ]
    pairs = build_gt_pairs(results)
    assert pairs == [(4.0, 4.0), (3.0, 3.5)]


# ---------------------------------------------------------------------------
# Suite run storage
# ---------------------------------------------------------------------------

def test_suite_run_roundtrip(tmp_db):
    storage.init()
    ok_scores = [4.0, 5.0, 3.0]
    gt_pairs = [(4.0, 4.0), (5.0, 4.0)]
    selected = ["HB-001", "HB-002", "HB-003"]
    failed = ["HB-004"]
    skipped = ["HB-005"]

    storage.save_suite_run(
        suite_name="Smoke",
        selected_call_ids=selected,
        failed_call_ids=failed,
        skipped_call_ids=skipped,
        ok_scores=ok_scores,
        gt_pairs=gt_pairs,
        model="test-model",
    )

    rows = storage.all_suite_runs()
    assert len(rows) == 1
    r = rows[0]

    assert r["suite_name"] == "Smoke"
    assert r["model"] == "test-model"
    assert r["n_ok"] == 3
    assert r["n_failed"] == 1
    assert r["n_skipped"] == 1
    assert r["n_calls"] == 5
    assert r["mean_overall"] == round(sum(ok_scores) / len(ok_scores), 2)
    expected_mae = round(sum(abs(p - g) for p, g in gt_pairs) / len(gt_pairs), 2)
    assert r["mae_vs_gt"] == expected_mae

    import json
    assert json.loads(r["selected_call_ids"]) == sorted(selected)


def test_suite_run_all_failed(tmp_db):
    # Edge case: no successful evaluations — must not divide by zero.
    storage.init()
    storage.save_suite_run(
        suite_name="Smoke",
        selected_call_ids=["HB-001", "HB-002"],
        failed_call_ids=["HB-001", "HB-002"],
        skipped_call_ids=[],
        ok_scores=[],
        gt_pairs=[],
        model="test-model",
    )

    rows = storage.all_suite_runs()
    assert len(rows) == 1
    r = rows[0]
    assert r["n_ok"] == 0
    assert r["n_failed"] == 2
    assert r["mean_overall"] is None
    assert r["mae_vs_gt"] is None


def test_runs_for_call_returns_all_runs(tmp_db):
    t = ingest.load_transcripts(include_generated=False)[0]
    ev = _fake_evaluate(t)
    storage.init()
    storage.save(ev, transcript=t)
    storage.save(ev, transcript=t)

    runs = storage.runs_for_call(t.call_id)
    assert len(runs) == 2
    # Newest first (descending run_id).
    assert runs[0]["run_id"] > runs[1]["run_id"]
    # Dimension scores present on every run row.
    for run in runs:
        for key in DIMENSION_KEYS:
            assert run[key] == 3


# ---------------------------------------------------------------------------
# Human review queue
# ---------------------------------------------------------------------------

def _make_dim(score: int) -> DimensionEvaluation:
    return DimensionEvaluation(
        score=score,
        evidence="n/a",
        reasoning="n/a",
        suggestion="n/a",
    )


def _make_evaluation(
    overall: float,
    call_type: str = "billing_inquiry",
    identity: int = 3,
    empathy: int = 3,
    accuracy: int = 3,
    protocol: int = 3,
    closing: int = 3,
) -> Evaluation:
    detail = TranscriptEvaluation(
        greeting_identity_verification=_make_dim(identity),
        empathy_tone=_make_dim(empathy),
        accuracy_completeness=_make_dim(accuracy),
        protocol_adherence=_make_dim(protocol),
        closing_next_steps=_make_dim(closing),
        summary="test",
        top_strengths=[],
        top_improvements=[],
    )
    return Evaluation(
        call_id="TEST-001",
        rep_id="REP-TEST",
        call_type=call_type,
        overall_score=overall,
        detail=detail,
    )


def test_needs_review_triggers():
    # Clean call — no triggers.
    assert needs_review(_make_evaluation(4.0, "billing_inquiry")) == []

    # Overall score: threshold lowered from 3.0 to 2.5.
    assert needs_review(_make_evaluation(2.9, "billing_inquiry")) == []  # no longer triggers
    assert needs_review(_make_evaluation(2.4, "billing_inquiry")) == ["low_score"]

    # Generic dimension-fail catch-all: tightened from <= 2 to <= 1.
    assert needs_review(_make_evaluation(4.0, "billing_inquiry", empathy=2)) == []  # no longer triggers
    reasons = needs_review(_make_evaluation(4.0, "billing_inquiry", empathy=1))
    assert reasons == ["dimension_fail"]

    # Identity verification <= 2 → privacy_identity_risk, but NOT the generic
    # dimension_fail catch-all anymore (that needs <= 1 now).
    reasons = needs_review(_make_evaluation(4.0, "billing_inquiry", identity=2))
    assert reasons == ["privacy_identity_risk"]
    # Identity at 1 trips both.
    reasons = needs_review(_make_evaluation(4.0, "billing_inquiry", identity=1))
    assert set(reasons) == {"dimension_fail", "privacy_identity_risk"}

    # A perfect clinical_triage / prescription_refill call no longer routes by
    # type alone — this is the core of the routing-noise fix.
    assert needs_review(_make_evaluation(4.0, "clinical_triage")) == []
    assert needs_review(_make_evaluation(4.0, "prescription_refill")) == []

    # clinical_triage routes when protocol_adherence OR accuracy_completeness is <= 2.
    assert needs_review(_make_evaluation(4.0, "clinical_triage", protocol=2)) == ["safety_triage"]
    assert needs_review(_make_evaluation(4.0, "clinical_triage", accuracy=2)) == ["safety_triage"]

    # prescription_refill routes only on protocol_adherence <= 2.
    assert needs_review(_make_evaluation(4.0, "prescription_refill", protocol=2)) == ["safety_prescription"]
    assert needs_review(_make_evaluation(4.0, "prescription_refill", accuracy=2)) == []

    # Multiple triggers: low score + triage risk + identity fail.
    reasons = needs_review(_make_evaluation(2.4, "clinical_triage", protocol=2, identity=2))
    assert set(reasons) == {"low_score", "safety_triage", "privacy_identity_risk"}


def test_save_and_fetch_review(tmp_db):
    storage.init()

    review_id = storage.save_review("HB-TEST-001", run_id=42, reasons=["low_score"])
    assert review_id is not None

    pending = storage.pending_reviews()
    assert len(pending) == 1
    row = pending[0]
    assert row["call_id"] == "HB-TEST-001"
    assert row["run_id"] == 42
    assert row["status"] == "Pending"
    assert row["reason"] == "low_score"
    assert row["reviewer_note"] is None

    storage.update_review_status(review_id, "Confirmed Issue", "Clear safety problem")

    # Should no longer appear in pending.
    assert storage.pending_reviews() == []

    # Should appear in all_reviews with updated fields.
    all_rows = storage.all_reviews()
    assert len(all_rows) == 1
    updated = all_rows[0]
    assert updated["status"] == "Confirmed Issue"
    assert updated["reviewer_note"] == "Clear safety problem"
    assert updated["updated_at"] is not None


def test_save_review_no_duplicate(tmp_db):
    storage.init()

    id1 = storage.save_review("HB-TEST-002", run_id=7, reasons=["low_score"])
    id2 = storage.save_review("HB-TEST-002", run_id=7, reasons=["low_score", "safety_triage"])

    # Same ID returned; no second row created.
    assert id1 == id2
    assert len(storage.all_reviews()) == 1


def test_review_created_automatically_on_triggered_save(tmp_db):
    # Saving an evaluation that trips a trigger must create a Pending review row.
    storage.init()
    ev = _make_evaluation(2.0, "billing_inquiry")  # overall < 2.5 → "low_score"
    storage.save(ev)

    pending = storage.pending_reviews()
    assert len(pending) == 1
    assert pending[0]["call_id"] == ev.call_id
    assert "low_score" in pending[0]["reason"]
    assert pending[0]["status"] == "Pending"


def test_no_review_created_for_clean_save(tmp_db):
    # A passing evaluation must not create a review row.
    storage.init()
    ev = _make_evaluation(4.0, "billing_inquiry")  # no triggers
    storage.save(ev)

    assert storage.pending_reviews() == []
    assert storage.all_reviews() == []


def test_review_dedupes_across_repeated_runs(tmp_db):
    # Re-evaluating the same call across N judge runs (re-run, model comparison,
    # suite re-run, ...) must not pile up N pending rows for one call.
    storage.init()
    ev1 = _make_evaluation(2.0, "billing_inquiry")  # low_score
    storage.save(ev1)
    ev2 = _make_evaluation(2.2, "billing_inquiry")  # low_score again
    run_id_2 = storage.save(ev2)

    pending = [r for r in storage.pending_reviews() if r["call_id"] == ev1.call_id]
    assert len(pending) == 1
    assert pending[0]["run_id"] == run_id_2
    assert len(storage.all_reviews()) == 1
