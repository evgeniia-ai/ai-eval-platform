"""Tests for scripts/import_gpt_transcripts.py's validation rules. Pure logic,
no API calls, no filesystem writes.
"""

from scripts.import_gpt_transcripts import strip_code_fences, validate_call


def _make_gpt_call(call_id: str = "HB-GPT-0001", n: int = 12) -> dict:
    transcript = []
    for i in range(n):
        secs = i * 5
        ts = f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
        transcript.append({
            "timestamp": ts,
            "speaker": "rep" if i % 2 == 0 else "patient",
            "text": f"line {i}",
        })
    return {
        "call_id": call_id,
        "call_type": "billing_inquiry",
        "duration_seconds": n * 5,
        "rep_id": "REP-GPT-01",
        "transcript": transcript,
    }


def test_validate_call_accepts_well_formed_sample():
    known_ids: set[str] = set()
    t, reason = validate_call(_make_gpt_call(), known_ids)
    assert reason is None
    assert t is not None
    assert t.call_id in known_ids


def test_validate_call_rejects_duplicate_id():
    known_ids = {"HB-GPT-0001"}
    t, reason = validate_call(_make_gpt_call(), known_ids)
    assert t is None
    assert "duplicate" in reason


def test_validate_call_rejects_backwards_timestamp():
    raw = _make_gpt_call()
    raw["transcript"][2]["timestamp"] = "00:00:00"  # earlier than transcript[1]
    t, reason = validate_call(raw, set())
    assert t is None
    assert "monoton" in reason


def test_validate_call_rejects_bad_call_id_pattern():
    t, reason = validate_call(_make_gpt_call(call_id="HB-2026-00147"), set())
    assert t is None
    assert "HB-GPT" in reason


def test_validate_call_rejects_ground_truth_present():
    raw = _make_gpt_call()
    raw["ground_truth_qa"] = {
        "overall_score": 4.0,
        "reviewer_notes": "n/a",
        "dimension_scores": {"greeting_identity_verification": 4},
    }
    t, reason = validate_call(raw, set())
    assert t is None
    assert "ground_truth_qa" in reason


def test_validate_call_rejects_too_few_utterances():
    raw = _make_gpt_call(n=5)
    t, reason = validate_call(raw, set())
    assert t is None
    assert "utterances" in reason


def test_strip_code_fences_tolerates_json_fence():
    fenced = "```json\n[{\"a\": 1}]\n```"
    assert strip_code_fences(fenced) == '[{"a": 1}]'


def test_strip_code_fences_passthrough_when_no_fence():
    plain = '[{"a": 1}]'
    assert strip_code_fences(plain) == plain
