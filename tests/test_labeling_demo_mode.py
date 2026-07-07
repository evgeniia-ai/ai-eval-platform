"""Tests for the Labeling page's read-only DEMO_MODE showcase: it's no longer
hidden, defaults to a richly-annotated (Observed/Concern/Impact) example,
pre-fills from the frozen export, and disables every editing control. Also
checks normal mode is unaffected.
"""

import json
import pathlib

from src import config, demo_data
from tests.test_labeling import _at

_DIMS = [
    "greeting_identity_verification",
    "empathy_tone",
    "accuracy_completeness",
    "protocol_adherence",
    "closing_next_steps",
]


def _call(call_id: str, call_type: str, score: int, notes: str) -> dict:
    return {
        "call_id": call_id,
        "call_type": call_type,
        "duration_seconds": 60,
        "rep_id": "REP-1",
        "ground_truth_qa": {
            "overall_score": score,
            "reviewer_notes": notes,
            "dimension_scores": {k: score for k in _DIMS},
        },
        "transcript": [{"timestamp": "00:00:00", "speaker": "rep", "text": f"hi from {call_id}"}],
    }


_PLAIN_GPT_CALL = _call("HB-GPT-0001", "billing_inquiry", 3, "Observed: adequate handling.")
_RICH_GPT_CALL = _call(
    "HB-GPT-0002",
    "appointment_scheduling",
    4,
    "Observed: identity verified before any account action.\n"
    "Concern: appointment date was not restated back to the patient.\n"
    "Impact: patient may miss the appointment.",
)
_SEED_CALL = _call("HB-2026-00001", "billing_inquiry", 5, "Observed: excellent call, no issues.")


def _write_demo_json(path, **overrides):
    data = {
        "exported_at": "2026-01-01T00:00:00.000",
        "suite_runs": [],
        "evaluations": [],
        "human_reviews": [],
        "labeling_stats": {"n_labeled": 3, "n_seed_labeled": 1, "n_gpt_labeled": 2, "n_holdout": 0},
        "gpt_calls": [],
        "seed_calls": [],
    }
    data.update(overrides)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    demo_data._load.cache_clear()


def _at_demo_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMO_MODE", "true")
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "src" / "app.py"))
    at.run(timeout=30)
    at.radio(key=at.radio[0].key).set_value("Labeling").run(timeout=30)
    return at


def test_demo_mode_defaults_to_call_with_rich_notes(monkeypatch, tmp_demo_path):
    _write_demo_json(
        demo_data.DEMO_PATH,
        gpt_calls=[_PLAIN_GPT_CALL, _RICH_GPT_CALL],
        seed_calls=[_SEED_CALL],
    )
    at = _at_demo_mode(monkeypatch)

    assert not at.exception
    assert at.radio(key="label_source").value == "GPT calls"
    assert at.selectbox(key="label_call_selectbox").value == _RICH_GPT_CALL["call_id"]


def test_demo_mode_prefills_and_disables_all_inputs(monkeypatch, tmp_demo_path):
    _write_demo_json(demo_data.DEMO_PATH, gpt_calls=[_RICH_GPT_CALL], seed_calls=[])
    at = _at_demo_mode(monkeypatch)

    call_id = _RICH_GPT_CALL["call_id"]
    score_boxes = [sb for sb in at.selectbox if sb.key.startswith(f"label_score_{call_id}")]
    assert len(score_boxes) == len(_DIMS)
    assert all(sb.value == 4 for sb in score_boxes)
    assert all(sb.disabled for sb in score_boxes)

    observed = at.text_area(key=f"label_observed_{call_id}")
    assert observed.disabled is True
    assert "identity verified" in observed.value
    assert at.text_area(key=f"label_concern_{call_id}").disabled is True
    assert "not restated" in at.text_area(key=f"label_concern_{call_id}").value
    assert at.text_area(key=f"label_impact_{call_id}").disabled is True


def test_demo_mode_has_no_save_button_and_shows_caption(monkeypatch, tmp_demo_path):
    _write_demo_json(demo_data.DEMO_PATH, gpt_calls=[_RICH_GPT_CALL], seed_calls=[])
    at = _at_demo_mode(monkeypatch)

    assert not any("Save" in b.label or "Update" in b.label for b in at.button)
    demo_captions = [c.value for c in at.caption if "Labeling is disabled in demo mode" in c.value]
    assert len(demo_captions) == 1
    assert "3 calls labeled" in demo_captions[0]


def test_normal_mode_labeling_page_unaffected(monkeypatch, tmp_db, tmp_gpt_path):
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([_PLAIN_GPT_CALL], f)

    at = _at(monkeypatch, tmp_db, tmp_gpt_path)

    assert not any(sb.disabled for sb in at.selectbox if sb.key.startswith("label_score_"))
    assert not any(ta.disabled for ta in at.text_area if ta.key.startswith("label_observed_"))
    assert any("Save" in b.label or "Update" in b.label for b in at.button)
