"""Tests for src/demo_data.py's read-only DEMO_MODE data layer, focused on the
fields added for the Phase 0 demo refresh: human_reviews and labeling_stats,
plus the Review Queue page's read-only rendering in demo mode.
"""

import json
import pathlib

from src import demo_data


def _write_demo_json(path, **overrides):
    data = {
        "exported_at": "2026-01-01T00:00:00.000",
        "suite_runs": [],
        "evaluations": [],
        "human_reviews": [],
        "labeling_stats": {},
    }
    data.update(overrides)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    demo_data._load.cache_clear()


_REVIEW_ROW = {
    "review_id": 1,
    "call_id": "HB-TEST-0001",
    "run_id": 1,
    "status": "Pending",
    "reason": "low_score",
    "reviewer_note": None,
    "created_at": "2026-01-01T00:00:00.000",
    "updated_at": None,
}


def test_all_reviews_reads_exported_reviews_newest_first(tmp_demo_path):
    reviews = [
        {**_REVIEW_ROW, "review_id": 1},
        {**_REVIEW_ROW, "review_id": 2, "status": "Confirmed Issue"},
    ]
    _write_demo_json(demo_data.DEMO_PATH, human_reviews=reviews)

    result = demo_data.all_reviews()
    assert [r["review_id"] for r in result] == [2, 1]


def test_all_reviews_empty_when_key_missing(tmp_demo_path):
    with open(demo_data.DEMO_PATH, "w", encoding="utf-8") as f:
        json.dump({"exported_at": "x", "suite_runs": [], "evaluations": []}, f)
    demo_data._load.cache_clear()

    assert demo_data.all_reviews() == []


def test_labeling_stats_reads_exported_stats(tmp_demo_path):
    stats = {"n_labeled": 39, "n_seed_labeled": 7, "n_gpt_labeled": 32, "n_holdout": 6}
    _write_demo_json(demo_data.DEMO_PATH, labeling_stats=stats)

    assert demo_data.labeling_stats() == stats


def test_labeling_stats_empty_when_key_missing(tmp_demo_path):
    with open(demo_data.DEMO_PATH, "w", encoding="utf-8") as f:
        json.dump({"exported_at": "x", "suite_runs": [], "evaluations": []}, f)
    demo_data._load.cache_clear()

    assert demo_data.labeling_stats() == {}


_LABELED_CALL = {
    "call_id": "HB-GPT-TEST",
    "call_type": "billing_inquiry",
    "duration_seconds": 60,
    "rep_id": "REP-1",
    "ground_truth_qa": {
        "overall_score": 4.0,
        "reviewer_notes": "Observed: fine.\nConcern: minor thing.\nImpact: small.",
        "dimension_scores": {"greeting_identity_verification": 4},
    },
    "transcript": [{"timestamp": "00:00:00", "speaker": "rep", "text": "hi"}],
}


def test_gpt_calls_reads_exported_snapshot(tmp_demo_path):
    _write_demo_json(demo_data.DEMO_PATH, gpt_calls=[_LABELED_CALL])
    assert demo_data.gpt_calls() == [_LABELED_CALL]


def test_gpt_calls_empty_when_key_missing(tmp_demo_path):
    with open(demo_data.DEMO_PATH, "w", encoding="utf-8") as f:
        json.dump({"exported_at": "x", "suite_runs": [], "evaluations": []}, f)
    demo_data._load.cache_clear()

    assert demo_data.gpt_calls() == []


def test_seed_calls_reads_exported_snapshot(tmp_demo_path):
    _write_demo_json(demo_data.DEMO_PATH, seed_calls=[_LABELED_CALL])
    assert demo_data.seed_calls() == [_LABELED_CALL]


def test_seed_calls_empty_when_key_missing(tmp_demo_path):
    with open(demo_data.DEMO_PATH, "w", encoding="utf-8") as f:
        json.dump({"exported_at": "x", "suite_runs": [], "evaluations": []}, f)
    demo_data._load.cache_clear()

    assert demo_data.seed_calls() == []


def test_review_queue_is_read_only_in_demo_mode(monkeypatch, tmp_demo_path):
    _write_demo_json(demo_data.DEMO_PATH, human_reviews=[dict(_REVIEW_ROW)])
    monkeypatch.setenv("DEMO_MODE", "true")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "src" / "app.py"))
    at.run()
    at.radio(key=at.radio[0].key).set_value("Review Queue").run()

    assert not at.exception
    assert not any("Save" in b.label for b in at.button)
    assert any("read-only" in c.value for c in at.caption)
    assert len(at.expander) >= 1  # the pending review renders as a card


def test_demo_banner_shows_labeled_count(monkeypatch, tmp_demo_path):
    _write_demo_json(demo_data.DEMO_PATH, labeling_stats={
        "n_labeled": 39, "n_seed_labeled": 7, "n_gpt_labeled": 32, "n_holdout": 6,
    })
    monkeypatch.setenv("DEMO_MODE", "true")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "src" / "app.py"))
    at.run()

    assert not at.exception
    assert any("39 human-labeled calls" in i.value for i in at.info)
