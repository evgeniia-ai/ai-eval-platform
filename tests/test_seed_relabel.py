"""Tests for re-labeling the seed calls on the Labeling page: the seed source
selector, the pre-relabel backup (src.labeling.save_seed_label /
backup_original_seed_label), and scripts/compare_relabels.py.
"""

import json
import os
import pathlib

from scripts.compare_relabels import compare
from src import config, labeling
from src.rubric import DIMENSION_KEYS
from tests.test_labeling import _at, _fill_and_save

_ORIGINAL_GT = {
    "overall_score": 4.2,
    "reviewer_notes": "Strong greeting and empathy. Minor gap in closing.",
    "dimension_scores": {
        "greeting_identity_verification": 5,
        "empathy_tone": 5,
        "accuracy_completeness": 4,
        "protocol_adherence": 4,
        "closing_next_steps": 3,
    },
}

_SEED_CALL = {
    "call_id": "HB-2026-00147",
    "call_type": "appointment_scheduling",
    "duration_seconds": 342,
    "rep_id": "REP-0042",
    "ground_truth_qa": dict(_ORIGINAL_GT),
    "transcript": [
        {"timestamp": "00:00:00", "speaker": "rep", "text": "Thank you for calling HealthBridge."},
        {"timestamp": "00:00:05", "speaker": "patient", "text": "I need to reschedule."},
    ],
}

_NEW_SCORES = {k: 4 for k in DIMENSION_KEYS}
_NEW_OBSERVED = "Re-reviewed under the new guidelines; identity verification was solid."


def _write_seed(path, calls):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calls, f)


# ---------------------------------------------------------------------------
# save_seed_label / backup
# ---------------------------------------------------------------------------

def test_save_seed_label_overwrites_ground_truth_qa(tmp_seed_path, tmp_seed_backup_path):
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])

    gt = labeling.save_seed_label(_SEED_CALL["call_id"], _NEW_SCORES, _NEW_OBSERVED)

    assert gt["dimension_scores"] == _NEW_SCORES
    assert gt["reviewer_notes"] == f"Observed: {_NEW_OBSERVED}"

    saved = labeling.load_calls(config.SEED_PATH)[0]
    assert saved["ground_truth_qa"] == gt
    # Everything else about the call is untouched.
    assert saved["call_type"] == _SEED_CALL["call_type"]
    assert saved["transcript"] == _SEED_CALL["transcript"]


def test_save_seed_label_creates_backup_once(tmp_seed_path, tmp_seed_backup_path):
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])
    assert not os.path.exists(config.SEED_LABELS_BACKUP_PATH)

    labeling.save_seed_label(_SEED_CALL["call_id"], _NEW_SCORES, _NEW_OBSERVED)

    backup = labeling.load_seed_backup()
    assert backup == {_SEED_CALL["call_id"]: _ORIGINAL_GT}


def test_backup_survives_multiple_relabels_unchanged(tmp_seed_path, tmp_seed_backup_path):
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])

    labeling.save_seed_label(_SEED_CALL["call_id"], _NEW_SCORES, _NEW_OBSERVED)
    # Re-label again with yet another set of scores/notes.
    third_scores = {k: 2 for k in DIMENSION_KEYS}
    labeling.save_seed_label(_SEED_CALL["call_id"], third_scores, "A third look, even stricter this time.")

    # The backup must still hold the very first (original) label.
    backup = labeling.load_seed_backup()
    assert backup == {_SEED_CALL["call_id"]: _ORIGINAL_GT}

    # But the seed file itself reflects the latest (third) save.
    saved = labeling.load_calls(config.SEED_PATH)[0]
    assert saved["ground_truth_qa"]["dimension_scores"] == third_scores


def test_backup_not_created_for_calls_without_prior_label(tmp_seed_path, tmp_seed_backup_path):
    unlabeled = {**_SEED_CALL, "call_id": "HB-2026-99999"}
    del unlabeled["ground_truth_qa"]
    _write_seed(config.SEED_PATH, [unlabeled])

    labeling.save_seed_label(unlabeled["call_id"], _NEW_SCORES, _NEW_OBSERVED)

    # Nothing to preserve — the call had no prior label.
    assert labeling.load_seed_backup() == {}


# ---------------------------------------------------------------------------
# compare_relabels.py
# ---------------------------------------------------------------------------

def test_compare_returns_empty_with_no_backup(tmp_seed_path, tmp_seed_backup_path):
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])
    assert compare() == []


def test_compare_reports_old_vs_new_and_deltas(tmp_seed_path, tmp_seed_backup_path):
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])
    labeling.save_seed_label(_SEED_CALL["call_id"], _NEW_SCORES, _NEW_OBSERVED)

    rows = compare()
    assert len(rows) == 1
    row = rows[0]
    assert row["call_id"] == _SEED_CALL["call_id"]
    assert row["old_overall"] == _ORIGINAL_GT["overall_score"]
    assert row["new_overall"] == round(labeling.weighted_overall(_NEW_SCORES), 1)
    assert row["delta_overall"] == round(row["new_overall"] - row["old_overall"], 2)

    expected_dim_deltas = {
        k: _NEW_SCORES[k] - _ORIGINAL_GT["dimension_scores"][k] for k in DIMENSION_KEYS
    }
    assert row["dimension_deltas"] == expected_dim_deltas


def test_compare_script_runs_end_to_end(tmp_seed_path, tmp_seed_backup_path, capsys):
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])
    labeling.save_seed_label(_SEED_CALL["call_id"], _NEW_SCORES, _NEW_OBSERVED)

    from scripts.compare_relabels import main
    main()

    out = capsys.readouterr().out
    assert _SEED_CALL["call_id"] in out
    assert "Mean absolute delta" in out


# ---------------------------------------------------------------------------
# AppTest — drives the real page's source selector, not just the helpers.
# ---------------------------------------------------------------------------

def test_seed_source_loads_and_prefills(monkeypatch, tmp_db, tmp_gpt_path, tmp_seed_path, tmp_seed_backup_path):
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])

    at = _at(monkeypatch, tmp_db, tmp_gpt_path)
    at.radio(key="label_source").set_value("Seed calls").run()
    assert not at.exception

    call_id = _SEED_CALL["call_id"]
    assert at.selectbox(key="label_call_selectbox").value == call_id
    scores = {
        sb.key: sb.value for sb in at.selectbox if sb.key.startswith(f"label_score_{call_id}")
    }
    for key, expected in _ORIGINAL_GT["dimension_scores"].items():
        assert scores[f"label_score_{call_id}_{key}"] == expected
    assert at.text_area(key=f"label_observed_{call_id}").value == _ORIGINAL_GT["reviewer_notes"]
    assert any("Update" in b.label for b in at.button)


def test_seed_source_save_creates_backup_and_overwrites(
    monkeypatch, tmp_db, tmp_gpt_path, tmp_seed_path, tmp_seed_backup_path
):
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
    _write_seed(config.SEED_PATH, [dict(_SEED_CALL)])

    at = _at(monkeypatch, tmp_db, tmp_gpt_path)
    at.radio(key="label_source").set_value("Seed calls").run()

    call_id = _SEED_CALL["call_id"]
    _fill_and_save(at, call_id, 2, _NEW_OBSERVED)

    # Backup preserves the pre-relabel label...
    assert labeling.load_seed_backup() == {call_id: _ORIGINAL_GT}
    # ...while the seed file now holds the new one.
    saved = labeling.load_calls(config.SEED_PATH)[0]
    assert saved["ground_truth_qa"]["dimension_scores"][DIMENSION_KEYS[0]] == 2
    assert saved["ground_truth_qa"]["reviewer_notes"] == f"Observed: {_NEW_OBSERVED}"


def _make_seed_call(call_id: str, score: int) -> dict:
    return {
        "call_id": call_id,
        "call_type": "appointment_scheduling",
        "duration_seconds": 100,
        "rep_id": "REP-1",
        "ground_truth_qa": {
            "overall_score": score,
            "reviewer_notes": f"Original notes for {call_id}.",
            "dimension_scores": {k: score for k in DIMENSION_KEYS},
        },
        "transcript": [{"timestamp": "00:00:00", "speaker": "rep", "text": f"hi {call_id}"}],
    }


def test_seed_source_update_advances_to_next_call_with_prefill(
    monkeypatch, tmp_db, tmp_gpt_path, tmp_seed_path, tmp_seed_backup_path
):
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
    first = _make_seed_call("HB-2026-AAAA", 4)
    second = _make_seed_call("HB-2026-BBBB", 3)
    _write_seed(config.SEED_PATH, [first, second])

    at = _at(monkeypatch, tmp_db, tmp_gpt_path)
    at.radio(key="label_source").set_value("Seed calls").run()
    assert at.selectbox(key="label_call_selectbox").value == "HB-2026-AAAA"

    _fill_and_save(at, "HB-2026-AAAA", 5, "Re-reviewed AAAA under the new guidelines.")

    # Advanced to the next seed call by order — not stuck (seed calls are
    # always all-labeled, so the old "next unlabeled" fallback never fired).
    assert at.selectbox(key="label_call_selectbox").value == "HB-2026-BBBB"

    # And its form shows ITS OWN stored label, not AAAA's edited values.
    scores = {
        sb.key: sb.value for sb in at.selectbox if sb.key.startswith("label_score_HB-2026-BBBB")
    }
    assert all(v == 3 for v in scores.values())
    assert at.text_area(key="label_observed_HB-2026-BBBB").value == "Original notes for HB-2026-BBBB."


def test_seed_source_last_call_shows_completion_toast(
    monkeypatch, tmp_db, tmp_gpt_path, tmp_seed_path, tmp_seed_backup_path
):
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
    _write_seed(config.SEED_PATH, [_make_seed_call("HB-2026-ONLY", 4)])

    at = _at(monkeypatch, tmp_db, tmp_gpt_path)
    at.radio(key="label_source").set_value("Seed calls").run()

    _fill_and_save(at, "HB-2026-ONLY", 5, "Final re-review of the only seed call.")

    assert not at.exception
    # No next call exists — stays put rather than crashing or wrapping.
    assert at.selectbox(key="label_call_selectbox").value == "HB-2026-ONLY"
    assert any("processed" in t.value for t in at.toast)
