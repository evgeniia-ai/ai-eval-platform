"""Tests for src/labeling.py (human ground-truth labeling of GPT-generated
calls) and the Labeling page's UX behavior in src/app.py.

Two layers:
- Pure logic (parse/build notes, next-unlabeled lookup, save/validate) — no
  Streamlit dependency, fast.
- AppTest-driven (auto-advance, pre-fill, no state leak) — drives the real
  src/app.py script via streamlit.testing.v1.AppTest, so it exercises the
  actual widget/session_state interactions, not just the helpers behind them.
"""

import json
import os
import pathlib
import subprocess
import sys

from src import config, labeling
from src.rubric import DIMENSION_KEYS

_UNLABELED_CALL = {
    "call_id": "HB-GPT-0001",
    "call_type": "billing_inquiry",
    "duration_seconds": 60,
    "rep_id": "REP-GPT-01",
    "transcript": [
        {"timestamp": "00:00:00", "speaker": "rep", "text": "Thank you for calling HealthBridge."},
        {"timestamp": "00:00:05", "speaker": "patient", "text": "Hi, I have a billing question."},
    ],
}

_FULL_SCORES = {k: 4 for k in DIMENSION_KEYS}
_OBSERVED = "Rep verified full name and DOB before pulling up the account."
_CONCERN = "Confirmed appointment date was not restated back to the patient."
_IMPACT = "Patient may leave the call unsure of the exact date and miss it."


def _write_calls(path, calls):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calls, f)


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def test_save_label_adds_valid_ground_truth_qa(tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])

    gt = labeling.save_label(_UNLABELED_CALL["call_id"], _FULL_SCORES, _OBSERVED)

    assert gt["dimension_scores"] == _FULL_SCORES
    assert gt["reviewer_notes"] == f"Observed: {_OBSERVED}"
    assert gt["overall_score"] == 4.0

    calls = labeling.load_calls()
    saved = next(c for c in calls if c["call_id"] == _UNLABELED_CALL["call_id"])
    assert saved["ground_truth_qa"] == gt
    assert saved["call_type"] == _UNLABELED_CALL["call_type"]
    assert saved["transcript"] == _UNLABELED_CALL["transcript"]


def test_save_label_combines_observed_concern_impact(tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])

    gt = labeling.save_label(
        _UNLABELED_CALL["call_id"], _FULL_SCORES, _OBSERVED, _CONCERN, _IMPACT
    )

    assert gt["reviewer_notes"] == (
        f"Observed: {_OBSERVED}\nConcern: {_CONCERN}\nImpact: {_IMPACT}"
    )


def test_build_reviewer_notes_omits_empty_optional_sections():
    assert labeling.build_reviewer_notes(_OBSERVED) == f"Observed: {_OBSERVED}"
    assert labeling.build_reviewer_notes(_OBSERVED, concern="  ", impact="") == f"Observed: {_OBSERVED}"
    assert labeling.build_reviewer_notes(_OBSERVED, concern=_CONCERN) == (
        f"Observed: {_OBSERVED}\nConcern: {_CONCERN}"
    )


def test_parse_reviewer_notes_round_trips_with_build():
    notes = labeling.build_reviewer_notes(_OBSERVED, _CONCERN, _IMPACT)
    assert labeling.parse_reviewer_notes(notes) == (_OBSERVED, _CONCERN, _IMPACT)


def test_parse_reviewer_notes_round_trips_observed_only():
    notes = labeling.build_reviewer_notes(_OBSERVED)
    assert labeling.parse_reviewer_notes(notes) == (_OBSERVED, "", "")


def test_parse_reviewer_notes_falls_back_for_freeform_text():
    freeform = "Strong greeting and empathy. Minor gap in closing."
    assert labeling.parse_reviewer_notes(freeform) == (freeform, "", "")


def test_save_label_decreases_unlabeled_count(tmp_gpt_path):
    other_call = {**_UNLABELED_CALL, "call_id": "HB-GPT-0002"}
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL), other_call])

    labeled_before, total_before = labeling.labeled_count(labeling.load_calls())
    assert (labeled_before, total_before) == (0, 2)

    labeling.save_label(_UNLABELED_CALL["call_id"], _FULL_SCORES, _OBSERVED)

    labeled_after, total_after = labeling.labeled_count(labeling.load_calls())
    assert (labeled_after, total_after) == (1, 2)


def test_save_label_overwrites_existing_label(tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])
    labeling.save_label(_UNLABELED_CALL["call_id"], _FULL_SCORES, _OBSERVED)

    new_scores = {k: 2 for k in DIMENSION_KEYS}
    new_observed = "Completely different observation replacing the first one entirely."
    gt = labeling.save_label(_UNLABELED_CALL["call_id"], new_scores, new_observed)

    assert gt["dimension_scores"] == new_scores
    assert gt["reviewer_notes"] == f"Observed: {new_observed}"

    saved = labeling.load_calls()[0]
    assert saved["ground_truth_qa"] == gt  # fully replaced, not merged


def test_save_label_rejects_missing_scores(tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])
    incomplete = {**_FULL_SCORES, "closing_next_steps": None}

    try:
        labeling.save_label(_UNLABELED_CALL["call_id"], incomplete, _OBSERVED)
        assert False, "expected ValueError for missing score"
    except ValueError as e:
        assert "Missing score" in str(e)

    calls = labeling.load_calls()
    assert not labeling.is_labeled(calls[0])


def test_save_label_rejects_short_observed(tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])

    try:
        labeling.save_label(_UNLABELED_CALL["call_id"], _FULL_SCORES, "too short")
        assert False, "expected ValueError for short Observed"
    except ValueError as e:
        assert "Observed" in str(e)


def test_save_label_allows_empty_concern_and_impact(tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])
    gt = labeling.save_label(_UNLABELED_CALL["call_id"], _FULL_SCORES, _OBSERVED, "", "")
    assert gt["reviewer_notes"] == f"Observed: {_OBSERVED}"


def test_next_unlabeled_call_id_finds_next_in_order():
    ids = ["A", "B", "C", "D"]
    by_id = {
        "A": {"ground_truth_qa": {}},
        "B": {},
        "C": {},
        "D": {"ground_truth_qa": {}},
    }
    assert labeling.next_unlabeled_call_id(ids, by_id, "A") == "B"


def test_next_unlabeled_call_id_wraps_around():
    ids = ["A", "B", "C"]
    by_id = {"A": {}, "B": {"ground_truth_qa": {}}, "C": {"ground_truth_qa": {}}}
    assert labeling.next_unlabeled_call_id(ids, by_id, "C") == "A"


def test_next_unlabeled_call_id_none_when_all_labeled():
    ids = ["A", "B"]
    by_id = {"A": {"ground_truth_qa": {}}, "B": {"ground_truth_qa": {}}}
    assert labeling.next_unlabeled_call_id(ids, by_id, "A") is None


def test_next_call_id_in_order_advances_by_one():
    ids = ["A", "B", "C"]
    assert labeling.next_call_id_in_order(ids, "A") == "B"
    assert labeling.next_call_id_in_order(ids, "B") == "C"


def test_next_call_id_in_order_none_at_last_no_wrap():
    ids = ["A", "B", "C"]
    assert labeling.next_call_id_in_order(ids, "C") is None


def test_next_call_id_in_order_none_when_not_found():
    assert labeling.next_call_id_in_order(["A", "B"], "Z") is None


def test_labeling_page_present_in_both_modes():
    # Labeling is a read-only showcase in demo mode now (see
    # tests/test_labeling_demo_mode.py), not hidden — it's present in PAGES
    # either way, just disabled/read-only when DEMO_MODE is true.
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])

    def _page_names(demo_mode: bool) -> list[str]:
        env = dict(os.environ)
        env["DEMO_MODE"] = "true" if demo_mode else "false"
        result = subprocess.run(
            [sys.executable, "-c", "import src.app as a, json; print(json.dumps(sorted(a.PAGES.keys())))"],
            cwd=repo_root, env=env, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert "Labeling" in _page_names(demo_mode=False)
    assert "Labeling" in _page_names(demo_mode=True)


# ---------------------------------------------------------------------------
# AppTest — drives the real page, catches Streamlit widget/session_state bugs
# that the pure-logic tests above can't see.
# ---------------------------------------------------------------------------

def _two_call_fixture():
    base = dict(_UNLABELED_CALL)
    return [base, {**base, "call_id": "HB-GPT-0002"}]


def _at(monkeypatch, tmp_db, tmp_gpt_path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DEMO_MODE", "false")
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "src" / "app.py"))
    at.run()
    at.radio(key=at.radio[0].key).set_value("Labeling").run()
    assert not at.exception
    return at


def _fill_and_save(at, call_id: str, score: int, observed: str):
    for sb in at.selectbox:
        if sb.key.startswith(f"label_score_{call_id}"):
            sb.set_value(score)
    at.text_area(key=f"label_observed_{call_id}").set_value(observed)
    at.run()
    button = next(b for b in at.button if "Save" in b.label or "Update" in b.label)
    button.click().run()
    assert not at.exception


def test_save_auto_advances_to_next_unlabeled(monkeypatch, tmp_db, tmp_gpt_path):
    _write_calls(config.GPT_PATH, _two_call_fixture())
    at = _at(monkeypatch, tmp_db, tmp_gpt_path)

    assert at.selectbox(key="label_call_selectbox").value == "HB-GPT-0001"
    _fill_and_save(at, "HB-GPT-0001", 4, _OBSERVED)

    assert at.selectbox(key="label_call_selectbox").value == "HB-GPT-0002"


def test_form_state_does_not_leak_between_calls(monkeypatch, tmp_db, tmp_gpt_path):
    _write_calls(config.GPT_PATH, _two_call_fixture())
    at = _at(monkeypatch, tmp_db, tmp_gpt_path)
    _fill_and_save(at, "HB-GPT-0001", 4, _OBSERVED)

    # Now on HB-GPT-0002 (unlabeled) — its widgets must be blank, not carry
    # over HB-GPT-0001's scores/text.
    scores_0002 = [sb.value for sb in at.selectbox if sb.key.startswith("label_score_HB-GPT-0002")]
    assert scores_0002 == [None] * len(DIMENSION_KEYS)
    assert at.text_area(key="label_observed_HB-GPT-0002").value == ""
    assert at.text_area(key="label_concern_HB-GPT-0002").value == ""
    assert at.text_area(key="label_impact_HB-GPT-0002").value == ""


def test_all_labeled_shows_celebration_banner(monkeypatch, tmp_db, tmp_gpt_path):
    _write_calls(config.GPT_PATH, [dict(_UNLABELED_CALL)])  # single call
    at = _at(monkeypatch, tmp_db, tmp_gpt_path)
    _fill_and_save(at, "HB-GPT-0001", 4, _OBSERVED)

    assert any("All calls labeled" in s.value for s in at.success)


def test_editing_labeled_call_prefills_and_overwrites(monkeypatch, tmp_db, tmp_gpt_path):
    labeled_call = {
        **_UNLABELED_CALL,
        "ground_truth_qa": {
            "overall_score": 3.0,
            "reviewer_notes": labeling.build_reviewer_notes(_OBSERVED, _CONCERN, _IMPACT),
            "dimension_scores": {k: 3 for k in DIMENSION_KEYS},
        },
    }
    _write_calls(config.GPT_PATH, [labeled_call])
    at = _at(monkeypatch, tmp_db, tmp_gpt_path)

    # Pre-filled from the existing label.
    scores = [sb.value for sb in at.selectbox if sb.key.startswith("label_score_HB-GPT-0001")]
    assert all(v == 3 for v in scores)
    assert at.text_area(key="label_observed_HB-GPT-0001").value == _OBSERVED
    assert at.text_area(key="label_concern_HB-GPT-0001").value == _CONCERN
    assert at.text_area(key="label_impact_HB-GPT-0001").value == _IMPACT
    assert any("Update" in b.label for b in at.button)

    # Editing overwrites the existing label — including clearing the
    # pre-filled Concern/Impact, which must not silently survive the save.
    new_observed = "A completely revised observation after re-review of the call."
    at.text_area(key="label_concern_HB-GPT-0001").set_value("")
    at.text_area(key="label_impact_HB-GPT-0001").set_value("")
    _fill_and_save(at, "HB-GPT-0001", 5, new_observed)

    saved = labeling.load_calls()[0]["ground_truth_qa"]
    assert saved["dimension_scores"] == {k: 5 for k in DIMENSION_KEYS}
    assert saved["reviewer_notes"] == f"Observed: {new_observed}"
