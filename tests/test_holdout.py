"""Tests for scripts/make_holdout.py (stratified holdout selection) and
src/test_sets.py's FULL_SET (seed + labeled GPT calls, minus holdout).
"""

import json
from collections import Counter

from scripts.make_holdout import compute_holdout_ids
from src import config
from src.test_sets import full_set


def _labeled_call(call_id: str, call_type: str) -> dict:
    return {
        "call_id": call_id,
        "call_type": call_type,
        "duration_seconds": 60,
        "rep_id": "REP-GPT-01",
        "transcript": [
            {"timestamp": "00:00:00", "speaker": "rep", "text": "hi"},
            {"timestamp": "00:00:05", "speaker": "patient", "text": "hey"},
        ],
        "ground_truth_qa": {
            "overall_score": 4.0,
            "reviewer_notes": "Observed: fine.",
            "dimension_scores": {"greeting_identity_verification": 4},
        },
    }


def _unlabeled_call(call_id: str, call_type: str) -> dict:
    c = _labeled_call(call_id, call_type)
    del c["ground_truth_qa"]
    return c


# ---------------------------------------------------------------------------
# compute_holdout_ids
# ---------------------------------------------------------------------------

def test_compute_holdout_ids_stratified_by_call_type():
    calls = (
        [_labeled_call(f"HB-GPT-A{i:02d}", "billing_inquiry") for i in range(10)]
        + [_labeled_call(f"HB-GPT-B{i:02d}", "clinical_triage") for i in range(5)]
    )
    holdout = compute_holdout_ids(calls, fraction=0.2, seed=1)

    by_type = {c["call_id"]: c["call_type"] for c in calls}
    counts = Counter(by_type[cid] for cid in holdout)
    assert counts["billing_inquiry"] == 2  # 20% of 10
    assert counts["clinical_triage"] == 1  # 20% of 5
    assert len(holdout) == 3


def test_compute_holdout_ids_matches_task_scale():
    # Mirrors the real data/gpt_transcripts.json distribution: 7,7,6,6,6 = 32
    # labeled calls -> round(32*0.2) = 6, largest-remainder apportioned.
    sizes = {
        "appointment_scheduling": 7,
        "billing_inquiry": 7,
        "clinical_triage": 6,
        "insurance_verification": 6,
        "prescription_refill": 6,
    }
    calls = [
        _labeled_call(f"HB-GPT-{ct}-{i:02d}", ct)
        for ct, n in sizes.items()
        for i in range(n)
    ]
    holdout = compute_holdout_ids(calls)  # default fraction/seed
    assert len(holdout) == 6

    by_type = {c["call_id"]: c["call_type"] for c in calls}
    counts = Counter(by_type[cid] for cid in holdout)
    assert counts["appointment_scheduling"] == 2  # largest remainder tie-break
    assert counts["billing_inquiry"] == 1
    assert counts["clinical_triage"] == 1
    assert counts["insurance_verification"] == 1
    assert counts["prescription_refill"] == 1


def test_compute_holdout_ids_ignores_unlabeled_calls():
    calls = (
        [_labeled_call(f"HB-GPT-A{i:02d}", "billing_inquiry") for i in range(5)]
        + [_unlabeled_call(f"HB-GPT-U{i:02d}", "billing_inquiry") for i in range(20)]
    )
    holdout = compute_holdout_ids(calls, fraction=0.2, seed=1)
    assert holdout  # sanity: something was selected
    assert all(cid.startswith("HB-GPT-A") for cid in holdout)


def test_compute_holdout_ids_deterministic_with_fixed_seed():
    calls = [_labeled_call(f"HB-GPT-{i:04d}", "billing_inquiry") for i in range(10)]
    first = compute_holdout_ids(calls, fraction=0.3, seed=99)
    second = compute_holdout_ids(calls, fraction=0.3, seed=99)
    assert first == second


def test_compute_holdout_ids_different_seed_changes_selection():
    calls = [_labeled_call(f"HB-GPT-{i:04d}", "billing_inquiry") for i in range(20)]
    a = compute_holdout_ids(calls, fraction=0.2, seed=1)
    b = compute_holdout_ids(calls, fraction=0.2, seed=2)
    assert a != b


# ---------------------------------------------------------------------------
# FULL_SET / full_set()
# ---------------------------------------------------------------------------

def test_full_set_excludes_holdout_and_unlabeled(tmp_seed_path, tmp_gpt_path, tmp_holdout_path):
    with open(config.SEED_PATH, "w", encoding="utf-8") as f:
        json.dump([{
            "call_id": "HB-2026-SEED1",
            "call_type": "billing_inquiry",
            "duration_seconds": 60,
            "rep_id": "REP-1",
            "transcript": [
                {"timestamp": "00:00:00", "speaker": "rep", "text": "hi"},
                {"timestamp": "00:00:05", "speaker": "patient", "text": "hey"},
            ],
        }], f)

    gpt_calls = [
        _labeled_call("HB-GPT-0001", "billing_inquiry"),
        _labeled_call("HB-GPT-0002", "billing_inquiry"),  # will be held out
        _unlabeled_call("HB-GPT-0003", "billing_inquiry"),  # not yet labeled
    ]
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump(gpt_calls, f)

    with open(config.HOLDOUT_PATH, "w", encoding="utf-8") as f:
        json.dump(["HB-GPT-0002"], f)

    result = full_set()

    assert result == frozenset({"HB-2026-SEED1", "HB-GPT-0001"})


def test_full_set_with_no_holdout_file_includes_all_labeled(tmp_seed_path, tmp_gpt_path, tmp_holdout_path):
    with open(config.SEED_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)
    with open(config.GPT_PATH, "w", encoding="utf-8") as f:
        json.dump([_labeled_call("HB-GPT-0001", "billing_inquiry")], f)
    # config.HOLDOUT_PATH deliberately left unwritten — make_holdout.py hasn't run yet.

    assert full_set() == frozenset({"HB-GPT-0001"})
