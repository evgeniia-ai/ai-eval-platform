"""Read-only data-access layer for DEMO_MODE.

Mirrors the subset of src.storage's read API that src.app needs, backed by
the frozen data/demo_results.json export (see scripts/export_demo.py) instead
of qa.db. There is no write side — DEMO_MODE hides or disables every control
that would otherwise call storage.save() / update_review_status() or the
Anthropic API.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

from .models import Transcript, TranscriptEvaluation
from .rubric import DIMENSION_KEYS

DEMO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demo_results.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(DEMO_PATH, encoding="utf-8") as f:
        return json.load(f)


def _evaluations() -> list[dict]:
    return _load()["evaluations"]


def _latest_per_call(rows: list[dict]) -> list[dict]:
    """One row per call_id, keeping the highest run_id — mirrors storage._LATEST_JOIN."""
    best: dict[str, dict] = {}
    for r in rows:
        cid = r["call_id"]
        if cid not in best or r["run_id"] > best[cid]["run_id"]:
            best[cid] = r
    return list(best.values())


def _hydrate(row: dict) -> dict:
    out = dict(row)
    out["detail"] = TranscriptEvaluation.model_validate_json(out["detail_json"])
    if out.get("transcript_json"):
        out["transcript"] = Transcript.model_validate_json(out["transcript_json"])
    return out


def count() -> int:
    return len({r["call_id"] for r in _evaluations()})


def all_evaluations() -> list[dict]:
    rows = _latest_per_call(_evaluations())
    return sorted(rows, key=lambda r: r["run_id"], reverse=True)


def runs_for_call(call_id: str) -> list[dict]:
    rows = [r for r in _evaluations() if r["call_id"] == call_id]
    return sorted(rows, key=lambda r: r["run_id"], reverse=True)


def get_evaluation(call_id: str, run_id: Optional[int] = None) -> Optional[dict]:
    rows = [r for r in _evaluations() if r["call_id"] == call_id]
    if run_id is not None:
        rows = [r for r in rows if r["run_id"] == run_id]
    if not rows:
        return None
    row = max(rows, key=lambda r: r["run_id"])
    return _hydrate(row)


def rep_ids() -> list[str]:
    return sorted({r["rep_id"] for r in _evaluations()})


def evaluations_for_rep(rep_id: str) -> list[dict]:
    rows = [r for r in _evaluations() if r["rep_id"] == rep_id]
    rows = _latest_per_call(rows)
    return sorted(rows, key=lambda r: r["call_id"])


def rep_dimension_averages(rep_id: str) -> dict[str, float]:
    rows = evaluations_for_rep(rep_id)
    if not rows:
        return {}
    return {
        key: round(sum(r[key] for r in rows) / len(rows), 2)
        for key in DIMENSION_KEYS
    }


def all_suite_runs() -> list[dict]:
    return sorted(_load()["suite_runs"], key=lambda r: r["suite_run_id"], reverse=True)


def all_reviews() -> list[dict]:
    # The demo export carries no human_reviews data — the Review Queue page
    # renders its "no reviews yet" empty state in DEMO_MODE, which also means
    # none of its status-editing controls (real DB writes) ever render.
    return []
