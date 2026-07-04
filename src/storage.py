"""SQLite-backed store for evaluations, powering the dashboard, API, and coaching loop.

Stores the full evaluation JSON for drill-down, plus extracted columns
(rep_id, call_type, overall, per-dimension scores) for fast aggregation and trends.
Each judge run is a separate row; callers that want one result per call use the
"latest run" queries (all_evaluations, get_evaluation, evaluations_for_rep).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from . import config
from .models import Transcript, TranscriptEvaluation
from .qa_engine import Evaluation
from .rubric import DIMENSION_KEYS, RUBRIC_VERSION

_SCHEMA = """
CREATE TABLE IF NOT EXISTS suite_runs (
    suite_run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    suite_name        TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    rubric_version    TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    n_calls           INTEGER NOT NULL,
    n_ok              INTEGER NOT NULL,
    n_failed          INTEGER NOT NULL,
    n_skipped         INTEGER NOT NULL,
    selected_call_ids TEXT    NOT NULL,
    failed_call_ids   TEXT    NOT NULL,
    skipped_call_ids  TEXT    NOT NULL,
    mean_overall      REAL,
    mae_vs_gt         REAL
);
CREATE INDEX IF NOT EXISTS idx_suite_runs_name ON suite_runs(suite_name);
CREATE TABLE IF NOT EXISTS evaluations (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_label      TEXT    NOT NULL,
    call_id        TEXT    NOT NULL,
    rep_id         TEXT    NOT NULL,
    call_type      TEXT    NOT NULL,
    model          TEXT,
    rubric_version TEXT,
    overall_score  REAL    NOT NULL,
    greeting_identity_verification INTEGER,
    empathy_tone                   INTEGER,
    accuracy_completeness          INTEGER,
    protocol_adherence             INTEGER,
    closing_next_steps             INTEGER,
    detail_json     TEXT    NOT NULL,
    transcript_json TEXT,
    ground_truth_overall REAL,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_eval_call ON evaluations(call_id);
CREATE INDEX IF NOT EXISTS idx_eval_rep  ON evaluations(rep_id);
CREATE TABLE IF NOT EXISTS human_reviews (
    review_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id       TEXT    NOT NULL,
    run_id        INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'Pending',
    reason        TEXT    NOT NULL,
    reviewer_note TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_status  ON human_reviews(status);
CREATE INDEX IF NOT EXISTS idx_reviews_call_id ON human_reviews(call_id);
"""

# Selects only the latest run per call_id. Used wherever pages need one row per call.
_LATEST_JOIN = """
    INNER JOIN (
        SELECT call_id, MAX(run_id) AS max_run_id
        FROM evaluations GROUP BY call_id
    ) _latest ON evaluations.call_id = _latest.call_id
              AND evaluations.run_id = _latest.max_run_id
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with _conn():
        pass


def save(
    evaluation: Evaluation,
    transcript: Optional[Transcript] = None,
    model: Optional[str] = None,
    rubric_version: str = RUBRIC_VERSION,
) -> int:
    """Insert an evaluation row. Returns the new run_id."""
    scores = evaluation.dimension_scores()
    gt_overall = (
        transcript.ground_truth_qa.overall_score
        if transcript and transcript.ground_truth_qa
        else None
    )
    now = datetime.now(timezone.utc)
    run_label = now.strftime("%Y%m%d-%H%M%S")
    created_at = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    _model = model if model is not None else config.model()
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO evaluations
              (run_label, call_id, rep_id, call_type, model, rubric_version,
               overall_score,
               greeting_identity_verification, empathy_tone, accuracy_completeness,
               protocol_adherence, closing_next_steps,
               detail_json, transcript_json, ground_truth_overall, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_label,
                evaluation.call_id,
                evaluation.rep_id,
                evaluation.call_type,
                _model,
                rubric_version,
                evaluation.overall_score,
                scores["greeting_identity_verification"],
                scores["empathy_tone"],
                scores["accuracy_completeness"],
                scores["protocol_adherence"],
                scores["closing_next_steps"],
                evaluation.detail.model_dump_json(),
                transcript.model_dump_json() if transcript else None,
                gt_overall,
                created_at,
            ),
        )
        run_id = cur.lastrowid
    # Routing to human review happens here (not in callers) so both ingest.run()
    # and the Test runs page get it automatically — single point, no duplication.
    # Side effect is intentional: in this project we always want review-routing
    # when an evaluation is saved.

    from .review import needs_review
    reasons = needs_review(evaluation)
    if reasons:
        save_review(evaluation.call_id, run_id, reasons)

    return run_id


def all_evaluations() -> list[dict]:
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            f"SELECT evaluations.* FROM evaluations {_LATEST_JOIN}"
            " ORDER BY evaluations.run_id DESC"
        )]


def get_evaluation(call_id: str, run_id: Optional[int] = None) -> Optional[dict]:
    with _conn() as conn:
        if run_id is not None:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE run_id=?", (run_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE call_id=? ORDER BY run_id DESC LIMIT 1",
                (call_id,),
            ).fetchone()
    if not row:
        return None
    out = dict(row)
    out["detail"] = TranscriptEvaluation.model_validate_json(out["detail_json"])
    if out.get("transcript_json"):
        out["transcript"] = Transcript.model_validate_json(out["transcript_json"])
    return out


def get_run(run_id: int) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE run_id=?", (run_id,)
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    out["detail"] = TranscriptEvaluation.model_validate_json(out["detail_json"])
    if out.get("transcript_json"):
        out["transcript"] = Transcript.model_validate_json(out["transcript_json"])
    return out


def runs_for_call(call_id: str) -> list[dict]:
    """All runs for a call, newest first. Used by the Call detail run selector."""
    with _conn() as conn:
        rows = conn.execute(
            """SELECT run_id, run_label, model, rubric_version, created_at, overall_score,
                      greeting_identity_verification, empathy_tone, accuracy_completeness,
                      protocol_adherence, closing_next_steps
               FROM evaluations WHERE call_id=? ORDER BY run_id DESC""",
            (call_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_suite_run(
    suite_name: str,
    selected_call_ids: list[str],
    failed_call_ids: list[str],
    skipped_call_ids: list[str],
    ok_scores: list[float],
    gt_pairs: list[tuple[float, float]],
    model: str,
    rubric_version: str = RUBRIC_VERSION,
) -> int:
    """Insert one suite_runs row after a suite run completes. Returns suite_run_id."""
    n_ok = len(ok_scores)
    n_failed = len(failed_call_ids)
    n_skipped = len(skipped_call_ids)
    mean_overall = round(sum(ok_scores) / n_ok, 2) if ok_scores else None
    mae_vs_gt = (
        round(sum(abs(p - g) for p, g in gt_pairs) / len(gt_pairs), 2)
        if gt_pairs else None
    )
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO suite_runs
              (suite_name, model, rubric_version, created_at,
               n_calls, n_ok, n_failed, n_skipped,
               selected_call_ids, failed_call_ids, skipped_call_ids,
               mean_overall, mae_vs_gt)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                suite_name,
                model,
                rubric_version,
                created_at,
                n_ok + n_failed + n_skipped,
                n_ok,
                n_failed,
                n_skipped,
                json.dumps(sorted(selected_call_ids)),
                json.dumps(sorted(failed_call_ids)),
                json.dumps(sorted(skipped_call_ids)),
                mean_overall,
                mae_vs_gt,
            ),
        )
    return cur.lastrowid


def all_suite_runs() -> list[dict]:
    """All suite_runs rows, newest first."""
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM suite_runs ORDER BY suite_run_id DESC"
        )]


def rep_ids() -> list[str]:
    with _conn() as conn:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT rep_id FROM evaluations ORDER BY rep_id"
        )]


def evaluations_for_rep(rep_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT evaluations.* FROM evaluations {_LATEST_JOIN}"
            " WHERE evaluations.rep_id=? ORDER BY call_id",
            (rep_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def rep_dimension_averages(rep_id: str) -> dict[str, float]:
    """Mean score per dimension for a rep, across all their evaluated calls (latest run per call)."""
    rows = evaluations_for_rep(rep_id)
    if not rows:
        return {}
    return {
        key: round(sum(r[key] for r in rows) / len(rows), 2)
        for key in DIMENSION_KEYS
    }


def count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(DISTINCT call_id) FROM evaluations").fetchone()[0]


# ---------------------------------------------------------------------------
# Human review queue
# ---------------------------------------------------------------------------

def save_review(call_id: str, run_id: int, reasons: list[str]) -> int:
    """Insert a Pending review row, or point the existing pending one at the latest run.

    A call already awaiting review must not accumulate one row per judge run —
    dedupe is keyed on call_id + Pending status (not run_id, which is unique
    per run and would never match across repeated evaluations of the same call).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    with _conn() as conn:
        existing = conn.execute(
            "SELECT review_id FROM human_reviews WHERE call_id=? AND status='Pending'",
            (call_id,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE human_reviews SET run_id=?, reason=?, updated_at=? WHERE review_id=?",
                (run_id, ",".join(reasons), now, existing["review_id"]),
            )
            return existing["review_id"]
        cur = conn.execute(
            """
            INSERT INTO human_reviews (call_id, run_id, status, reason, created_at)
            VALUES (?, ?, 'Pending', ?, ?)
            """,
            (call_id, run_id, ",".join(reasons), now),
        )
        return cur.lastrowid


def pending_reviews() -> list[dict]:
    """All rows with status='Pending', newest first."""
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM human_reviews WHERE status='Pending' ORDER BY review_id DESC"
        )]


def all_reviews() -> list[dict]:
    """All review rows, newest first."""
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM human_reviews ORDER BY review_id DESC"
        )]


def update_review_status(review_id: int, status: str, reviewer_note: Optional[str] = None) -> None:
    """Set status, reviewer_note, and updated_at on an existing review row."""
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    with _conn() as conn:
        conn.execute(
            """
            UPDATE human_reviews
               SET status=?, reviewer_note=?, updated_at=?
             WHERE review_id=?
            """,
            (status, reviewer_note, updated_at, review_id),
        )
