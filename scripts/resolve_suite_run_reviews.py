"""One-off cleanup: bulk-resolve human_reviews rows created by suite runs.

Before the review-routing overhaul, every evaluation saved during a suite run
(Smoke/Regression/Full — calibration experiments on the judge) was routed
through the same review logic as production traffic, flooding the queue with
noise. ingest.run_suite() now disables routing entirely for suite runs (see
`route_to_review` in src/storage.save), so this backlog will not grow further
— this script cleans up what already accumulated.

A review row is identified as a suite-run artifact by joining its `run_id`
back to the evaluation that suite run produced for that call_id + model (the
same match used by scripts/export_demo.py). Only rows still 'Pending' are
touched — a review a human has already triaged (Confirmed Issue / False
Alarm / Needs Rubric Update) is left alone, since that judgment stands
regardless of how the row originated.

Usage:
    python scripts/resolve_suite_run_reviews.py            # apply
    python scripts/resolve_suite_run_reviews.py --dry-run   # preview only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config, storage

BULK_RESOLVE_STATUS = "Resolved"
BULK_RESOLVE_NOTE = "bulk-resolved: calibration run artifact"


def _matching_evaluation_run_id(conn: sqlite3.Connection, call_id: str, model: str, before: str) -> int | None:
    row = conn.execute(
        """
        SELECT run_id FROM evaluations
         WHERE call_id=? AND model=? AND created_at<=?
         ORDER BY created_at DESC LIMIT 1
        """,
        (call_id, model, before),
    ).fetchone()
    return row[0] if row else None


def suite_run_evaluation_run_ids(conn: sqlite3.Connection) -> set[int]:
    """run_ids of evaluations produced by any suite run (Smoke/Regression/Full)."""
    run_ids: set[int] = set()
    for sr in conn.execute("SELECT * FROM suite_runs").fetchall():
        for call_id in json.loads(sr["selected_call_ids"]):
            run_id = _matching_evaluation_run_id(conn, call_id, sr["model"], sr["created_at"])
            if run_id is not None:
                run_ids.add(run_id)
    return run_ids


def find_reviews_to_resolve(conn: sqlite3.Connection) -> list[dict]:
    """Still-Pending human_reviews rows whose run_id traces back to a suite run."""
    suite_run_ids = suite_run_evaluation_run_ids(conn)
    if not suite_run_ids:
        return []
    placeholders = ",".join("?" * len(suite_run_ids))
    rows = conn.execute(
        f"SELECT * FROM human_reviews WHERE status='Pending' AND run_id IN ({placeholders})",
        tuple(suite_run_ids),
    ).fetchall()
    return [dict(r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-resolve pending reviews created by suite runs.")
    parser.add_argument("--dry-run", action="store_true", help="List what would change, without writing.")
    args = parser.parse_args()

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    to_resolve = find_reviews_to_resolve(conn)
    conn.close()

    if not to_resolve:
        print("Nothing to resolve — no pending reviews trace back to a suite run.")
        return

    verb = "Would resolve" if args.dry_run else "Resolving"
    print(f"{verb} {len(to_resolve)} review(s):")
    for row in to_resolve:
        print(f"  #{row['review_id']}  {row['call_id']}  run_id={row['run_id']}  reason={row['reason']}")
        if not args.dry_run:
            storage.update_review_status(row["review_id"], BULK_RESOLVE_STATUS, BULK_RESOLVE_NOTE)

    if not args.dry_run:
        print(f"\nDone — {len(to_resolve)} review(s) marked '{BULK_RESOLVE_STATUS}'.")


if __name__ == "__main__":
    main()
