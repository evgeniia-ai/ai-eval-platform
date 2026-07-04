"""Export a frozen demo dataset from qa.db for DEMO_MODE.

Selects the 6 most recent suite_runs (expected to be the clean, post-fix
matrix: 2 suites x 3 judge models) and every evaluation row referenced by
them (scores, judge reasoning, model, rubric_version, transcript — one row
per call_id x model), and writes a single pretty-printed
data/demo_results.json. No API keys or absolute paths are written; only
DB row contents.

Usage:
    python scripts/export_demo.py
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
from datetime import datetime, timezone

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config

OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "demo_results.json"
N_SUITE_RUNS = 6


def _matching_evaluation(conn: sqlite3.Connection, call_id: str, model: str, before: str) -> sqlite3.Row | None:
    """The evaluation row this suite run produced for `call_id`: the latest row
    for (call_id, model) at or before the suite run's own created_at timestamp."""
    return conn.execute(
        """
        SELECT * FROM evaluations
         WHERE call_id=? AND model=? AND created_at<=?
         ORDER BY created_at DESC LIMIT 1
        """,
        (call_id, model, before),
    ).fetchone()


def export() -> dict:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row

    suite_rows = conn.execute(
        "SELECT * FROM suite_runs ORDER BY created_at DESC LIMIT ?", (N_SUITE_RUNS,)
    ).fetchall()

    demo_suite_runs: list[dict] = []
    demo_evaluations: list[dict] = []
    seen_run_ids: set[int] = set()

    for sr in suite_rows:
        demo_suite_runs.append(dict(sr))
        for call_id in json.loads(sr["selected_call_ids"]):
            row = _matching_evaluation(conn, call_id, sr["model"], sr["created_at"])
            if row is None or row["run_id"] in seen_run_ids:
                continue
            seen_run_ids.add(row["run_id"])
            demo_evaluations.append(dict(row))

    conn.close()

    demo_suite_runs.sort(key=lambda r: r["suite_run_id"])
    demo_evaluations.sort(key=lambda r: r["run_id"])

    return {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "suite_runs": demo_suite_runs,
        "evaluations": demo_evaluations,
    }


def main() -> None:
    data = export()
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {OUT_PATH} — {len(data['suite_runs'])} suite runs, "
        f"{len(data['evaluations'])} evaluation rows."
    )


if __name__ == "__main__":
    main()
