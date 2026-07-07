"""Export a frozen demo dataset from qa.db for DEMO_MODE.

Selects the 3 most recent Full suite_runs (33 calls x 3 judge models) and
every evaluation row referenced by them (scores, judge reasoning, model,
rubric_version, transcript — one row per call_id x model), the full
human_reviews table (so the Review Queue page has something to render in
demo mode), and labeled-ground-truth stats (n human-labeled calls, holdout
size). Smoke/Regression suite_runs are deliberately NOT exported — they're
stale pre-guidelines runs and would only be shown, never re-run, in demo
mode, so there's no reason to ship them. Writes a single pretty-printed
data/demo_results.json. No API keys or absolute paths are written; only DB
row contents plus the two small derived stats.

Usage:
    python scripts/export_demo.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
from datetime import datetime, timezone

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config, ingest, labeling

OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "demo_results.json"
N_FULL_RUNS = 3


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


def _labeling_stats() -> dict:
    """n human-labeled calls (seed + labeled GPT) and the held-out split size."""
    seed_calls = ingest.load_transcripts(include_generated=False)
    n_seed_labeled = sum(1 for t in seed_calls if t.ground_truth_qa is not None)

    gpt_calls = labeling.load_calls()
    n_gpt_labeled = sum(1 for c in gpt_calls if labeling.is_labeled(c))

    n_holdout = 0
    if os.path.exists(config.HOLDOUT_PATH):
        with open(config.HOLDOUT_PATH, encoding="utf-8") as f:
            n_holdout = len(json.load(f))

    return {
        "n_labeled": n_seed_labeled + n_gpt_labeled,
        "n_seed_labeled": n_seed_labeled,
        "n_gpt_labeled": n_gpt_labeled,
        "n_holdout": n_holdout,
    }


def export() -> dict:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row

    suite_rows = conn.execute(
        "SELECT * FROM suite_runs WHERE suite_name='Full' ORDER BY created_at DESC LIMIT ?",
        (N_FULL_RUNS,),
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

    demo_reviews = [dict(r) for r in conn.execute("SELECT * FROM human_reviews").fetchall()]

    conn.close()

    demo_suite_runs.sort(key=lambda r: r["suite_run_id"])
    demo_evaluations.sort(key=lambda r: r["run_id"])
    demo_reviews.sort(key=lambda r: r["review_id"])

    return {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "suite_runs": demo_suite_runs,
        "evaluations": demo_evaluations,
        "human_reviews": demo_reviews,
        "labeling_stats": _labeling_stats(),
        # Full snapshot of both labeling sources, so the Labeling page can
        # render read-only in DEMO_MODE without ever touching the live files.
        "gpt_calls": labeling.load_calls(config.GPT_PATH),
        "seed_calls": labeling.load_calls(config.SEED_PATH),
    }


def main() -> None:
    data = export()
    OUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Wrote {OUT_PATH} — {len(data['suite_runs'])} suite runs, "
        f"{len(data['evaluations'])} evaluation rows, "
        f"{len(data['human_reviews'])} review rows, "
        f"{len(data['gpt_calls'])} gpt_calls, {len(data['seed_calls'])} seed_calls, "
        f"labeling_stats={data['labeling_stats']}."
    )


if __name__ == "__main__":
    main()
