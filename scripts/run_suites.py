"""CLI batch runner for the Smoke / Regression eval suites.

Drives the exact same pipeline as the Test runs dashboard page
(evaluate -> storage.save -> build_gt_pairs -> storage.save_suite_run, via
src.ingest.run_suite) so a run launched from here is indistinguishable in the
DB from one launched through Streamlit.

Usage:
    python scripts/run_suites.py --suite smoke --model claude-sonnet-4-6
    python scripts/run_suites.py --all              # both suites x 3 models, 6 runs
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import ingest, storage
from src.test_sets import REGRESSION_SET, SMOKE_SET

SUITES: dict[str, tuple[str, list[str]]] = {
    "smoke": ("Smoke", SMOKE_SET),
    "regression": ("Regression", REGRESSION_SET),
}

# The three judge models offered on the dashboard (src/app.py's _JUDGE_MODELS).
# Kept in sync manually since the dashboard list carries UI-only display labels.
ALL_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
]


def _run_one(suite_key: str, model_id: str, transcript_map: dict) -> dict:
    suite_name, call_ids = SUITES[suite_key]
    print(f"\n=== {suite_name} / {model_id} ({len(call_ids)} calls) ===")
    os.environ["ANTHROPIC_MODEL"] = model_id

    def _progress(done: int, total: int, msg: str) -> None:
        print(f"  [{done}/{total}] {msg}")

    result = ingest.run_suite(
        call_ids=call_ids,
        transcript_map=transcript_map,
        suite_name=suite_name,
        model=model_id,
        progress=_progress,
    )
    n_ok = len(result["ok_scores"])
    n_failed = len(result["failed_ids"])
    mean_overall = result["mean_overall"]
    mae_vs_gt = result["mae_vs_gt"]
    print(
        f"  -> ok={n_ok} failed={n_failed} "
        f"mean_overall={mean_overall if mean_overall is not None else '—'} "
        f"mae_vs_gt={mae_vs_gt if mae_vs_gt is not None else '—'}"
    )
    return {
        "suite": suite_name,
        "model": model_id,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "mean_overall": mean_overall,
        "mae_vs_gt": mae_vs_gt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run QA eval suites from the CLI.")
    parser.add_argument("--suite", choices=sorted(SUITES), help="Which suite to run.")
    parser.add_argument("--model", help="Judge model id, e.g. claude-sonnet-4-6.")
    parser.add_argument(
        "--all", action="store_true",
        help="Run both suites x all 3 judge models (6 runs total).",
    )
    args = parser.parse_args()

    if not args.all and not (args.suite and args.model):
        parser.error("either --all, or both --suite and --model, are required")

    storage.init()
    transcript_map = {t.call_id: t for t in ingest.load_transcripts(include_generated=True)}

    if args.all:
        jobs = [(suite_key, model_id) for suite_key in SUITES for model_id in ALL_MODELS]
    else:
        jobs = [(args.suite, args.model)]

    summary = [_run_one(suite_key, model_id, transcript_map) for suite_key, model_id in jobs]

    print("\n=== Summary (sorted by suite, then MAE) ===")
    header = f"{'suite':<12}{'model':<28}{'ok/failed':<11}{'mean':<8}{'MAE vs GT'}"
    print(header)
    print("-" * len(header))
    for row in sorted(
        summary,
        key=lambda r: (r["suite"], r["mae_vs_gt"] if r["mae_vs_gt"] is not None else float("inf")),
    ):
        ok_fail = f"{row['n_ok']}/{row['n_failed']}"
        mean = f"{row['mean_overall']:.2f}" if row["mean_overall"] is not None else "—"
        mae = f"{row['mae_vs_gt']:.2f}" if row["mae_vs_gt"] is not None else "—"
        print(f"{row['suite']:<12}{row['model']:<28}{ok_fail:<11}{mean:<8}{mae}")


if __name__ == "__main__":
    main()
