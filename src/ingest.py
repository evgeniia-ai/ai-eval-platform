"""Pipeline: load transcripts (seed + generated), evaluate each, store results.

Usage:
    python -m src.ingest            # load seed + generated, evaluate, save to DB
    python -m src.ingest --seed     # seed transcripts only
"""

from __future__ import annotations

import json
import os
import sys

from . import config, storage
from .data_gen import build_gt_pairs
from .models import Transcript
from .qa_engine import QAError, evaluate


def load_transcripts(include_generated: bool = True) -> list[Transcript]:
    transcripts: list[Transcript] = []
    with open(config.SEED_PATH) as f:
        for raw in json.load(f):
            transcripts.append(Transcript.model_validate(raw))
    if include_generated and os.path.exists(config.GENERATED_PATH):
        with open(config.GENERATED_PATH) as f:
            for raw in json.load(f):
                raw.pop("_scenario", None)
                transcripts.append(Transcript.model_validate(raw))
    return transcripts


def run(include_generated: bool = True, progress=None) -> tuple[int, int]:
    """Evaluate and store all transcripts. Returns (succeeded, failed)."""
    storage.init()
    transcripts = load_transcripts(include_generated)
    ok = fail = 0
    for i, t in enumerate(transcripts):
        try:
            evaluation = evaluate(t)
            storage.save(evaluation, transcript=t)
            ok += 1
            if progress:
                progress(i + 1, len(transcripts), f"{t.call_id} -> {evaluation.overall_score}")
        except QAError as e:
            fail += 1
            if progress:
                progress(i + 1, len(transcripts), f"{t.call_id} FAILED: {e}")
    return ok, fail


def run_suite(
    call_ids: list[str],
    transcript_map: dict[str, Transcript],
    suite_name: str,
    model: str,
    progress=None,
) -> dict:
    """Evaluate `call_ids`, saving each result and a `suite_runs` row for the batch.

    Shared by the Test runs dashboard page and the `scripts/run_suites.py` CLI so
    a suite run looks identical (same fields, same GT-filtering logic) regardless
    of where it was launched from. `progress(done, total, msg)` is called after
    each call_id is processed if provided.
    """
    n = len(call_ids)
    ok_scores: list[float] = []
    failed_msgs: list[str] = []       # "call_id: error" for display
    failed_ids: list[str] = []        # bare call_ids for storage
    skipped_ids: list[str] = []       # bare call_ids for storage
    raw_results: list[tuple[str, float, float | None]] = []  # (call_id, predicted, ground_truth)

    try:
        for i, cid in enumerate(call_ids):
            t = transcript_map.get(cid)
            if t is None:
                skipped_ids.append(cid)
                if progress:
                    progress(i + 1, n, f"{cid}: skipped (no transcript on disk)")
                continue
            try:
                ev = evaluate(t)
                # Calibration/suite runs are experiments on the judge, not
                # production traffic — never route them to human review.
                storage.save(ev, transcript=t, route_to_review=False)
                ok_scores.append(ev.overall_score)
                gt_overall = t.ground_truth_qa.overall_score if t.ground_truth_qa else None
                raw_results.append((cid, ev.overall_score, gt_overall))
                if progress:
                    progress(i + 1, n, f"{cid} -> {ev.overall_score}")
            except QAError as e:
                failed_ids.append(cid)
                failed_msgs.append(f"{cid}: {e}")
                if progress:
                    progress(i + 1, n, f"{cid} FAILED: {e}")
            except Exception as e:
                failed_ids.append(cid)
                failed_msgs.append(f"{cid}: {type(e).__name__}: {e}")
                if progress:
                    progress(i + 1, n, f"{cid} FAILED: {type(e).__name__}: {e}")
    finally:
        gt_pairs = build_gt_pairs(raw_results)
        suite_run_id = storage.save_suite_run(
            suite_name=suite_name,
            selected_call_ids=call_ids,
            failed_call_ids=failed_ids,
            skipped_call_ids=skipped_ids,
            ok_scores=ok_scores,
            gt_pairs=gt_pairs,
            model=model,
        )

    mean_overall = round(sum(ok_scores) / len(ok_scores), 2) if ok_scores else None
    mae_vs_gt = (
        round(sum(abs(p - g) for p, g in gt_pairs) / len(gt_pairs), 2) if gt_pairs else None
    )

    return {
        "suite_run_id": suite_run_id,
        "ok_scores": ok_scores,
        "failed_ids": failed_ids,
        "failed_msgs": failed_msgs,
        "skipped_ids": skipped_ids,
        "gt_pairs": gt_pairs,
        "mean_overall": mean_overall,
        "mae_vs_gt": mae_vs_gt,
    }


if __name__ == "__main__":
    include_generated = "--seed" not in sys.argv

    def _p(done, total, msg):
        print(f"[{done}/{total}] {msg}")

    ok, fail = run(include_generated=include_generated, progress=_p)
    print(f"\nDone. {ok} evaluated, {fail} failed. Stored {storage.count()} total.")
