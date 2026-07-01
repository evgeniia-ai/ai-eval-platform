"""Evaluate a small set of call_ids against the real judge and print a comparison table.

Usage:
    python scripts/debug_judge.py HB-2026-00147 HB-2026-00148 HB-2026-00203

Prints for each call: call_id, judge accuracy_completeness score,
judge reasoning text, and the human ground-truth score (if present in the transcript).

Intended for cheap spot-checks after rubric changes (~10c per call vs 61c for the
full regression suite).
"""

import argparse
import json
import pathlib
import sys

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models import Transcript
from src.qa_engine import QAError, evaluate

SEED_PATH = pathlib.Path("data/seed_transcripts.json")
GENERATED_PATH = pathlib.Path("data/generated_transcripts.json")


def _load_all_transcripts() -> dict[str, Transcript]:
    transcripts: dict[str, Transcript] = {}
    for path in (SEED_PATH, GENERATED_PATH):
        if not path.exists():
            continue
        for raw in json.loads(path.read_text(encoding="utf-8")):
            t = Transcript.model_validate(raw)
            transcripts[t.call_id] = t
    return transcripts


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-check accuracy_completeness scoring for specific calls.")
    parser.add_argument("call_ids", nargs="+", metavar="CALL_ID")
    args = parser.parse_args()

    all_transcripts = _load_all_transcripts()

    for call_id in args.call_ids:
        print(f"\n{'=' * 70}")
        print(f"CALL: {call_id}")

        transcript = all_transcripts.get(call_id)
        if transcript is None:
            print(f"  ERROR: call_id not found in seed or generated transcripts.")
            continue

        gt_score = None
        if transcript.ground_truth_qa and "accuracy_completeness" in transcript.ground_truth_qa.dimension_scores:
            gt_score = transcript.ground_truth_qa.dimension_scores["accuracy_completeness"]

        try:
            ev = evaluate(transcript)
        except QAError as e:
            print(f"  ERROR: {e}")
            continue

        dim = ev.detail.accuracy_completeness
        print(f"  Judge score  : {dim.score}/5")
        print(f"  Human GT     : {gt_score}/5" if gt_score is not None else "  Human GT     : (none)")
        print(f"  Reasoning    : {dim.reasoning}")
        print(f"  Evidence     : {dim.evidence}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
