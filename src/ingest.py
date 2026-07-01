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


if __name__ == "__main__":
    include_generated = "--seed" not in sys.argv

    def _p(done, total, msg):
        print(f"[{done}/{total}] {msg}")

    ok, fail = run(include_generated=include_generated, progress=_p)
    print(f"\nDone. {ok} evaluated, {fail} failed. Stored {storage.count()} total.")
