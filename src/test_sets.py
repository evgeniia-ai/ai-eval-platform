"""Named evaluation sets for the Test runs page.

SMOKE_SET      — 5 calls, one per call type, for a fast sanity check.
REGRESSION_SET — 9 calls, covering high- and low-quality examples across
                 all call types, for a fuller rubric regression.
FULL_SET       — dynamically computed: every seed call plus every labeled
                 GPT-generated call (data/gpt_transcripts.json), minus the
                 held-out split (data/holdout_ids.json — see
                 scripts/make_holdout.py). Used for full calibration runs;
                 the held-out calls must never leak into a calibration set.
"""

from __future__ import annotations

import json
import os

from . import config, ingest, labeling

SMOKE_SET: frozenset[str] = frozenset({
    "HB-2026-00203",   # insurance_verification — low score, seed
    "HB-2026-00512",   # billing_inquiry — high score, seed
    "HB-SYNTH-00120",  # prescription_refill — mid score, synthetic
    "HB-2026-00401",   # clinical_triage — low score, only available
    "HB-2026-00315",   # prescription_refill — high score, seed
})

REGRESSION_SET: frozenset[str] = frozenset({
    "HB-2026-00147",   # appointment_scheduling — high, seed
    "HB-2026-00289",   # billing_inquiry — high, seed
    "HB-SYNTH-00118",  # billing_inquiry — high, synthetic
    "HB-SYNTH-00119",  # billing_inquiry — low, synthetic
    "HB-2026-00401",   # clinical_triage — only available
    "HB-2026-00203",   # insurance_verification — low, seed
    "HB-2026-00315",   # prescription_refill — high, seed
    "HB-2026-00587",   # prescription_refill — low, seed
    "HB-SYNTH-00120",  # prescription_refill — mid, synthetic
})


def _holdout_ids() -> frozenset[str]:
    if not os.path.exists(config.HOLDOUT_PATH):
        return frozenset()
    with open(config.HOLDOUT_PATH, encoding="utf-8") as f:
        return frozenset(json.load(f))


def full_set() -> frozenset[str]:
    """Seed calls + labeled GPT calls, excluding the held-out split.

    A function (not just the frozen FULL_SET constant below) so tests can
    call it fresh against monkeypatched data paths.
    """
    seed_ids = {t.call_id for t in ingest.load_transcripts(include_generated=False)}
    gpt_labeled_ids = {c["call_id"] for c in labeling.load_calls() if labeling.is_labeled(c)}
    return frozenset(seed_ids | gpt_labeled_ids) - _holdout_ids()


FULL_SET: frozenset[str] = full_set()
