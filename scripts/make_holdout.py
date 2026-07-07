"""Select a held-out evaluation split from the labeled GPT-generated calls.

~20% of labeled calls (data/gpt_transcripts.json), stratified by call_type
using largest-remainder apportionment so the total lands close to the target
fraction, with a fixed random seed so the split is reproducible. Writes the
selected call_ids to data/holdout_ids.json.

These calls must never appear in a suite/eval set used for model calibration
(MAE tracking, model comparison) — src/test_sets.py's FULL_SET already
excludes them automatically.

Usage:
    python scripts/make_holdout.py
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
from collections import defaultdict

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import labeling

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
OUT_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "holdout_ids.json"


def compute_holdout_ids(
    calls: list[dict], fraction: float = HOLDOUT_FRACTION, seed: int = HOLDOUT_SEED
) -> list[str]:
    """~`fraction` of labeled calls in `calls`, stratified by call_type.

    Per-type quotas are apportioned by largest remainder so the total is
    round(n_labeled * fraction) rather than drifting from independent
    per-group rounding. Deterministic for a given `calls` + `seed`.
    """
    by_type: dict[str, list[str]] = defaultdict(list)
    for c in calls:
        if labeling.is_labeled(c):
            by_type[c["call_type"]].append(c["call_id"])

    total = sum(len(ids) for ids in by_type.values())
    target_total = round(total * fraction)

    quotas = {ct: len(ids) * fraction for ct, ids in by_type.items()}
    base = {ct: int(quotas[ct]) for ct in quotas}  # floor
    remainders = {ct: quotas[ct] - base[ct] for ct in quotas}

    remaining = target_total - sum(base.values())
    # Largest remainder first; alphabetical call_type as a deterministic tie-break.
    order = sorted(remainders, key=lambda ct: (-remainders[ct], ct))
    for ct in order[:remaining]:
        base[ct] += 1

    rng = random.Random(seed)
    holdout: list[str] = []
    for ct in sorted(by_type):  # sorted call_type order, independent of dict/insertion order
        ids = sorted(by_type[ct])  # sorted before shuffling, for determinism
        rng.shuffle(ids)
        holdout.extend(ids[: base[ct]])
    return sorted(holdout)


def main() -> None:
    calls = labeling.load_calls()
    holdout_ids = compute_holdout_ids(calls)

    OUT_PATH.write_text(json.dumps(holdout_ids, indent=2), encoding="utf-8")
    print(f"Wrote {len(holdout_ids)} holdout call_ids -> {OUT_PATH}")
    for cid in holdout_ids:
        print(f"  {cid}")


if __name__ == "__main__":
    main()
