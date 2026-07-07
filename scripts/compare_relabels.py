"""Compare pre-guidelines vs post-guidelines seed labels.

Reads data/seed_labels_v0_backup.json (the original label for each seed call,
snapshotted once before its first re-label — see src.labeling.save_seed_label)
against the current data/seed_transcripts.json, and prints, per re-labeled
call: old vs new overall score and per-dimension deltas, plus summary stats
(mean absolute delta per dimension). This measures annotation drift between
pre- and post-guidelines labeling (docs/human-annotation-guidelines.md).

Usage:
    python scripts/compare_relabels.py
"""

from __future__ import annotations

import pathlib
import sys

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config, labeling
from src.rubric import DIMENSION_KEYS, RUBRIC

_NAME = {d.key: d.name for d in RUBRIC}


def compare() -> list[dict]:
    """One row per re-labeled seed call: old/new overall + per-dimension deltas.

    [] if no seed call has been re-labeled yet (backup is empty/missing).
    """
    backup = labeling.load_seed_backup()
    if not backup:
        return []

    current_calls = {c["call_id"]: c for c in labeling.load_calls(config.SEED_PATH)}

    rows = []
    for call_id, old_gt in backup.items():
        call = current_calls.get(call_id)
        new_gt = call.get("ground_truth_qa") if call else None
        if new_gt is None:
            continue

        dimension_deltas = {}
        for key in DIMENSION_KEYS:
            old_v = old_gt["dimension_scores"].get(key)
            new_v = new_gt["dimension_scores"].get(key)
            if old_v is not None and new_v is not None:
                dimension_deltas[key] = new_v - old_v

        rows.append({
            "call_id": call_id,
            "old_overall": old_gt["overall_score"],
            "new_overall": new_gt["overall_score"],
            "delta_overall": round(new_gt["overall_score"] - old_gt["overall_score"], 2),
            "dimension_deltas": dimension_deltas,
        })
    return sorted(rows, key=lambda r: r["call_id"])


def _print_table(rows: list[dict]) -> None:
    dim_headers = "".join(f"{_NAME[k][:10]:<12}" for k in DIMENSION_KEYS)
    header = f"{'call_id':<16}{'old':<6}{'new':<6}{'delta':<10}{dim_headers}"
    print(header)
    print("-" * len(header))
    for row in rows:
        dim_cells = "".join(
            f"{row['dimension_deltas'][k]:+d}".ljust(12) if k in row["dimension_deltas"] else f"{'n/a':<12}"
            for k in DIMENSION_KEYS
        )
        print(
            f"{row['call_id']:<16}{row['old_overall']:<6}{row['new_overall']:<6}"
            f"{row['delta_overall']:+.2f}    {dim_cells}"
        )


def _print_summary(rows: list[dict]) -> None:
    print("\n=== Mean absolute delta (annotation drift) ===")
    for key in DIMENSION_KEYS:
        deltas = [abs(row["dimension_deltas"][key]) for row in rows if key in row["dimension_deltas"]]
        if deltas:
            mad = round(sum(deltas) / len(deltas), 2)
            print(f"  {_NAME[key]}: {mad}  (n={len(deltas)})")
    overall_deltas = [abs(row["delta_overall"]) for row in rows]
    print(f"  Overall: {round(sum(overall_deltas) / len(overall_deltas), 2)}  (n={len(overall_deltas)})")


def main() -> None:
    rows = compare()
    if not rows:
        print(f"No re-labeled seed calls yet ({config.SEED_LABELS_BACKUP_PATH} is empty or missing).")
        return
    _print_table(rows)
    _print_summary(rows)


if __name__ == "__main__":
    main()
