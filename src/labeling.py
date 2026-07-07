"""Human ground-truth labeling for GPT-generated calls (data/gpt_transcripts.json)
and re-labeling of the seed calls (data/seed_transcripts.json).

Pure read/write/validate logic, no Streamlit dependency — the "Labeling" page
in src/app.py is a thin UI over these functions. Every read/write function
defaults to the GPT path; pass `path=config.SEED_PATH` (or use the
save_seed_label/load_seed_backup helpers) to operate on seed calls instead.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Optional

from . import config
from .rubric import DIMENSION_KEYS, RUBRIC, weighted_overall

_NAME = {d.key: d.name for d in RUBRIC}
MIN_OBSERVED_LEN = 20
_NOTES_SECTION_RE = re.compile(r"^(Observed|Concern|Impact):\s?(.*)$")


def load_calls(path: Optional[str] = None) -> list[dict]:
    """Raw call dicts from `path` (default config.GPT_PATH), or [] if it doesn't exist yet."""
    path = path or config.GPT_PATH
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_labeled(call: dict) -> bool:
    return call.get("ground_truth_qa") is not None


def labeled_count(calls: list[dict]) -> tuple[int, int]:
    """(labeled, total)."""
    return sum(1 for c in calls if is_labeled(c)), len(calls)


def validate_label(dimension_scores: dict[str, Optional[int]], observed: str) -> Optional[str]:
    """None if the label is complete and savable, else a human-readable error.

    Only 'Observed' has a minimum length — Concern and Impact are optional
    (a perfect call legitimately has no concern to raise).
    """
    missing = [k for k in DIMENSION_KEYS if dimension_scores.get(k) is None]
    if missing:
        names = ", ".join(_NAME.get(k, k) for k in missing)
        return f"Missing score(s) for: {names}"
    if len(observed.strip()) < MIN_OBSERVED_LEN:
        return (
            f"'Observed' must be at least {MIN_OBSERVED_LEN} characters "
            f"(currently {len(observed.strip())})."
        )
    return None


def build_reviewer_notes(observed: str, concern: str = "", impact: str = "") -> str:
    """Combine the Observed/Concern/Impact fields into the single reviewer_notes
    string stored on disk — same plain-string schema as seed_transcripts.json.
    Empty optional sections are omitted.
    """
    lines = [f"Observed: {observed.strip()}"]
    if concern.strip():
        lines.append(f"Concern: {concern.strip()}")
    if impact.strip():
        lines.append(f"Impact: {impact.strip()}")
    return "\n".join(lines)


def parse_reviewer_notes(notes: str) -> tuple[str, str, str]:
    """Best-effort inverse of build_reviewer_notes(): split a stored reviewer_notes
    string back into (observed, concern, impact) for pre-filling the edit form.

    Falls back to putting the entire string into `observed` if it doesn't start
    with a recognized 'Observed:' line (e.g. free-form seed-style notes).
    """
    lines = notes.splitlines()
    if not lines or not _NOTES_SECTION_RE.match(lines[0]):
        return notes, "", ""

    sections = {"Observed": "", "Concern": "", "Impact": ""}
    current: Optional[str] = None
    for line in lines:
        m = _NOTES_SECTION_RE.match(line)
        if m:
            current = m.group(1)
            sections[current] = m.group(2)
        elif current:
            sections[current] += "\n" + line
    return sections["Observed"], sections["Concern"], sections["Impact"]


def next_unlabeled_call_id(
    call_ids: list[str], calls_by_id: dict[str, dict], current_id: str
) -> Optional[str]:
    """The next unlabeled call_id after `current_id`, wrapping around `call_ids`
    in order. None if every call is labeled. Used to auto-advance after a save.
    """
    start = call_ids.index(current_id) + 1 if current_id in call_ids else 0
    ordered = call_ids[start:] + call_ids[:start]
    for cid in ordered:
        if not is_labeled(calls_by_id[cid]):
            return cid
    return None


def next_call_id_in_order(call_ids: list[str], current_id: str) -> Optional[str]:
    """The call_id immediately after `current_id` in `call_ids` — no wrapping.

    Fallback for next_unlabeled_call_id when every call is already labeled
    (always true on the seed source, and eventually true on the GPT source),
    so re-labeling can still advance sequentially. None if `current_id` is
    last (or not found) — the caller should treat that as "done".
    """
    if current_id not in call_ids:
        return None
    idx = call_ids.index(current_id)
    if idx + 1 >= len(call_ids):
        return None
    return call_ids[idx + 1]


def has_full_ocim_notes(call: dict) -> bool:
    """True if `call`'s reviewer_notes has all three sections (Observed,
    Concern, Impact) non-empty — used to pick a rich showcase example."""
    gt = call.get("ground_truth_qa")
    if not gt:
        return False
    observed, concern, impact = parse_reviewer_notes(gt.get("reviewer_notes", ""))
    return bool(observed.strip()) and bool(concern.strip()) and bool(impact.strip())


def find_call_with_rich_notes(calls: list[dict]) -> Optional[str]:
    """The first call_id in `calls` with a full Observed/Concern/Impact note,
    or None if none qualify."""
    for call in calls:
        if has_full_ocim_notes(call):
            return call["call_id"]
    return None


def _atomic_write(data, path: str) -> None:
    """Write `data` (JSON-serializable) to `path` via temp-file + os.replace."""
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".labeling_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_label(
    call_id: str,
    dimension_scores: dict[str, int],
    observed: str,
    concern: str = "",
    impact: str = "",
    path: Optional[str] = None,
) -> dict:
    """Validate and write ground_truth_qa for `call_id` in `path` (default
    config.GPT_PATH). Returns the saved block.

    Same shape as seed_transcripts.json's ground_truth_qa: overall_score (weighted,
    rounded to 1 decimal), reviewer_notes (Observed/Concern/Impact combined into one
    plain string via build_reviewer_notes), dimension_scores. Raises ValueError if
    the label is incomplete or call_id isn't found.
    """
    path = path or config.GPT_PATH
    error = validate_label(dimension_scores, observed)
    if error:
        raise ValueError(error)

    calls = load_calls(path)
    ground_truth_qa = {
        "overall_score": round(weighted_overall(dimension_scores), 1),
        "reviewer_notes": build_reviewer_notes(observed, concern, impact),
        "dimension_scores": dict(dimension_scores),
    }

    found = False
    for call in calls:
        if call["call_id"] == call_id:
            call["ground_truth_qa"] = ground_truth_qa
            found = True
            break
    if not found:
        raise ValueError(f"call_id '{call_id}' not found in {path}")

    _atomic_write(calls, path)
    return ground_truth_qa


def load_seed_backup() -> dict[str, dict]:
    """The seed re-labeling backup: {call_id: original ground_truth_qa}, or {}."""
    if not os.path.exists(config.SEED_LABELS_BACKUP_PATH):
        return {}
    with open(config.SEED_LABELS_BACKUP_PATH, encoding="utf-8") as f:
        return json.load(f)


def backup_original_seed_label(call_id: str, ground_truth_qa: dict) -> None:
    """Snapshot `ground_truth_qa` under `call_id` in the seed backup file, but
    only if that call_id isn't already backed up. The backup must capture the
    ORIGINAL (pre-relabel) label exactly once — later re-labels never touch it.
    """
    backups = load_seed_backup()
    if call_id in backups:
        return
    backups[call_id] = ground_truth_qa
    _atomic_write(backups, config.SEED_LABELS_BACKUP_PATH)


def save_seed_label(
    call_id: str,
    dimension_scores: dict[str, int],
    observed: str,
    concern: str = "",
    impact: str = "",
) -> dict:
    """Like save_label, but targets data/seed_transcripts.json and snapshots the
    call's pre-relabel ground_truth_qa into the seed backup before the first
    overwrite (see backup_original_seed_label — a no-op on later re-labels).
    """
    calls = load_calls(config.SEED_PATH)
    call = next((c for c in calls if c["call_id"] == call_id), None)
    if call is None:
        raise ValueError(f"call_id '{call_id}' not found in {config.SEED_PATH}")
    if call.get("ground_truth_qa") is not None:
        backup_original_seed_label(call_id, call["ground_truth_qa"])

    return save_label(call_id, dimension_scores, observed, concern, impact, path=config.SEED_PATH)
