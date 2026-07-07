"""Import and validate externally-generated (e.g. GPT-written) call transcripts.

Reads every *.json file in data/gpt_raw/ (one call object, or an array of call
objects, per file — ```json code fences are tolerated and stripped), validates
each against the Transcript model plus a few extra rules (see validate_call),
and writes the accepted calls as a single sorted-by-call_id data/gpt_transcripts.json.

These are unlabeled candidate transcripts: ground_truth_qa must be absent —
human reviewers add labels later, so a call carrying one is rejected as a
sign the source (or a prior run) already baked in a label.

Usage:
    python scripts/import_gpt_transcripts.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Optional

# Make sure the repo root is on sys.path when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from src import config
from src.models import Transcript

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "gpt_raw"
OUT_PATH = REPO_ROOT / "data" / "gpt_transcripts.json"

CALL_ID_RE = re.compile(r"^HB-GPT-\d{4}$")
MIN_UTTERANCES = 12

Reject = tuple[str, str, str]  # (file, call_id_or_'?', reason)


def strip_code_fences(text: str) -> str:
    """Tolerate a lone ```json ... ``` (or bare ```) fence wrapping the JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _timestamp_to_seconds(ts: str) -> Optional[int]:
    parts = ts.split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


def load_existing_call_ids() -> set[str]:
    """call_ids already used by the seed and (if present) generated data files."""
    ids: set[str] = set()
    for path in (config.SEED_PATH, config.GENERATED_PATH):
        if not pathlib.Path(path).exists():
            continue
        with open(path, encoding="utf-8") as f:
            for raw in json.load(f):
                ids.add(raw["call_id"])
    return ids


def validate_call(raw: object, known_ids: set[str]) -> tuple[Optional[Transcript], Optional[str]]:
    """Validate one candidate call. Returns (transcript, None) if accepted, or
    (None, reason) if rejected. On acceptance, `known_ids` gains the call_id —
    pass the same set across calls in a run so duplicates are caught.
    """
    try:
        t = Transcript.model_validate(raw)
    except ValidationError as e:
        return None, f"schema validation failed: {e.errors()[0]['msg']}"

    if not CALL_ID_RE.match(t.call_id):
        return None, f"call_id '{t.call_id}' does not match HB-GPT-\\d{{4}}"

    if t.call_id in known_ids:
        return None, f"duplicate call_id '{t.call_id}'"

    if t.ground_truth_qa is not None:
        return None, "ground_truth_qa must not be present (labels are added by human review later)"

    if len(t.transcript) < MIN_UTTERANCES:
        return None, f"only {len(t.transcript)} utterances (need >= {MIN_UTTERANCES})"

    seconds: list[int] = []
    for u in t.transcript:
        secs = _timestamp_to_seconds(u.timestamp)
        if secs is None:
            return None, f"unparseable timestamp '{u.timestamp}'"
        seconds.append(secs)
    for prev, cur in zip(seconds, seconds[1:]):
        if cur <= prev:
            return None, f"timestamps not monotonically increasing (found {prev}s -> {cur}s)"

    known_ids.add(t.call_id)
    return t, None


def _call_id_guess(raw: object) -> str:
    if isinstance(raw, dict):
        return str(raw.get("call_id", "?"))
    return "?"


def import_all() -> tuple[list[Transcript], list[Reject]]:
    known_ids = load_existing_call_ids()
    accepted: list[Transcript] = []
    rejected: list[Reject] = []

    for file_path in sorted(RAW_DIR.glob("*.json")):
        try:
            text = strip_code_fences(file_path.read_text(encoding="utf-8"))
            data = json.loads(text)
        except json.JSONDecodeError as e:
            rejected.append((file_path.name, "?", f"invalid JSON: {e}"))
            continue

        calls = data if isinstance(data, list) else [data]
        for raw in calls:
            t, reason = validate_call(raw, known_ids)
            if reason is not None:
                rejected.append((file_path.name, _call_id_guess(raw), reason))
            else:
                assert t is not None
                accepted.append(t)

    accepted.sort(key=lambda t: t.call_id)
    return accepted, rejected


def main() -> None:
    if not RAW_DIR.exists():
        print(f"No {RAW_DIR} directory — nothing to import.")
        return

    accepted, rejected = import_all()

    OUT_PATH.write_text(
        json.dumps([t.model_dump(mode="json") for t in accepted], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Accepted: {len(accepted)} -> {OUT_PATH}")
    print(f"Rejected: {len(rejected)}")
    for file_name, call_id, reason in rejected:
        print(f"  [{file_name}] {call_id}: {reason}")


if __name__ == "__main__":
    main()
