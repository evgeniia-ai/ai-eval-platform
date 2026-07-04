"""Self-improvement loop: per-rep coaching derived from stored evaluations.

Aggregates a rep's stored evaluations, identifies recurring weaknesses (lowest
average dimensions), and asks Claude to turn that evidence into a personalized,
actionable coaching summary.

`coaching_directive_for_prompt` turns a coaching summary into a system-prompt
addendum fed back into future call handling for that rep — closing the loop.
"""

from __future__ import annotations

import json

import anthropic

from . import config, storage
from .models import CoachingSummary
from .qa_engine import QAError
from .rubric import RUBRIC, WEIGHTS

_NAME = {d.key: d.name for d in RUBRIC}


def recurring_weaknesses(
    rep_id: str, threshold: float = 3.5, avgs: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    """Dimensions where the rep's average falls below `threshold`, weakest first.

    Pass `avgs` to score against already-fetched averages (e.g. from DEMO_MODE's
    read-only data layer) instead of hitting storage directly.
    """
    if avgs is None:
        avgs = storage.rep_dimension_averages(rep_id)
    weak = [(k, v) for k, v in avgs.items() if v < threshold]
    return sorted(weak, key=lambda kv: kv[1])


def _build_prompt(rep_id: str) -> str:
    rows = storage.evaluations_for_rep(rep_id)
    avgs = storage.rep_dimension_averages(rep_id)
    weak = recurring_weaknesses(rep_id)

    # Compact evidence digest: per-call scores + the improvements the engine flagged.
    digest = []
    for r in rows:
        detail = json.loads(r["detail_json"])
        digest.append({
            "call_id": r["call_id"],
            "call_type": r["call_type"],
            "overall": r["overall_score"],
            "scores": {k: r[k] for k in avgs},
            "improvements": detail.get("top_improvements", []),
        })

    avg_lines = "\n".join(
        f"- {_NAME[k]} (`{k}`): avg {v}  (rubric weight {int(WEIGHTS[k] * 100)}%)"
        for k, v in avgs.items()
    )
    weak_lines = ", ".join(f"{_NAME[k]} ({v})" for k, v in weak) or "none below threshold"

    return f"""Generate a personalized coaching summary for service representative {rep_id},
based on {len(rows)} evaluated calls.

AVERAGE SCORES BY DIMENSION (1-5)
{avg_lines}

RECURRING WEAKNESSES (avg below 3.5)
{weak_lines}

PER-CALL EVIDENCE (scores + flagged improvements)
{json.dumps(digest, indent=2)}

Write coaching that is specific, evidence-based, and encouraging. Prioritize the
recurring weaknesses above. Reference patterns across calls rather than one-offs.
Return the required structured format."""


def generate_coaching(rep_id: str) -> CoachingSummary:
    rows = storage.evaluations_for_rep(rep_id)
    if not rows:
        raise QAError(f"No evaluations stored for {rep_id}.")
    try:
        response = config.client().messages.parse(
            model=config.model(),
            max_tokens=1500,
            system=(
                "You are a supportive but candid contact-center coach. You write coaching "
                "grounded strictly in the QA evidence provided, focused on the highest-leverage "
                "behavior changes."
            ),
            messages=[{"role": "user", "content": _build_prompt(rep_id)}],
            output_format=CoachingSummary,
        )
    except anthropic.APIStatusError as e:
        raise QAError(f"API error generating coaching for {rep_id}: {e.status_code} {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise QAError(f"Network error generating coaching for {rep_id}: {e}") from e

    summary = response.parsed_output
    if summary is None:
        raise QAError(f"No coaching summary returned for {rep_id}.")
    return summary


def coaching_directive_for_prompt(summary: CoachingSummary) -> str:
    """Bonus feedback loop: convert coaching into a system-prompt addendum.

    Injecting this into the system prompt of a live call-assist agent steers future
    handling toward the rep's focus areas — coaching insights flow back into the work.
    """
    focus = "\n".join(f"- {f}" for f in summary.focus_areas)
    actions = "\n".join(f"- {a}" for a in summary.suggested_actions)
    return (
        "COACHING CONTEXT FOR THIS REP (incorporate into how you assist them):\n"
        f"Focus areas to reinforce:\n{focus}\n"
        f"Behaviors to prompt proactively:\n{actions}"
    )
