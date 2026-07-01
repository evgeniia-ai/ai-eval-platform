"""The quality rubric — the single source of truth for dimensions and weights.

Keys match the seed data's `ground_truth_qa.dimension_scores` exactly, so model
output can be compared directly against human ground truth.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    key: str
    name: str
    description: str
    weight: float


RUBRIC: list[Dimension] = [
    Dimension(
        key="greeting_identity_verification",
        name="Greeting & Identity Verification",
        description=(
            "Rep introduced themselves, verified patient identity (full name AND date of "
            "birth — both are required) before taking any account action, and stated the "
            "purpose of the call. Failure to collect date of birth, regardless of how well "
            "the rest of the call went, must result in a score of 1 or 2. Partial "
            "verification (name only, or DOB collected after account action) scores no "
            "higher than 2."
        ),
        weight=0.12,
    ),
    Dimension(
        key="empathy_tone",
        name="Empathy & Tone",
        description=(
            "Acknowledged the patient's concerns, used empathetic language, and "
            "maintained a professional, warm tone throughout the call."
        ),
        weight=0.25,
    ),
    Dimension(
        key="accuracy_completeness",
        name="Accuracy & Completeness",
        description=(
            "Did the rep gather the information needed before acting — verifying the "
            "patient and asking the right clarifying questions — and either fully "
            "resolve the request or correctly escalate it to a human when it falls "
            "outside what the agent should handle alone (e.g. a billing dispute or a "
            "suspected coding error)? Score down for acting on unverified or outdated "
            "info, deflecting a question the patient actually asked, or failing to "
            "resolve or escalate. Judge only what information was gathered and whether "
            "the task was completed or properly handed off — do NOT score tone, "
            "politeness, or empathy here. "
            "When an error occurs and is then corrected, distinguish who caught it: a "
            "rep who proactively verifies the currency of data and catches their own "
            "mistake demonstrates strong completeness (full credit); a correction made "
            "only after the patient objects is acceptable but not exemplary — it should "
            "score about one point lower — especially when the rep could have confirmed "
            "the data's currency before acting. Do not penalize the rep for information "
            "the patient never provided that could not reasonably have been on file."
        ),
        weight=0.25,
    ),
    Dimension(
        key="protocol_adherence",
        name="Protocol Adherence",
        description=(
            "Followed the appropriate workflow for this call type: identity before "
            "any account action; insurance verified before confirming an appointment; "
            "clinical concerns triaged appropriately; billing disputes documented."
        ),
        weight=0.25,
    ),
    Dimension(
        key="closing_next_steps",
        name="Closing & Next Steps",
        description=(
            "Summarized the actions taken, confirmed concrete next steps (dates, "
            "amounts, follow-ups), and closed the call professionally."
        ),
        weight=0.13,
    ),
]

DIMENSION_KEYS = [d.key for d in RUBRIC]
WEIGHTS = {d.key: d.weight for d in RUBRIC}

# Bump this manually whenever dimension definitions or weights change, so every
# stored evaluation row is traceable to the rubric that produced it.
# 1.0 → 1.1: accuracy_completeness wording tightened (escalation standard);
#             weights changed from 15/25/25/25/15 to 12/25/25/25/13.
RUBRIC_VERSION = "1.1"


def weighted_overall(scores: dict[str, int]) -> float:
    """Weighted average of the per-dimension 1-5 scores, rounded to 2 decimals.

    Note: the brief's stated weights sum to 105% (15+25+25+25+15), which would
    push a perfect call above 5.0. We normalize by the weight total so the result
    stays on a true 1-5 scale while preserving the brief's relative weighting.
    """
    weight_total = sum(d.weight for d in RUBRIC)
    total = sum(scores[d.key] * d.weight for d in RUBRIC)
    return round(total / weight_total, 2)


def rubric_for_prompt() -> str:
    """Human-readable rubric block to embed in the evaluation prompt."""
    lines = []
    for i, d in enumerate(RUBRIC, 1):
        lines.append(f"{i}. {d.name} (`{d.key}`, weight {int(d.weight * 100)}%)\n   {d.description}")
    return "\n".join(lines)
