"""Pydantic schemas for transcripts, evaluations, and coaching.

These drive the API's structured-output mode (`client.messages.parse`), so Claude
returns validated objects directly — no regex/JSON-extraction guesswork.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Structured outputs support `enum`, so a Literal of the valid scores both
# constrains the model and validates client-side.
Score = Literal[1, 2, 3, 4, 5]

CallType = Literal[
    "appointment_scheduling",
    "insurance_verification",
    "billing_inquiry",
    "prescription_refill",
    "clinical_triage",
]


# ---------------------------------------------------------------------------
# Transcript input
# ---------------------------------------------------------------------------

class Utterance(BaseModel):
    timestamp: str
    speaker: Literal["rep", "patient"]
    text: str


class GroundTruthQA(BaseModel):
    overall_score: float
    reviewer_notes: str
    dimension_scores: dict[str, int]


class Transcript(BaseModel):
    call_id: str
    call_type: CallType
    duration_seconds: int
    rep_id: str
    patient_satisfaction_score: Optional[float] = None
    ground_truth_qa: Optional[GroundTruthQA] = None
    transcript: list[Utterance]

    def render(self) -> str:
        """Flatten the utterances into a readable transcript for the prompt."""
        return "\n".join(f"[{u.timestamp}] {u.speaker.upper()}: {u.text}" for u in self.transcript)


# ---------------------------------------------------------------------------
# Evaluation output (what Claude produces)
# ---------------------------------------------------------------------------

class DimensionEvaluation(BaseModel):
    score: Score = Field(description="Integer 1 (poor) to 5 (excellent) for this dimension.")
    evidence: str = Field(
        description="A direct quote from the transcript that justifies the score, or "
        "an explicit note that the expected behavior was absent."
    )
    reasoning: str = Field(description="One or two sentences explaining the score.")
    suggestion: str = Field(description="A concrete, actionable improvement for the rep.")


class TranscriptEvaluation(BaseModel):
    """The model-produced evaluation. Overall score is computed in code, not here."""

    greeting_identity_verification: DimensionEvaluation
    empathy_tone: DimensionEvaluation
    accuracy_completeness: DimensionEvaluation
    protocol_adherence: DimensionEvaluation
    closing_next_steps: DimensionEvaluation
    summary: str = Field(description="2-3 sentence overall assessment of the call.")
    top_strengths: list[str] = Field(description="1-3 things the rep did well.")
    top_improvements: list[str] = Field(description="1-3 highest-priority improvements.")

    def dimension_scores(self) -> dict[str, int]:
        return {
            "greeting_identity_verification": self.greeting_identity_verification.score,
            "empathy_tone": self.empathy_tone.score,
            "accuracy_completeness": self.accuracy_completeness.score,
            "protocol_adherence": self.protocol_adherence.score,
            "closing_next_steps": self.closing_next_steps.score,
        }


# ---------------------------------------------------------------------------
# Coaching output
# ---------------------------------------------------------------------------

class CoachingSummary(BaseModel):
    headline: str = Field(description="One-line summary of the rep's current standing.")
    strengths: list[str] = Field(description="Consistent strengths across the rep's calls.")
    focus_areas: list[str] = Field(
        description="The 2-3 recurring weaknesses to prioritize, most important first."
    )
    coaching_plan: str = Field(
        description="A short, specific, encouraging coaching narrative the rep can act on."
    )
    suggested_actions: list[str] = Field(
        description="Concrete next steps (e.g. scripts to practice, behaviors to adopt)."
    )
