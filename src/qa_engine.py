"""LLM-as-judge QA evaluation engine.

Accepts a transcript, sends it to Claude with a rubric-based prompt, and returns a
fully structured evaluation: per-dimension scores + evidence + suggestions, a
code-computed weighted overall, and call-level summary/strengths/improvements.

Uses `client.messages.parse` with a Pydantic schema (structured outputs), so the
response is validated automatically — no brittle regex JSON extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from . import config
from .models import Transcript, TranscriptEvaluation
from .rubric import rubric_for_prompt, weighted_overall


class QAError(Exception):
    """Raised when an evaluation cannot be produced (bad input or API failure)."""


@dataclass
class Evaluation:
    """The complete evaluation result: model output plus the computed overall score."""

    call_id: str
    rep_id: str
    call_type: str
    overall_score: float
    detail: TranscriptEvaluation

    def dimension_scores(self) -> dict[str, int]:
        return self.detail.dimension_scores()


SYSTEM_PROMPT = (
    "You are a meticulous healthcare contact-center QA evaluator for HealthBridge. "
    "You assess patient-facing calls strictly against a defined rubric, grounding every "
    "score in evidence from the transcript. You are fair but rigorous: a behavior that "
    "was expected but absent (e.g. identity never verified) is scored low even if the "
    "rest of the call went well. Score conservatively and consistently."
)


def _build_prompt(t: Transcript) -> str:
    return f"""Evaluate the following call against the five-dimension quality rubric.

QUALITY RUBRIC
{rubric_for_prompt()}

SCORING GUIDANCE
- Each dimension is scored 1 (poor) to 5 (excellent).
- For `evidence`, quote the transcript directly. If the expected behavior is missing,
  say so explicitly (e.g. "Rep never asked for date of birth").
- Protocol adherence is call-type-specific. For this call (`{t.call_type}`), enforce the
  correct ordering: identity verification before any account action, and insurance
  verified before confirming an appointment where applicable.

CALL METADATA
- call_id: {t.call_id}
- call_type: {t.call_type}
- duration_seconds: {t.duration_seconds}

TRANSCRIPT
{t.render()}

Return your evaluation in the required structured format."""


def evaluate(transcript: Transcript | dict) -> Evaluation:
    """Evaluate a single transcript. Accepts a Transcript or a raw dict.

    Raises QAError on malformed input or API failure.
    """
    # --- Input validation (malformed input) ---
    if isinstance(transcript, dict):
        try:
            transcript = Transcript.model_validate(transcript)
        except Exception as e:  # pydantic ValidationError and friends
            raise QAError(f"Malformed transcript: {e}") from e
    if not transcript.transcript:
        raise QAError(f"Transcript {transcript.call_id} has no utterances.")

    # --- API call with structured output ---
    try:
        response = config.client().messages.parse(
            model=config.model(),
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_prompt(transcript)}],
            output_format=TranscriptEvaluation,
        )
    except anthropic.APIStatusError as e:
        raise QAError(f"API error evaluating {transcript.call_id}: {e.status_code} {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise QAError(f"Network error evaluating {transcript.call_id}: {e}") from e

    detail = response.parsed_output
    if detail is None:
        # Refusal or unparseable output.
        raise QAError(f"No structured evaluation returned for {transcript.call_id} "
                      f"(stop_reason={response.stop_reason}).")

    overall = weighted_overall(detail.dimension_scores())
    return Evaluation(
        call_id=transcript.call_id,
        rep_id=transcript.rep_id,
        call_type=transcript.call_type,
        overall_score=overall,
        detail=detail,
    )
