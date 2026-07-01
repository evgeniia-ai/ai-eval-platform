"""Layer 2: regression suite — calls the real Anthropic API.

Skipped by default (see pytest.ini addopts). Opt in with:
    pytest -m regression

Also skipped automatically if ANTHROPIC_API_KEY is not set.

Evaluates all 7 seed transcripts (which carry human ground-truth QA scores)
and asserts that the judge's MAE stays within acceptable thresholds.
"""

import os

import pytest

from src import ingest
from src.qa_engine import evaluate
from src.rubric import DIMENSION_KEYS

pytestmark = pytest.mark.regression

_OVERALL_MAE_THRESHOLD = 0.75   # on the 1–5 scale; tighten once baseline is known
_DIM_MAE_THRESHOLD = 1.0        # per-dimension; catches right-total-wrong-reason cases


@pytest.fixture(scope="module", autouse=True)
def require_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — skipping regression suite")


@pytest.fixture(scope="module")
def evaluated_seed_calls(require_api_key):
    """Run the real judge on all ground-truth seed transcripts; cache for the module."""
    transcripts = [
        t for t in ingest.load_transcripts(include_generated=False)
        if t.ground_truth_qa
    ]
    assert transcripts, "No seed transcripts with ground_truth_qa found"
    return [(t, evaluate(t)) for t in transcripts]


def test_judge_overall_mae(evaluated_seed_calls):
    rows = [
        (t.call_id, ev.overall_score, t.ground_truth_qa.overall_score,
         abs(ev.overall_score - t.ground_truth_qa.overall_score))
        for t, ev in evaluated_seed_calls
    ]
    mae = sum(r[3] for r in rows) / len(rows)
    detail = "\n".join(
        f"  {cid}: predicted={pred:.2f}  ground_truth={gt:.2f}  |err|={err:.2f}"
        for cid, pred, gt, err in rows
    )
    assert mae <= _OVERALL_MAE_THRESHOLD, (
        f"Overall MAE {mae:.3f} exceeds threshold {_OVERALL_MAE_THRESHOLD}\n{detail}"
    )


@pytest.mark.parametrize("dimension", DIMENSION_KEYS)
def test_judge_dimension_mae(dimension, evaluated_seed_calls):
    errors = [
        abs(
            ev.dimension_scores()[dimension]
            - t.ground_truth_qa.dimension_scores[dimension]
        )
        for t, ev in evaluated_seed_calls
    ]
    mae = sum(errors) / len(errors)
    assert mae <= _DIM_MAE_THRESHOLD, (
        f"Dimension '{dimension}' MAE {mae:.3f} exceeds threshold {_DIM_MAE_THRESHOLD}"
    )
