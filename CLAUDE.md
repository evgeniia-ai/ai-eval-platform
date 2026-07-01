# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Verify environment is set up correctly
python verify_setup.py

# Run the Streamlit dashboard (main entry point)
streamlit run src/app.py

# Run the ingestion pipeline from the command line
python -m src.ingest                # evaluate seed + generated transcripts
python -m src.ingest --seed         # seed transcripts only

# Generate synthetic transcripts from CLI
python -m src.data_gen

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Model for QA evaluation and coaching |
| `DATAGEN_MODEL` | `claude-sonnet-4-6` | Model for synthetic data generation |
| `QA_DB_PATH` | `data/qa.db` | SQLite database path |

A `.env` file is loaded automatically via `python-dotenv` in `src/config.py`.

## Architecture

The system has three components wired together through a shared SQLite store:

**Component 1 — QA Engine (`src/qa_engine.py`):** Accepts a `Transcript`, builds a rubric-grounded prompt, and calls `client.messages.parse` with `output_format=TranscriptEvaluation`. The API returns a validated Pydantic object directly — no JSON extraction. The weighted overall score is computed in code (`src/rubric.py:weighted_overall`), not by the model, to keep scoring deterministic.

**Component 2 — Self-Improvement Loop (`src/coaching.py`):** Reads a rep's stored evaluations from SQLite, identifies dimensions below threshold 3.5, and generates a `CoachingSummary` via structured output. `coaching_directive_for_prompt` converts the summary into a system-prompt addendum that can be injected into a live call-assist agent, closing the feedback loop.

**Component 3 — Admin Dashboard (`src/app.py`):** Four-page Streamlit app. Pages call `storage.*` directly for data; the "Overview" page drives the full ingestion pipeline via `ingest.run`. Generated coaching is cached in `st.session_state` per rep.

**Data flow:**
```
data/seed_transcripts.json  ─┐
data/generated_transcripts.json ─┤  ingest.py → qa_engine.py → storage.py (SQLite)
                               └─              ↑                      ↓
                          data_gen.py ─────────┘              coaching.py / app.py
```

**Pydantic models (`src/models.py`):** `Transcript` is the input; `TranscriptEvaluation` and `CoachingSummary` are structured outputs Claude must conform to. `Score = Literal[1,2,3,4,5]` constrains scores at both the schema and model level.

**Rubric (`src/rubric.py`):** Single source of truth for the five QA dimensions and their weights. `DIMENSION_KEYS` is used throughout for column names, prompt rendering, and aggregation. The weights sum to 105% (per the original brief), so `weighted_overall` normalizes by the weight total to keep scores on a 1–5 scale.

**Config (`src/config.py`):** Constructs the shared `Anthropic` client (cached via `@lru_cache`) and resolves model names. All components import from here — never construct clients directly.

**Storage (`src/storage.py`):** The `evaluations` table stores both extracted columns (for fast aggregation) and full JSON blobs (`detail_json`, `transcript_json`) for drill-down. `UPSERT` on `call_id` so re-running the pipeline doesn't duplicate rows.

## Data Files

- `data/seed_transcripts.json` — 7 hand-crafted transcripts with human ground-truth QA scores
- `data/generated_transcripts.json` — synthetic transcripts written by `data_gen.py` (gitignored if not present)
- `data/qa.db` — SQLite database created at runtime by `storage.init()`
