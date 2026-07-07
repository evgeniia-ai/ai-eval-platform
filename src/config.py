

"""Central configuration: Anthropic client and model selection.

A single place to construct the client and resolve which model to use, so every
component (QA engine, coaching, data generation) shares the same setup.
"""

import os
from functools import lru_cache

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Default to the most capable current model. The retired `claude-sonnet-4-20250514`
# (the old repo default) now 404s, so we never fall back to it. Override per-deploy
# with ANTHROPIC_MODEL if you want a cheaper/faster model for high-volume runs.
DEFAULT_MODEL = "claude-opus-4-8"

# A cheaper, fast model for bulk synthetic-data generation where top-tier
# reasoning is less critical. Override with DATAGEN_MODEL.
DEFAULT_DATAGEN_MODEL = "claude-sonnet-4-6"

# Path to the on-disk SQLite store. Overridable for tests.
DB_PATH = os.environ.get("QA_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "qa.db"))

SEED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "seed_transcripts.json")
GENERATED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "generated_transcripts.json")
GPT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gpt_transcripts.json")
HOLDOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "holdout_ids.json")
SEED_LABELS_BACKUP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "seed_labels_v0_backup.json"
)


def model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def datagen_model() -> str:
    return os.environ.get("DATAGEN_MODEL", DEFAULT_DATAGEN_MODEL)


@lru_cache(maxsize=1)
def client() -> Anthropic:
    """Shared Anthropic client. Reads ANTHROPIC_API_KEY from the environment."""
    return Anthropic()
