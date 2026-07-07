import pytest
import src.config as _config


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect all storage writes to a throwaway SQLite file."""
    monkeypatch.setattr(_config, "DB_PATH", str(tmp_path / "test.db"))


@pytest.fixture
def tmp_gpt_path(monkeypatch, tmp_path):
    """Redirect data/gpt_transcripts.json reads/writes to a throwaway file."""
    monkeypatch.setattr(_config, "GPT_PATH", str(tmp_path / "gpt_transcripts.json"))


@pytest.fixture
def tmp_seed_path(monkeypatch, tmp_path):
    """Redirect data/seed_transcripts.json reads to a throwaway file."""
    monkeypatch.setattr(_config, "SEED_PATH", str(tmp_path / "seed_transcripts.json"))


@pytest.fixture
def tmp_holdout_path(monkeypatch, tmp_path):
    """Redirect data/holdout_ids.json reads to a throwaway file."""
    monkeypatch.setattr(_config, "HOLDOUT_PATH", str(tmp_path / "holdout_ids.json"))


@pytest.fixture
def tmp_seed_backup_path(monkeypatch, tmp_path):
    """Redirect data/seed_labels_v0_backup.json reads/writes to a throwaway file."""
    monkeypatch.setattr(_config, "SEED_LABELS_BACKUP_PATH", str(tmp_path / "seed_labels_v0_backup.json"))


@pytest.fixture
def tmp_demo_path(monkeypatch, tmp_path):
    """Redirect data/demo_results.json reads to a throwaway file, and clear
    demo_data's internal cache so each test sees a fresh read."""
    from src import demo_data as _demo_data

    monkeypatch.setattr(_demo_data, "DEMO_PATH", str(tmp_path / "demo_results.json"))
    _demo_data._load.cache_clear()
    yield
    _demo_data._load.cache_clear()
