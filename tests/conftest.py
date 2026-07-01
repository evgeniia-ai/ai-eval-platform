import pytest
import src.config as _config


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect all storage writes to a throwaway SQLite file."""
    monkeypatch.setattr(_config, "DB_PATH", str(tmp_path / "test.db"))
