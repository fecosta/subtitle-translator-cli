from __future__ import annotations

import os
from pathlib import Path

import pytest

from subtranslate.config import Settings


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_srt(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.en.srt"


@pytest.fixture
def hope_like_batch(fixtures_dir: Path) -> Path:
    return fixtures_dir / "hope_like_batch.srt"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=2,
    )


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


@pytest.fixture
def no_mounts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a host with no mount points anywhere in the tested paths."""
    monkeypatch.setattr(os.path, "ismount", lambda _path: False)


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Prevent real network calls if anything leaks through.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/invalid")


@pytest.fixture(autouse=True)
def _no_translator_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("subtranslate.translator.time.sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _isolate_checkpoints(monkeypatch: pytest.MonkeyPatch, checkpoint_dir: Path) -> None:
    monkeypatch.setattr("subtranslate.checkpoint.CHECKPOINT_BASE_DIR", checkpoint_dir)
