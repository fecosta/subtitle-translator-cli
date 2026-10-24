from __future__ import annotations

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
def settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=2,
    )


@pytest.fixture(autouse=True)
def _ensure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # Prevent real network calls if anything leaks through.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/invalid")
