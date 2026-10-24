from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from subtranslate.config import Settings
from subtranslate.debug import translate_batch_by_id, translate_batch_by_range
from subtranslate.models import TranslationResult, Usage


def _settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=5,
    )


def _fake_result(client, settings, batch) -> TranslationResult:
    translations = {
        entry.id: f"[PT] {entry.text}"
        for entry in batch.entries
        if entry.id in batch.target_ids
    }
    return TranslationResult(
        translations=translations,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_translate_batch_by_id_returns_correct_batch(hope_like_batch: Path) -> None:
    settings = _settings()
    with patch("subtranslate.debug.translate_batch", _fake_result):
        batch, result = translate_batch_by_id(hope_like_batch, settings, 1361)

    assert batch.target_ids == {1361, 1362, 1363, 1364, 1365}
    assert result.translations[1361] == "[PT] Line thirteen sixty-one"


def test_translate_batch_by_id_rejects_unknown_id(hope_like_batch: Path) -> None:
    settings = _settings()
    with (
        patch("subtranslate.debug.translate_batch", _fake_result),
        pytest.raises(ValueError, match="not found"),
    ):
        translate_batch_by_id(hope_like_batch, settings, 999999)


def test_translate_batch_by_range_returns_contained_batch(hope_like_batch: Path) -> None:
    settings = _settings()
    with patch("subtranslate.debug.translate_batch", _fake_result):
        batch, result = translate_batch_by_range(hope_like_batch, settings, 1360, 1365)

    assert batch.target_ids == {1361, 1362, 1363, 1364, 1365}
    assert 1361 in result.translations
