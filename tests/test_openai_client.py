from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from subtranslate.config import Settings
from subtranslate.models import Batch, SubtitleEntry
from subtranslate.openai_client import translate_batch


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


def _make_response(translations: list[dict], prompt_tokens: int = 10, completion_tokens: int = 5):
    content = json.dumps({"subtitles": translations})
    response = MagicMock()
    response.choices = [_FakeChoice(_FakeMessage(content))]
    response.usage = _FakeUsage(prompt_tokens, completion_tokens)
    return response


def _batch() -> Batch:
    entries = [
        SubtitleEntry(id=1, start=1000, end=2000, text="Hello"),
        SubtitleEntry(id=2, start=2000, end=3000, text="World"),
    ]
    return Batch(entries=entries, target_ids={1, 2})


def _settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=2,
    )


def test_translate_batch_returns_translations() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response(
        [{"id": 1, "text": "Olá"}, {"id": 2, "text": "mundo"}]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response

    result = translate_batch(client, settings, batch)
    assert result.translations == {1: "Olá", 2: "mundo"}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_translate_batch_rejects_missing_ids() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"id": 1, "text": "Olá"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="identifier mismatch"):
        translate_batch(client, settings, batch)


def test_translate_batch_rejects_unexpected_ids() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response(
        [{"id": 1, "text": "Olá"}, {"id": 2, "text": "mundo"}, {"id": 99, "text": "extra"}]
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="identifier mismatch"):
        translate_batch(client, settings, batch)


def test_translate_batch_rejects_malformed_json() -> None:
    batch = _batch()
    settings = _settings()
    response = MagicMock()
    response.choices = [_FakeChoice(_FakeMessage("not json"))]
    response.usage = None
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError):
        translate_batch(client, settings, batch)


def test_translate_batch_rejects_duplicate_ids() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"id": 1, "text": "Olá"}, {"id": 1, "text": "mundo"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="Duplicate"):
        translate_batch(client, settings, batch)
