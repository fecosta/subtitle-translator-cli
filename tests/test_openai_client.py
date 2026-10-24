from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from subtranslate.config import Settings
from subtranslate.models import Batch, SubtitleEntry
from subtranslate.openai_client import SYSTEM_PROMPT, _response_format, translate_batch


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


def _make_response(translations: list[Any], prompt_tokens: int = 10, completion_tokens: int = 5):
    content = json.dumps({"subtitles": translations})
    response = MagicMock()
    response.choices = [_FakeChoice(_FakeMessage(content))]
    response.usage = _FakeUsage(prompt_tokens, completion_tokens)
    return response


def _batch() -> Batch:
    entries = [
        SubtitleEntry(id=1, start=1000, end=2000, text="Before"),
        SubtitleEntry(id=2, start=2000, end=3000, text="Hello"),
        SubtitleEntry(id=3, start=3000, end=4000, text="World"),
        SubtitleEntry(id=4, start=4000, end=5000, text="After"),
    ]
    return Batch(entries=entries, target_ids={2, 3})


def _settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=2,
    )


def test_system_prompt_requests_positional_responses() -> None:
    assert "in the same order" in SYSTEM_PROMPT.lower()
    assert "do not include the 'id'" in SYSTEM_PROMPT.lower()
    assert "do not return them" in SYSTEM_PROMPT.lower()


def test_translate_batch_sends_context_entries() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"text": "Olá"}, {"text": "mundo"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    translate_batch(client, settings, batch)

    call_args = client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    payload = json.loads(messages[1]["content"])
    sent_ids = {item["id"] for item in payload["subtitles"]}
    assert sent_ids == {1, 2, 3, 4}
    context_items = [item for item in payload["subtitles"] if not item["translate"]]
    assert {item["id"] for item in context_items} == {1, 4}


def test_translate_batch_maps_responses_by_position() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"text": "Olá"}, {"text": "mundo"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    result = translate_batch(client, settings, batch)
    assert result.translations == {2: "Olá", 3: "mundo"}


def test_translate_batch_accepts_text_strings_in_fallback_mode() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response(["Olá", "mundo"])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    result = translate_batch(client, settings, batch)
    assert result.translations == {2: "Olá", 3: "mundo"}


def test_translate_batch_rejects_too_few_responses() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"text": "Olá"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="size mismatch"):
        translate_batch(client, settings, batch)


def test_translate_batch_rejects_too_many_responses() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"text": "Olá"}, {"text": "mundo"}, {"text": "extra"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="size mismatch"):
        translate_batch(client, settings, batch)


def test_translate_batch_rejects_empty_translation() -> None:
    batch = _batch()
    settings = _settings()
    response = _make_response([{"text": "Olá"}, {"text": "   "}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="Empty translation"):
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


def test_translate_batch_rejects_non_list_subtitles() -> None:
    batch = _batch()
    settings = _settings()
    response = MagicMock()
    response.choices = [_FakeChoice(_FakeMessage(json.dumps({"subtitles": None})))]
    response.usage = None
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with pytest.raises(ValueError, match="list"):
        translate_batch(client, settings, batch)


def test_structured_output_for_supported_models() -> None:
    fmt = _response_format("gpt-4.1-mini")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert "schema" in fmt["json_schema"]


def test_json_object_fallback_for_unsupported_models() -> None:
    fmt = _response_format("gpt-3.5-turbo")
    assert fmt["type"] == "json_object"


def test_positional_mapping_with_high_subtitle_ids() -> None:
    """Regression for id hallucination (e.g. 1361 -> 13561)."""
    entries = [
        SubtitleEntry(id=1360, start=1000, end=2000, text="Before"),
        SubtitleEntry(id=1361, start=2000, end=3000, text="Target"),
        SubtitleEntry(id=1362, start=3000, end=4000, text="After"),
    ]
    batch = Batch(entries=entries, target_ids={1361})
    settings = _settings()
    response = _make_response([{"text": "Alvo"}])
    client = MagicMock()
    client.chat.completions.create.return_value = response

    result = translate_batch(client, settings, batch)
    assert result.translations == {1361: "Alvo"}
