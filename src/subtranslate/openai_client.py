from __future__ import annotations

import json
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .models import Batch, TranslationResult, Usage


class _TranslationText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Translated subtitle text")


class _BatchTranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtitles: list[_TranslationText] = Field(
        ...,
        description=(
            "Translated subtitle texts, one per target entry, in the same order "
            "as the entries marked 'translate': true"
        ),
    )


_JSON_SCHEMA = {
    "name": "subtitle_translation",
    "strict": True,
    "schema": _BatchTranslationResponse.model_json_schema(),
}

SYSTEM_PROMPT = (
    "You are an expert subtitle translator. "
    "Translate the provided subtitles naturally into the target language. "
    "Preserve meaning, conversational context, character names, proper nouns, "
    "speaker distinctions, and any formatting tags. "
    "Prefer natural Brazilian Portuguese over literal translations when the target is pt-BR. "
    "Do not add, remove, or invent dialogue. "
    "Do not include explanations, notes, or markdown. "
    "Return ONLY a JSON object with a 'subtitles' array. "
    "The array must contain exactly one item for each entry marked 'translate': true, "
    "in the same order as those entries. "
    "Each item must be an object with only the translated 'text'. "
    "Do not include the 'id' field. "
    "Entries marked 'translate': false are context only; do not return them."
)


def _build_user_message(batch: Batch, source_lang: str, target_lang: str) -> str:
    payload: list[dict[str, Any]] = []
    for entry in batch.entries:
        payload.append(
            {
                "id": entry.id,
                "translate": entry.id in batch.target_ids,
                "text": entry.text,
            }
        )
    return json.dumps(
        {
            "source_language": source_lang,
            "target_language": target_lang,
            "subtitles": payload,
        },
        ensure_ascii=False,
    )


def _supports_json_schema(model: str) -> bool:
    """Return True for model families known to support strict JSON schemas."""
    name = model.lower()
    return name.startswith(("gpt-4o", "gpt-4.1"))


def _response_format(model: str) -> dict[str, Any]:
    if _supports_json_schema(model):
        return {"type": "json_schema", "json_schema": _JSON_SCHEMA}
    return {"type": "json_object"}


def _extract_text(item: Any) -> str:
    """Return the translated text from a positional response item."""
    if isinstance(item, dict):
        value = item.get("text")
    elif isinstance(item, str):
        value = item
    else:
        raise ValueError("Subtitle item must be an object with 'text' or a string.")
    if not isinstance(value, str):
        raise ValueError("Subtitle text must be a string.")
    return value


def translate_batch(
    client: openai.OpenAI,
    settings: Settings,
    batch: Batch,
) -> TranslationResult:
    """Translate a single batch and return id-to-text mapping plus token usage."""
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_message(
                    batch, settings.source_lang, settings.target_lang
                ),
            },
        ],
        response_format=_response_format(settings.model),
        temperature=0.3,
    )

    if not response.choices or not response.choices[0].message.content:
        raise ValueError("OpenAI returned an empty response.")

    content = response.choices[0].message.content
    data = json.loads(content)
    if not isinstance(data, dict) or "subtitles" not in data:
        raise ValueError("OpenAI response is missing the 'subtitles' key.")

    subtitles = data["subtitles"]
    if not isinstance(subtitles, list):
        raise ValueError("OpenAI response 'subtitles' must be a list.")

    target_entries = [entry for entry in batch.entries if entry.id in batch.target_ids]
    if len(subtitles) != len(target_entries):
        raise ValueError(
            f"Batch size mismatch: expected {len(target_entries)} translations, "
            f"received {len(subtitles)}."
        )

    result: dict[int, str] = {}
    for entry, item in zip(target_entries, subtitles, strict=True):
        text = _extract_text(item)
        if not text.strip():
            raise ValueError(f"Empty translation returned for id {entry.id}.")
        result[entry.id] = text

    usage = Usage()
    if response.usage:
        usage.input_tokens = response.usage.prompt_tokens or 0
        usage.output_tokens = response.usage.completion_tokens or 0

    return TranslationResult(translations=result, usage=usage)
