from __future__ import annotations

import json
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings
from .models import Batch, TranslationResult, Usage


class _TranslationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., description="Original subtitle identifier")
    text: str = Field(..., description="Translated subtitle text")


class _BatchTranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtitles: list[_TranslationItem] = Field(
        ..., description="Translated subtitle entries for the target ids"
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
    "Return ONLY a JSON object with a 'subtitles' array containing the entries "
    "marked 'translate': true. Each item must have its original 'id' and the translated 'text'. "
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

    result: dict[int, str] = {}
    seen_ids: set[int] = set()
    all_ids = batch.all_ids
    for item in subtitles:
        if not isinstance(item, dict):
            raise ValueError("Invalid subtitle item in response.")
        entry_id = item.get("id")
        text = item.get("text")
        if type(entry_id) is not int or not isinstance(text, str):
            raise ValueError("Subtitle item must contain integer 'id' and string 'text'.")
        if entry_id not in all_ids:
            raise ValueError(f"Unexpected subtitle identifier returned: {entry_id}.")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate subtitle identifier returned: {entry_id}.")
        seen_ids.add(entry_id)
        result[entry_id] = text

    # Drop any context entries the model returned despite the prompt.
    result = {
        entry_id: text for entry_id, text in result.items() if entry_id in batch.target_ids
    }
    returned_ids = set(result.keys())

    if returned_ids != batch.target_ids:
        missing = batch.target_ids - returned_ids
        raise ValueError(f"Batch identifier mismatch (missing ids: {sorted(missing)}).")

    for entry_id in batch.target_ids:
        if not result[entry_id].strip():
            raise ValueError(f"Empty translation returned for id {entry_id}.")

    usage = Usage()
    if response.usage:
        usage.input_tokens = response.usage.prompt_tokens or 0
        usage.output_tokens = response.usage.completion_tokens or 0

    return TranslationResult(translations=result, usage=usage)
