from __future__ import annotations

import json
from typing import Any

import openai

from .config import Settings
from .models import Batch, TranslationResult, Usage

SYSTEM_PROMPT = (
    "You are an expert subtitle translator. "
    "Translate the provided subtitles naturally into the target language. "
    "Preserve meaning, conversational context, character names, proper nouns, "
    "speaker distinctions, and any formatting tags. "
    "Prefer natural Brazilian Portuguese over literal translations when the target is pt-BR. "
    "Do not add, remove, or invent dialogue. "
    "Do not include explanations, notes, or markdown. "
    "Return ONLY a JSON object with a 'subtitles' array containing every input entry, "
    "each with its original 'id' and the translated 'text'. "
    "Entries marked 'translate': false are context; return them unchanged."
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
        response_format={"type": "json_object"},
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
    for item in subtitles:
        if not isinstance(item, dict):
            raise ValueError("Invalid subtitle item in response.")
        entry_id = item.get("id")
        text = item.get("text")
        if not isinstance(entry_id, int) or not isinstance(text, str):
            raise ValueError("Subtitle item must contain integer 'id' and string 'text'.")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate subtitle identifier returned: {entry_id}.")
        seen_ids.add(entry_id)
        result[entry_id] = text

    if seen_ids != batch.all_ids:
        missing = batch.all_ids - seen_ids
        extra = seen_ids - batch.all_ids
        parts = []
        if missing:
            parts.append(f"missing ids: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected ids: {sorted(extra)}")
        raise ValueError(f"Batch identifier mismatch ({'; '.join(parts)}).")

    usage = Usage()
    if response.usage:
        usage.input_tokens = response.usage.prompt_tokens or 0
        usage.output_tokens = response.usage.completion_tokens or 0

    return TranslationResult(translations=result, usage=usage)
