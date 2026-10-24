from __future__ import annotations

import os
from dataclasses import dataclass

from .exceptions import ConfigurationError

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "pt-BR"
DEFAULT_BATCH_SIZE = 40

# Approximate model pricing in USD per 1 million tokens.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    model: str
    source_lang: str
    target_lang: str
    batch_size: int


def get_settings(
    model: str | None = None,
    source_lang: str | None = None,
    target_lang: str | None = None,
    batch_size: int | None = None,
) -> Settings:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set.")

    return Settings(
        api_key=api_key,
        model=model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL,
        source_lang=source_lang or DEFAULT_SOURCE_LANG,
        target_lang=target_lang or DEFAULT_TARGET_LANG,
        batch_size=batch_size
        or int(os.environ.get("SUBTRANSLATE_BATCH_SIZE") or DEFAULT_BATCH_SIZE),
    )
