from __future__ import annotations

from pathlib import Path

import openai

from .batching import make_batches
from .config import Settings
from .models import Batch, SubtitleEntry, TranslationResult
from .openai_client import translate_batch
from .srt import load_subtitles


def _find_batch_for_id(entries: list[SubtitleEntry], settings: Settings, target_id: int) -> Batch:
    """Return the translation batch that contains ``target_id``."""
    batches = make_batches(entries, settings.batch_size)
    for batch in batches:
        if target_id in batch.target_ids:
            return batch
    raise ValueError(f"Subtitle id {target_id} not found in any batch")


def _find_batch_for_range(
    entries: list[SubtitleEntry], settings: Settings, start_id: int, end_id: int
) -> Batch:
    """Return a batch whose target ids fall within ``[start_id, end_id]``."""
    if start_id > end_id:
        raise ValueError("start_id must be <= end_id")
    batches = make_batches(entries, settings.batch_size)
    for batch in batches:
        if batch.target_ids <= set(range(start_id, end_id + 1)):
            return batch
    raise ValueError(f"No complete batch found in range {start_id}-{end_id}")


def translate_batch_by_id(
    input_path: Path,
    settings: Settings,
    target_id: int,
) -> tuple[Batch, TranslationResult]:
    """Translate the single batch containing ``target_id``."""
    entries = load_subtitles(input_path)
    batch = _find_batch_for_id(entries, settings, target_id)
    client = openai.OpenAI(api_key=settings.api_key)
    result = translate_batch(client, settings, batch)
    return batch, result


def translate_batch_by_range(
    input_path: Path,
    settings: Settings,
    start_id: int,
    end_id: int,
) -> tuple[Batch, TranslationResult]:
    """Translate the first fully-contained batch in the given id range."""
    entries = load_subtitles(input_path)
    batch = _find_batch_for_range(entries, settings, start_id, end_id)
    client = openai.OpenAI(api_key=settings.api_key)
    result = translate_batch(client, settings, batch)
    return batch, result
