from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import openai

from .batching import make_batches
from .config import MODEL_PRICING, Settings
from .exceptions import (
    ConfigurationError,
    FileConflictError,
    OutputExistsError,
    TranslationError,
)
from .fileio import atomic_write_path
from .models import SubtitleEntry, TranslationResult, Usage
from .openai_client import translate_batch
from .srt import load_subtitles

MAX_RETRIES = 3
PROMPT_OVERHEAD_TOKENS = 200


@dataclass(slots=True)
class TranslationStats:
    duration_seconds: float = 0.0
    total_entries: int = 0
    batch_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    dry_run: bool = False


@dataclass(slots=True)
class DryRunInfo:
    total_entries: int
    batch_count: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float | None
    model: str


def _estimate_tokens(text: str) -> int:
    """Very rough token estimate; sufficient for dry-run planning."""
    return max(1, len(text) // 4)


def _validate_batch(result: TranslationResult, batch) -> None:
    """Ensure the model returned exactly the target entries, each translated."""
    returned_ids = set(result.translations.keys())

    if len(returned_ids) != len(result.translations):
        raise ValueError("Duplicate subtitle identifiers returned by the model.")

    if returned_ids != batch.target_ids:
        missing = batch.target_ids - returned_ids
        extra = returned_ids - batch.target_ids
        parts = []
        if missing:
            parts.append(f"missing ids: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected ids: {sorted(extra)}")
        raise ValueError(f"Batch identifier mismatch ({'; '.join(parts)}).")

    for entry_id in batch.target_ids:
        text = result.translations[entry_id]
        if not text.strip():
            raise ValueError(f"Empty translation returned for id {entry_id}.")


def _translate_batch_with_retry(
    client: openai.OpenAI,
    settings: Settings,
    batch,
    progress_callback: Callable[[int, int], None] | None = None,
    batch_index: int = 0,
    total_batches: int = 0,
) -> TranslationResult:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = translate_batch(client, settings, batch)
            _validate_batch(result, batch)
            if progress_callback:
                progress_callback(batch_index + 1, total_batches)
            return result
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
            continue

    raise TranslationError(
        f"Batch {batch_index + 1}/{total_batches} failed after "
        f"{MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error


def _build_translated_entries(
    original: list[SubtitleEntry], translations: dict[int, str]
) -> list[SubtitleEntry]:
    """Build a new entry list preserving ids, timestamps and order."""
    return [
        SubtitleEntry(
            id=entry.id,
            start=entry.start,
            end=entry.end,
            text=translations[entry.id],
        )
        for entry in original
    ]


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return None
    return (
        pricing["input"] * input_tokens + pricing["output"] * output_tokens
    ) / 1_000_000


def _dry_run_info(entries: list[SubtitleEntry], settings: Settings) -> DryRunInfo:
    batches = make_batches(entries, settings.batch_size)
    text_chars = sum(len(entry.text) for entry in entries)
    estimated_input = (
        sum(_estimate_tokens(entry.text) for entry in entries)
        + len(batches) * PROMPT_OVERHEAD_TOKENS
    )
    estimated_output = text_chars // 3
    return DryRunInfo(
        total_entries=len(entries),
        batch_count=len(batches),
        estimated_input_tokens=estimated_input,
        estimated_output_tokens=estimated_output,
        estimated_cost_usd=_calculate_cost(
            settings.model, estimated_input, estimated_output
        ),
        model=settings.model,
    )


def translate(
    input_path: Path,
    output_path: Path | None,
    settings: Settings,
    *,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TranslationStats:
    """Translate an SRT file and write the result."""
    if not settings.api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set.")

    entries = load_subtitles(input_path)

    if output_path is None:
        output_path = _default_output_path(input_path, settings.target_lang)

    if output_path.exists() and not force and not dry_run:
        raise OutputExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    if input_path.resolve() == output_path.resolve():
        raise FileConflictError(
            "Output path cannot be the same as the input subtitle file."
        )

    batches = make_batches(entries, settings.batch_size)

    if dry_run:
        return TranslationStats(
            duration_seconds=0.0,
            total_entries=len(entries),
            batch_count=len(batches),
            dry_run=True,
        )

    client = openai.OpenAI(api_key=settings.api_key)
    translations: dict[int, str] = {}
    total_usage = Usage()
    start = time.perf_counter()

    try:
        for index, batch in enumerate(batches):
            result = _translate_batch_with_retry(
                client,
                settings,
                batch,
                progress_callback=progress_callback,
                batch_index=index,
                total_batches=len(batches),
            )
            for entry_id in batch.target_ids:
                translations[entry_id] = result.translations[entry_id]
            total_usage.input_tokens += result.usage.input_tokens
            total_usage.output_tokens += result.usage.output_tokens
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError(f"Translation failed: {exc}") from exc

    duration = time.perf_counter() - start

    if len(translations) != len(entries):
        raise TranslationError(
            f"Incomplete translation: {len(translations)}/{len(entries)} entries received."
        )

    translated_entries = _build_translated_entries(entries, translations)

    try:
        atomic_write_path(translated_entries, output_path, force=force)
    except (OutputExistsError, FileConflictError):
        raise
    except Exception as exc:
        raise TranslationError(f"Could not write output file: {exc}") from exc

    stats = TranslationStats(
        duration_seconds=duration,
        total_entries=len(entries),
        batch_count=len(batches),
        input_tokens=total_usage.input_tokens,
        output_tokens=total_usage.output_tokens,
        estimated_cost_usd=_calculate_cost(
            settings.model, total_usage.input_tokens, total_usage.output_tokens
        ),
    )
    return stats


def _default_output_path(input_path: Path, target_lang: str) -> Path:
    """Generate a default output path based on the input name."""
    suffix = target_lang.replace(" ", "-").replace("(", "").replace(")", "")
    stem = input_path.stem
    new_stem = f"{stem[:-3]}.{suffix}" if stem.lower().endswith(".en") else f"{stem}.{suffix}"
    return input_path.with_stem(new_stem)


def get_dry_run_info(input_path: Path, settings: Settings) -> DryRunInfo:
    entries = load_subtitles(input_path)
    return _dry_run_info(entries, settings)
