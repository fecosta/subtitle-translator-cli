from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import openai

from .batching import make_batches
from .checkpoint import (
    build_identity,
    checkpoint_path,
    compute_source_hash,
    load_checkpoint,
    remove_checkpoint,
    save_checkpoint,
)
from .config import MODEL_PRICING, Settings
from .exceptions import (
    BatchValidationError,
    ConfigurationError,
    FileConflictError,
    OutputExistsError,
    TranslationError,
)
from .fileio import atomic_write_path
from .models import Batch, SubtitleEntry, TranslationResult, Usage
from .naming import default_output_path
from .openai_client import translate_batch
from .srt import load_subtitles

MAX_RETRIES = 3
MIN_SPLIT_SIZE = 1
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


def _validate_batch(result: TranslationResult, batch: Batch) -> None:
    """Ensure the model returned exactly the target entries, each translated."""
    returned_ids = set(result.translations.keys())

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
    batch: Batch,
) -> TranslationResult:
    """Translate a batch with bounded retries."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = translate_batch(client, settings, batch)
            _validate_batch(result, batch)
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

    message = f"Batch failed after {MAX_RETRIES + 1} attempts: {last_error}"
    if isinstance(last_error, ValueError):
        raise BatchValidationError(message) from last_error
    raise TranslationError(message) from last_error


def _detailed_batch_error(
    batch: Batch,
    batch_index: int,
    total_batches: int,
    checkpoint_path: Path,
    cause: Exception,
) -> TranslationError:
    """Build a diagnostic error message for a failed batch."""
    target_ids = sorted(batch.target_ids)
    if len(target_ids) > 1:
        target_range = f"{target_ids[0]}-{target_ids[-1]}"
    else:
        target_range = str(target_ids[0])
    return TranslationError(
        f"Batch {batch_index + 1}/{total_batches} failed after {MAX_RETRIES + 1} attempts.\n"
        f"Expected target ids: {target_ids}\n"
        f"Range: {target_range}\n"
        f"Cause: {cause}\n"
        f"Completed batches were checkpointed.\n"
        f"Checkpoint: {checkpoint_path}\n"
        "Rerun the same command to resume the translation."
    )


def _translate_batch_recursive(
    client: openai.OpenAI,
    settings: Settings,
    batch: Batch,
    batch_index: int,
    total_batches: int,
    cp_path: Path,
    identity: dict,
    translations: dict[int, str],
) -> TranslationResult:
    """Translate a batch, splitting it on validation failures.

    Successful translations are merged into ``translations`` and checkpointed
    immediately, so even split sub-batches are preserved if a sibling fails.
    """
    try:
        result = _translate_batch_with_retry(client, settings, batch)
    except BatchValidationError as exc:
        target_ids = sorted(batch.target_ids)
        if len(target_ids) <= MIN_SPLIT_SIZE:
            raise _detailed_batch_error(
                batch, batch_index, total_batches, cp_path, exc
            ) from exc

        # Split target entries in half and translate each half with the same context.
        mid = len(target_ids) // 2
        left_batch = Batch(entries=batch.entries, target_ids=set(target_ids[:mid]))
        right_batch = Batch(entries=batch.entries, target_ids=set(target_ids[mid:]))

        left_result = _translate_batch_recursive(
            client,
            settings,
            left_batch,
            batch_index,
            total_batches,
            cp_path,
            identity,
            translations,
        )
        right_result = _translate_batch_recursive(
            client,
            settings,
            right_batch,
            batch_index,
            total_batches,
            cp_path,
            identity,
            translations,
        )
        combined = {**left_result.translations, **right_result.translations}
        usage = Usage(
            input_tokens=left_result.usage.input_tokens + right_result.usage.input_tokens,
            output_tokens=left_result.usage.output_tokens + right_result.usage.output_tokens,
        )
        return TranslationResult(translations=combined, usage=usage)
    except TranslationError as exc:
        raise _detailed_batch_error(
            batch, batch_index, total_batches, cp_path, exc
        ) from exc

    for entry_id, text in result.translations.items():
        translations[entry_id] = text
    save_checkpoint(cp_path, identity, translations)
    return result


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
    """Translate an SRT file and write the result, using checkpoints for resume."""
    if not settings.api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set.")

    entries = load_subtitles(input_path)

    if output_path is None:
        output_path = default_output_path(input_path, settings.target_lang)

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

    source_hash = compute_source_hash(input_path)
    cp_path = checkpoint_path(input_path, settings, source_hash)
    identity = build_identity(input_path, settings, source_hash)

    expected_ids = {entry.id for entry in entries}
    translations: dict[int, str] = load_checkpoint(cp_path, identity, expected_ids)

    if translations and verbose:
        print(f"Resuming from checkpoint: {cp_path} ({len(translations)} entries)")

    client = openai.OpenAI(api_key=settings.api_key)
    total_usage = Usage()
    start = time.perf_counter()

    try:
        for index, batch in enumerate(batches):
            missing_targets = batch.target_ids - translations.keys()
            if not missing_targets:
                if progress_callback:
                    progress_callback(index + 1, len(batches))
                continue

            batch_to_translate = batch
            if missing_targets != batch.target_ids:
                batch_to_translate = Batch(
                    entries=batch.entries, target_ids=missing_targets
                )

            batch_result = _translate_batch_recursive(
                client,
                settings,
                batch_to_translate,
                index,
                len(batches),
                cp_path,
                identity,
                translations,
            )

            total_usage.input_tokens += batch_result.usage.input_tokens
            total_usage.output_tokens += batch_result.usage.output_tokens

            if progress_callback:
                progress_callback(index + 1, len(batches))
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

    remove_checkpoint(cp_path)

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



def get_dry_run_info(input_path: Path, settings: Settings) -> DryRunInfo:
    entries = load_subtitles(input_path)
    return _dry_run_info(entries, settings)
