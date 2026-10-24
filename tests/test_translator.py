from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from subtranslate.exceptions import FileConflictError, OutputExistsError, TranslationError
from subtranslate.models import TranslationResult, Usage
from subtranslate.srt import load_subtitles
from subtranslate.translator import get_dry_run_info, translate


def _fake_result(client, settings, batch) -> TranslationResult:
    translations = {
        entry.id: f"[PT] {entry.text}"
        for entry in batch.entries
        if entry.id in batch.target_ids
    }
    return TranslationResult(
        translations=translations,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def test_translate_preserves_count_and_timestamps(
    sample_srt: Path, settings, tmp_path: Path
) -> None:
    output = tmp_path / "out.srt"

    def fake_translate_batch(client, settings, batch):
        return _fake_result(client, settings, batch)

    with patch("subtranslate.translator.translate_batch", fake_translate_batch):
        stats = translate(sample_srt, output, settings)

    assert stats.total_entries == 5
    original = load_subtitles(sample_srt)
    translated = load_subtitles(output)
    assert len(translated) == len(original)
    for orig, new in zip(original, translated, strict=True):
        assert orig.start == new.start
        assert orig.end == new.end
        assert new.text.startswith("[PT]")


def test_translate_refuses_overwrite(sample_srt: Path, settings, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"
    output.write_text("exists")
    with pytest.raises(OutputExistsError):
        translate(sample_srt, output, settings)


def test_translate_refuses_input_as_output(sample_srt: Path, settings) -> None:
    with pytest.raises(FileConflictError):
        translate(sample_srt, sample_srt, settings, force=True)


def test_translate_allows_overwrite_with_force(sample_srt: Path, settings, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"
    output.write_text("exists")

    with patch("subtranslate.translator.translate_batch", _fake_result):
        translate(sample_srt, output, settings, force=True)
    assert load_subtitles(output)


def test_translate_rejects_missing_ids(sample_srt: Path, settings, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"

    def fake_translate_batch(client, settings, batch):
        # Return one fewer target entry than required.
        translations = {
            entry.id: f"[PT] {entry.text}"
            for entry in batch.entries
            if entry.id in batch.target_ids
        }
        translations.pop(next(iter(batch.target_ids)))
        return TranslationResult(translations=translations)

    with (
        patch("subtranslate.translator.translate_batch", fake_translate_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    assert not output.exists()


def test_translate_rejects_unknown_ids(sample_srt: Path, settings, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"

    def fake_translate_batch(client, settings, batch):
        translations = {
            entry.id: f"[PT] {entry.text}"
            for entry in batch.entries
            if entry.id in batch.target_ids
        }
        # Inject an identifier that does not belong to the batch.
        first = next(iter(batch.target_ids))
        translations[first + 1000] = "[PT] extra"
        return TranslationResult(translations=translations)

    with (
        patch("subtranslate.translator.translate_batch", fake_translate_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)


def test_translate_retries_then_succeeds(sample_srt: Path, settings, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"
    calls = []

    def fake_translate_batch(client, settings, batch):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("malformed response")
        return _fake_result(client, settings, batch)

    with patch("subtranslate.translator.translate_batch", fake_translate_batch), patch(
        "subtranslate.translator.time.sleep"
    ):
        translate(sample_srt, output, settings)

    assert len(calls) > 1
    assert output.exists()


def test_dry_run_makes_no_api_calls(sample_srt: Path, settings, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"

    with patch("subtranslate.translator.translate_batch") as mock_translate:
        stats = translate(sample_srt, output, settings, dry_run=True)
    mock_translate.assert_not_called()
    assert not output.exists()
    assert stats.dry_run
    assert stats.total_entries == 5
    assert stats.batch_count == 3


def test_get_dry_run_info(sample_srt: Path, settings) -> None:
    info = get_dry_run_info(sample_srt, settings)
    assert info.total_entries == 5
    assert info.batch_count == 3
    assert info.estimated_input_tokens > 0
    assert info.model == settings.model


def test_context_entries_not_required_in_response(
    sample_srt: Path, settings, tmp_path: Path
) -> None:
    """Regression: context entries may be omitted from the model response."""
    output = tmp_path / "out.srt"

    def fake_translate_batch(client, settings, batch):
        # Return only target ids, even if context ids are present in the batch.
        return TranslationResult(
            translations={
                entry.id: f"[PT] {entry.text}"
                for entry in batch.entries
                if entry.id in batch.target_ids
            }
        )

    with patch("subtranslate.translator.translate_batch", fake_translate_batch):
        stats = translate(sample_srt, output, settings)

    assert stats.total_entries == 5
    translated = load_subtitles(output)
    for entry in translated:
        assert entry.text.startswith("[PT]")


def test_context_entries_returned_as_translations_are_rejected(
    sample_srt: Path, settings, tmp_path: Path
) -> None:
    """Regression: returning context ids as translations is treated as unexpected."""
    output = tmp_path / "out.srt"

    def fake_translate_batch(client, settings, batch):
        translations = {
            entry.id: f"[PT] {entry.text}" for entry in batch.entries
        }
        return TranslationResult(translations=translations)

    with (
        patch("subtranslate.translator.translate_batch", fake_translate_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    assert not output.exists()


def test_empty_target_translation_is_rejected(
    sample_srt: Path, settings, tmp_path: Path
) -> None:
    output = tmp_path / "out.srt"

    def fake_translate_batch(client, settings, batch):
        translations = {
            entry.id: f"[PT] {entry.text}"
            for entry in batch.entries
            if entry.id in batch.target_ids
        }
        translations[next(iter(batch.target_ids))] = "   "
        return TranslationResult(translations=translations)

    with (
        patch("subtranslate.translator.translate_batch", fake_translate_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    assert not output.exists()
