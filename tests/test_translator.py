from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from subtranslate.checkpoint import (
    build_identity,
    checkpoint_path,
    compute_source_hash,
    load_checkpoint,
    save_checkpoint,
)
from subtranslate.config import Settings
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


def test_checkpoint_created_on_failure_and_removed_on_success(
    sample_srt: Path, settings, tmp_path: Path, checkpoint_dir: Path
) -> None:
    output = tmp_path / "out.srt"
    cp_path = checkpoint_path(sample_srt, settings)

    def failing_batch(client, settings, batch):
        if 3 in batch.target_ids:
            raise ValueError("simulated failure")
        return _fake_result(client, settings, batch)

    with (
        patch("subtranslate.translator.translate_batch", failing_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    assert not output.exists()
    assert cp_path.exists()

    with patch("subtranslate.translator.translate_batch", _fake_result):
        translate(sample_srt, output, settings)

    assert output.exists()
    assert not cp_path.exists()


def test_resume_skips_completed_batches(
    sample_srt: Path, settings, tmp_path: Path, checkpoint_dir: Path
) -> None:
    output = tmp_path / "out.srt"
    cp_path = checkpoint_path(sample_srt, settings)

    fail_on_three = True

    def conditional_batch(client, settings, batch):
        nonlocal fail_on_three
        if fail_on_three and 3 in batch.target_ids:
            raise ValueError("simulated failure")
        return _fake_result(client, settings, batch)

    with (
        patch("subtranslate.translator.translate_batch", conditional_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    assert cp_path.exists()
    fail_on_three = False

    with patch("subtranslate.translator.translate_batch", conditional_batch):
        translate(sample_srt, output, settings)

    translated = load_subtitles(output)
    assert len(translated) == 5


def test_completed_batches_not_retransmitted(
    sample_srt: Path, settings, tmp_path: Path, checkpoint_dir: Path
) -> None:
    output = tmp_path / "out.srt"

    calls = []

    def tracking_batch(client, settings, batch):
        calls.append(sorted(batch.target_ids))
        return _fake_result(client, settings, batch)

    # Pre-populate checkpoint with ids 1 and 2.
    cp_path = checkpoint_path(sample_srt, settings)
    identity = build_identity(sample_srt, settings, compute_source_hash(sample_srt))
    save_checkpoint(cp_path, identity, {1: "[PT] one", 2: "[PT] two"})

    with patch("subtranslate.translator.translate_batch", tracking_batch):
        translate(sample_srt, output, settings)

    # Batches for ids {1,2} should have been skipped.
    assert [1, 2] not in calls
    assert output.exists()


def test_smaller_batch_recovery(
    sample_srt: Path, settings, tmp_path: Path
) -> None:
    """A batch that fails as a whole succeeds after being split into smaller units."""
    output = tmp_path / "out.srt"

    def split_sensitive_batch(client, settings, batch):
        if len(batch.target_ids) > 1 and 3 in batch.target_ids:
            raise ValueError("too many entries")
        return _fake_result(client, settings, batch)

    with patch("subtranslate.translator.translate_batch", split_sensitive_batch):
        translate(sample_srt, output, settings)

    translated = load_subtitles(output)
    assert len(translated) == 5
    for entry in translated:
        assert entry.text.startswith("[PT]")


def test_diagnostic_message_includes_batch_and_checkpoint_details(
    sample_srt: Path, settings, tmp_path: Path, checkpoint_dir: Path
) -> None:
    output = tmp_path / "out.srt"

    def failing_batch(client, settings, batch):
        if 3 in batch.target_ids:
            raise ValueError("bad output")
        return _fake_result(client, settings, batch)

    with (
        patch("subtranslate.translator.translate_batch", failing_batch),
        pytest.raises(TranslationError) as exc_info,
    ):
        translate(sample_srt, output, settings)

    message = str(exc_info.value)
    assert "Batch 2/3" in message
    assert "Expected target ids" in message
    assert "Rerun the same command" in message
    assert str(checkpoint_dir) in message


def test_split_sub_batches_are_checkpointed_independently(
    sample_srt: Path, settings, tmp_path: Path, checkpoint_dir: Path
) -> None:
    """Successful split sub-batches are checkpointed even if a sibling fails."""
    output = tmp_path / "out.srt"
    cp_path = checkpoint_path(sample_srt, settings)

    def partial_batch(client, settings, batch):
        if 4 in batch.target_ids and len(batch.target_ids) > 1:
            raise ValueError("entry four fails in a larger batch")
        if batch.target_ids == {4}:
            raise ValueError("entry four fails alone")
        return _fake_result(client, settings, batch)

    with (
        patch("subtranslate.translator.translate_batch", partial_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    assert not output.exists()
    # Batch 1 ({1,2}) and the left split of batch 2 ({3}) should be saved.
    identity = build_identity(sample_srt, settings, compute_source_hash(sample_srt))
    checkpoint = load_checkpoint(cp_path, identity, {1, 2, 3, 4, 5})
    assert checkpoint.get(1)
    assert checkpoint.get(2)
    assert checkpoint.get(3)
    assert 4 not in checkpoint


def test_positional_response_maps_large_ids_correctly(
    hope_like_batch: Path, tmp_path: Path
) -> None:
    """Regression for id hallucination: positional responses map to the right ids."""
    settings = Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=40,
    )
    output = tmp_path / "out.srt"

    def positional_batch(client, settings, batch):
        translations = {
            entry.id: f"[PT] {entry.text}"
            for entry in batch.entries
            if entry.id in batch.target_ids
        }
        return TranslationResult(
            translations=translations,
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    with patch("subtranslate.translator.translate_batch", positional_batch):
        translate(hope_like_batch, output, settings)

    translated = load_subtitles(output)
    assert any(
        entry.text == "[PT] Line thirteen sixty-one" for entry in translated
    )


def test_resume_after_partial_split_does_not_retransmit_completed_ids(
    sample_srt: Path, settings, tmp_path: Path, checkpoint_dir: Path
) -> None:
    """Resume after a partial split only translates the still-missing ids."""
    output = tmp_path / "out.srt"
    cp_path = checkpoint_path(sample_srt, settings)

    def partial_batch(client, settings, batch):
        if 4 in batch.target_ids and len(batch.target_ids) > 1:
            raise ValueError("entry four fails in a larger batch")
        if batch.target_ids == {4}:
            raise ValueError("entry four fails alone")
        return _fake_result(client, settings, batch)

    with (
        patch("subtranslate.translator.translate_batch", partial_batch),
        pytest.raises(TranslationError),
    ):
        translate(sample_srt, output, settings)

    identity = build_identity(sample_srt, settings, compute_source_hash(sample_srt))
    checkpoint = load_checkpoint(cp_path, identity, {1, 2, 3, 4, 5})
    assert 1 in checkpoint
    assert 2 in checkpoint
    assert 3 in checkpoint
    assert 4 not in checkpoint

    # Resume: only id 4 should be sent to the model.
    calls_after_resume = []

    def resume_batch(client, settings, batch):
        calls_after_resume.append(sorted(batch.target_ids))
        return _fake_result(client, settings, batch)

    with patch("subtranslate.translator.translate_batch", resume_batch):
        translate(sample_srt, output, settings)

    assert output.exists()
    assert [4] in calls_after_resume
    assert [3] not in calls_after_resume
    assert [3, 4] not in calls_after_resume
