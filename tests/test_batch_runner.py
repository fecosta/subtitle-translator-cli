from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from subtranslate.batch_runner import build_batch_plan, execute_batch
from subtranslate.config import get_settings
from subtranslate.exceptions import ConfigurationError
from subtranslate.translator import TranslationStats


@pytest.fixture
def batch_settings() -> None:
    return get_settings(
        model="gpt-4.1-mini",
        target_lang="pt-BR",
        batch_size=40,
    )


def _write_srt(path: Path, text: str = "Hello world") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond line\n",
        encoding="utf-8",
    )


def _plan(library: Path, settings):
    return build_batch_plan(library, settings)


def _fake_translate(input_path: Path, output_path: Path, settings, **kwargs):
    from subtranslate.srt import load_subtitles, save_subtitles

    entries = load_subtitles(input_path)
    save_subtitles(entries, output_path)
    return TranslationStats(
        total_entries=len(entries),
        batch_count=1,
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0001,
    )


def test_batch_translates_eligible_movie(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    source = library / "Movie A" / "Movie.A.en.srt"
    _write_srt(source)

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        result = execute_batch(library, batch_settings, _plan(library, batch_settings))

    assert result.translated == 1
    assert result.failed == 0
    assert (library / "Movie A" / "Movie.A.pt-BR.srt").exists()


def test_batch_skips_existing_portuguese(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie A" / "Movie.A.pt-BR.srt")

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        result = execute_batch(library, batch_settings, _plan(library, batch_settings))

    assert result.translated == 0
    assert result.skipped == 0  # skipped at planning stage, not execution


def test_batch_handles_api_failure_and_continues(
    tmp_path: Path, batch_settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie B" / "Movie.B.en.srt")

    calls = []

    def conditional_translate(input_path, output_path, settings, **kwargs):
        calls.append(input_path.name)
        if "Movie.A" in input_path.name:
            raise RuntimeError("simulated API failure")
        return _fake_translate(input_path, output_path, settings, **kwargs)

    with patch("subtranslate.batch_runner.translate", conditional_translate):
        result = execute_batch(library, batch_settings, _plan(library, batch_settings))

    assert result.translated == 1
    assert result.failed == 1


def test_batch_resumes_after_failed_movie(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch(
        "subtranslate.batch_runner.translate",
        side_effect=RuntimeError("simulated failure"),
    ):
        result = execute_batch(library, batch_settings, _plan(library, batch_settings))
    assert result.failed == 1

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        result = execute_batch(library, batch_settings, _plan(library, batch_settings))

    assert result.translated == 1
    assert result.failed == 0


def test_batch_skips_output_that_appears_after_planning(
    tmp_path: Path, batch_settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    plan = _plan(library, batch_settings)
    (library / "Movie A" / "Movie.A.pt-BR.srt").write_text("exists")

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        result = execute_batch(library, batch_settings, plan)

    assert result.translated == 0
    assert result.skipped == 1


def test_batch_rejects_missing_mount(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    library.mkdir()
    with pytest.raises(FileNotFoundError):
        build_batch_plan(library, batch_settings, require_mount=True)


def test_batch_rejects_symlinked_root(tmp_path: Path, batch_settings) -> None:
    real_library = tmp_path / "real_library"
    real_library.mkdir()
    link = tmp_path / "link_library"
    link.symlink_to(real_library)
    with pytest.raises(ValueError):
        execute_batch(link, batch_settings, _plan(link, batch_settings))


def test_batch_preserves_source_files(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    source = library / "Movie A" / "Movie.A.en.srt"
    _write_srt(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        execute_batch(library, batch_settings, _plan(library, batch_settings))

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_batch_no_api_calls_during_planning(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch("subtranslate.translator.translate_batch") as mock_batch:
        plan = build_batch_plan(library, batch_settings)
    mock_batch.assert_not_called()
    assert plan.eligible_count == 1


def test_batch_cost_ceiling_per_file(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        result = execute_batch(
            library,
            batch_settings,
            _plan(library, batch_settings),
            max_cost_usd=0.00001,
        )

    assert result.translated == 0
    assert result.budget_exceeded == 1


def test_batch_cost_unknown_with_ceiling(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    settings = get_settings(model="unknown-model", target_lang="pt-BR")

    result = execute_batch(library, settings, _plan(library, settings), max_cost_usd=1.0)

    assert result.translated == 0
    assert result.budget_exceeded == 1


def test_batch_result_exit_code_on_failure(tmp_path: Path, batch_settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch(
        "subtranslate.batch_runner.translate",
        side_effect=RuntimeError("fail"),
    ):
        result = execute_batch(library, batch_settings, _plan(library, batch_settings))

    assert result.failed == 1


def test_batch_lock_prevents_concurrent_runs(tmp_path: Path, batch_settings) -> None:
    from subtranslate.batch_runner import LibraryLock, _lock_path

    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    lock_path = _lock_path(library)

    with (
        LibraryLock(lock_path),
        pytest.raises(ConfigurationError),
        LibraryLock(lock_path),
    ):
        pass
