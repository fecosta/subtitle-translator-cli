from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from subtranslate.batch_runner import BatchResult
from subtranslate.cli import _batch_progress_callback, app
from subtranslate.scanner import ScanItem

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "translate" in result.output


def test_cli_translate_dry_run(sample_srt: Path) -> None:
    result = runner.invoke(app, ["translate", str(sample_srt), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output or "dry" in result.output.lower()
    assert "Subtitles:" in result.output


def test_cli_missing_api_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    empty = tmp_path / "empty.srt"
    empty.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    result = runner.invoke(app, ["translate", str(empty)])
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_cli_refuses_overwrite(sample_srt: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.srt"
    output.write_text("exists")
    result = runner.invoke(app, ["translate", str(sample_srt), "--output", str(output)])
    assert result.exit_code == 2
    assert "already exists" in result.output


def test_cli_argument_validation(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.srt"
    result = runner.invoke(app, ["translate", str(missing)])
    assert result.exit_code != 0


def test_cli_rejects_invalid_target_language(tmp_path: Path) -> None:
    srt = tmp_path / "movie.en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    result = runner.invoke(app, ["translate", str(srt), "--to", "<>:\"|?*"])
    assert result.exit_code == 1
    assert "Invalid target language tag" in result.output


def _fake_result(client, settings, batch):
    from subtranslate.models import TranslationResult, Usage

    translations = {
        entry.id: f"[PT] {entry.text}"
        for entry in batch.entries
        if entry.id in batch.target_ids
    }
    return TranslationResult(
        translations=translations,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def test_cli_debug_batch_by_id(hope_like_batch: Path) -> None:
    with patch("subtranslate.debug.translate_batch", _fake_result):
        result = runner.invoke(
            app, ["debug", "batch-by-id", str(hope_like_batch), "1361", "--batch-size", "5"]
        )
    assert result.exit_code == 0
    assert "1361" in result.output
    assert "[PT]" in result.output


def test_cli_debug_batch_by_range(hope_like_batch: Path) -> None:
    with patch("subtranslate.debug.translate_batch", _fake_result):
        result = runner.invoke(
            app,
            [
                "debug",
                "batch-by-range",
                str(hope_like_batch),
                "1361",
                "1365",
                "--batch-size",
                "5",
            ],
        )
    assert result.exit_code == 0
    assert "1361" in result.output
    assert "[PT]" in result.output


def _write_srt(path: Path, text: str = "Hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nLine two\n",
        encoding="utf-8",
    )


def _fake_translate(input_path: Path, output_path: Path, settings, **kwargs):
    from subtranslate.srt import load_subtitles, save_subtitles

    entries = load_subtitles(input_path)
    save_subtitles(entries, output_path)
    from subtranslate.translator import TranslationStats

    return TranslationStats(
        total_entries=len(entries),
        batch_count=1,
        input_tokens=10,
        output_tokens=5,
        estimated_cost_usd=0.0001,
    )


def test_cli_batch_requires_confirmation(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch("subtranslate.cli.execute_batch") as mock_batch:
        result = runner.invoke(app, ["batch", str(library)], input="n\n")

    assert result.exit_code == 0
    assert "Batch cancelled" in result.output
    mock_batch.assert_not_called()


def test_cli_batch_runs_with_yes(tmp_path: Path) -> None:
    from subtranslate.batch_runner import BatchResult

    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch(
        "subtranslate.cli.execute_batch", return_value=BatchResult(translated=1)
    ) as mock_batch:
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 0
    mock_batch.assert_called_once()


def test_cli_batch_cost_ceiling_exceeded(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    result = runner.invoke(
        app, ["batch", str(library), "--max-cost-usd", "0.00001"]
    )

    assert result.exit_code == 1
    assert "exceeds ceiling" in result.output


def test_cli_batch_unknown_cost_with_ceiling(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    result = runner.invoke(
        app, ["batch", str(library), "--model", "unknown-model", "--max-cost-usd", "1.0"]
    )

    assert result.exit_code == 1
    assert "model pricing is unknown" in result.output


def test_cli_batch_requires_mount(tmp_path: Path, no_mounts) -> None:
    library = tmp_path / "library"
    library.mkdir()

    result = runner.invoke(app, ["batch", str(library), "--require-mount"])

    assert result.exit_code == 2
    assert "not a mounted filesystem" in result.output


def test_cli_batch_rejects_unmounted_mount_point(tmp_path: Path) -> None:
    mount_point = tmp_path / "nas"
    mount_point.mkdir()
    library = mount_point / "movies"
    library.mkdir()

    result = runner.invoke(
        app, ["batch", str(library), "--mount-point", str(mount_point)]
    )

    assert result.exit_code == 2
    assert "not under the required mount point" in result.output


def test_cli_batch_exits_nonzero_on_scan_errors(tmp_path: Path) -> None:
    library = tmp_path / "library"
    bad = library / "Movie A" / "Movie.A.en.srt"
    bad.parent.mkdir(parents=True)
    bad.write_text("not valid srt", encoding="utf-8")

    result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 1
    assert "Errors:" in result.output or "planned_error" in result.output


def test_cli_batch_immediate_feedback_with_yes(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie B" / "Movie.B.en.srt")

    with patch(
        "subtranslate.cli.execute_batch", return_value=BatchResult(translated=2)
    ) as mock_batch:
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 0
    assert "Starting translation of 2 planned file(s)" in result.output
    start_pos = result.output.find("Starting translation")
    summary_pos = result.output.find("Batch Translation Complete")
    assert start_pos != -1
    assert summary_pos != -1
    assert start_pos < summary_pos
    mock_batch.assert_called_once()


def test_cli_batch_immediate_feedback_after_confirmation(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch(
        "subtranslate.cli.execute_batch", return_value=BatchResult(translated=1)
    ) as mock_batch:
        result = runner.invoke(app, ["batch", str(library)], input="y\n")

    assert result.exit_code == 0
    assert "Starting translation of 1 planned file(s)" in result.output
    mock_batch.assert_called_once()


def test_cli_batch_no_feedback_when_cancelled(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    with patch("subtranslate.cli.execute_batch") as mock_batch:
        result = runner.invoke(app, ["batch", str(library)], input="n\n")

    assert result.exit_code == 0
    assert "Starting translation" not in result.output
    mock_batch.assert_not_called()


def test_cli_batch_progress_counts(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie B" / "Movie.B.en.srt")
    _write_srt(library / "Movie C" / "Movie.C.en.srt")

    with patch("subtranslate.batch_runner.translate", _fake_translate):
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 0
    assert "[1/3] Translating Movie.A" in result.output
    assert "[2/3] Translating Movie.B" in result.output
    assert "[3/3] Translating Movie.C" in result.output
    assert "(1/3 completed: 1 ok, 0 failed, 2 remaining)" in result.output
    assert "(2/3 completed: 2 ok, 0 failed, 1 remaining)" in result.output
    assert "(3/3 completed: 3 ok, 0 failed, 0 remaining)" in result.output


def test_cli_batch_reports_failure_and_exits_nonzero(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie B" / "Movie.B.en.srt")

    def conditional_translate(input_path, output_path, settings, **kwargs):
        if "Movie.A" in input_path.name:
            raise RuntimeError("simulated API failure")
        return _fake_translate(input_path, output_path, settings, **kwargs)

    with patch("subtranslate.batch_runner.translate", conditional_translate):
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 1
    assert "Failed:" in result.output
    assert "simulated API failure" in result.output
    assert "Translated" in result.output


def test_cli_batch_summary_content_unchanged(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie B" / "Movie.B.en.srt")

    def conditional_translate(input_path, output_path, settings, **kwargs):
        if "Movie.B" in input_path.name:
            raise RuntimeError("simulated API failure")
        return _fake_translate(input_path, output_path, settings, **kwargs)

    with patch("subtranslate.batch_runner.translate", conditional_translate):
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 1
    assert "Batch Translation Complete" in result.output
    assert "Eligible files:" in result.output
    assert "Translated:" in result.output
    assert "Skipped:" in result.output
    assert "Failed:" in result.output
    assert "Input tokens:" in result.output
    assert "Estimated cost:" in result.output
    assert "Duration:" in result.output
    assert "Per-file issues" in result.output
    assert "Movie.B" in result.output


def _scan_item(movie_name: str = "Movie") -> ScanItem:
    return ScanItem(
        source=Path("x.en.srt"),
        movie_dir=Path("."),
        movie_name=movie_name,
        output_path=Path("x.pt.srt"),
        status="TRANSLATE",
        reason="",
    )


def test_batch_progress_callback_escapes_rich_markup() -> None:
    console = Console(file=io.StringIO())
    callback = _batch_progress_callback(console)

    callback("start", 0, 2, _scan_item("[/x]"), "", "")
    callback("done", 0, 2, _scan_item("[/x]"), "translated", "ok")
    callback("start", 1, 2, _scan_item("[bold]"), "", "")
    callback("done", 1, 2, _scan_item("[bold]"), "failed", "boom")

    output = console.file.getvalue()
    assert "[/x]" in output
    assert "[bold]" in output


@pytest.mark.parametrize(
    "status",
    ["translated", "failed", "skipped_existing", "budget_exceeded"],
)
def test_batch_progress_callback_tty_spinner_lifecycle(status: str) -> None:
    console = Console(force_terminal=True, file=io.StringIO())
    spinner = MagicMock()
    with patch.object(console, "status", return_value=spinner) as mock_status:
        callback = _batch_progress_callback(console)
        item = _scan_item()
        callback("start", 0, 1, item, "", "")
        mock_status.assert_called_once_with("Translating Movie...", spinner="dots")
        spinner.start.assert_called_once()
        callback("done", 0, 1, item, status, "reason")
        spinner.stop.assert_called_once()


def test_batch_progress_callback_close_stops_spinner() -> None:
    console = Console(force_terminal=True, file=io.StringIO())
    spinner = MagicMock()
    with patch.object(console, "status", return_value=spinner):
        callback = _batch_progress_callback(console)
        item = _scan_item()
        callback("start", 0, 1, item, "", "")
        callback.close()
    spinner.stop.assert_called_once()


def test_batch_progress_callback_counts_mixed_results() -> None:
    console = Console(file=io.StringIO())
    callback = _batch_progress_callback(console)
    items = [_scan_item(f"Movie {i}") for i in range(4)]

    callback("start", 0, 4, items[0], "", "")
    callback("done", 0, 4, items[0], "translated", "ok")
    callback("start", 1, 4, items[1], "", "")
    callback("done", 1, 4, items[1], "failed", "boom")
    callback("start", 2, 4, items[2], "", "")
    callback("done", 2, 4, items[2], "skipped_existing", "exists")
    callback("start", 3, 4, items[3], "", "")
    callback("done", 3, 4, items[3], "budget_exceeded", "over")

    output = console.file.getvalue()
    assert "(1/4 completed: 1 ok, 0 failed, 3 remaining)" in output
    assert "(2/4 completed: 1 ok, 1 failed, 2 remaining)" in output
    assert "(3/4 completed: 1 ok, 1 failed, 1 skipped, 1 remaining)" in output
    assert "(4/4 completed: 1 ok, 1 failed, 2 skipped, 0 remaining)" in output


def test_cli_batch_spinner_cleanup_on_interrupt(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    callback_mock = MagicMock()

    with patch("subtranslate.cli.execute_batch", side_effect=KeyboardInterrupt), \
         patch("subtranslate.cli._batch_progress_callback", return_value=callback_mock):
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert result.exit_code == 130
    callback_mock.close.assert_called_once()


def test_cli_batch_spinner_cleanup_on_unexpected_exception(tmp_path: Path) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    callback_mock = MagicMock()

    with patch("subtranslate.cli.execute_batch", side_effect=RuntimeError("boom")), \
         patch("subtranslate.cli._batch_progress_callback", return_value=callback_mock):
        result = runner.invoke(app, ["batch", str(library), "--yes"])

    assert isinstance(result.exception, RuntimeError)
    callback_mock.close.assert_called_once()
