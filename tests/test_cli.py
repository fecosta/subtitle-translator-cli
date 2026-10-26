from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from subtranslate.cli import app

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
