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
