from __future__ import annotations

from pathlib import Path

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
