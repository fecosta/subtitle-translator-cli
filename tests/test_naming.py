from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from subtranslate.naming import default_output_path, parse_subtitle_name


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        ("Movie.eng.srt", "Movie.pt-BR.srt"),
        ("Movie.en.srt", "Movie.pt-BR.srt"),
        ("Movie.english.srt", "Movie.pt-BR.srt"),
        ("Movie.srt", "Movie.pt-BR.srt"),
        ("Movie.en.forced.srt", "Movie.pt-BR.forced.srt"),
        ("Movie.eng.sdh.srt", "Movie.pt-BR.sdh.srt"),
        ("Movie.2024.en.srt", "Movie.2024.pt-BR.srt"),
        ("Movie.EN.FORCED.SRT", "Movie.pt-BR.FORCED.srt"),
        ("Movie.ENG.SRT", "Movie.pt-BR.srt"),
    ],
)
def test_default_output_path(input_name: str, expected_output: str) -> None:
    assert default_output_path(Path(input_name), "pt-BR") == Path(expected_output)


def test_parses_unlabeled_file() -> None:
    parsed = parse_subtitle_name(Path("Movie.srt"))
    assert parsed.movie_name == "Movie"
    assert parsed.language_tag is None
    assert parsed.attributes == ()
    assert parsed.is_unlabeled
    assert not parsed.is_source_language


def test_parses_english_source() -> None:
    parsed = parse_subtitle_name(Path("Movie.eng.srt"))
    assert parsed.movie_name == "Movie"
    assert parsed.language_tag == "eng"
    assert parsed.is_source_language
    assert not parsed.is_target_language


def test_parses_brazilian_portuguese() -> None:
    for name in ("Movie.pt-BR.srt", "Movie.pt_BR.srt", "Movie.pob.srt"):
        parsed = parse_subtitle_name(Path(name))
        assert parsed.is_target_language
        assert parsed.is_brazilian_portuguese
        assert not parsed.is_generic_portuguese


def test_parses_generic_portuguese() -> None:
    for name in ("Movie.pt.srt", "Movie.por.srt"):
        parsed = parse_subtitle_name(Path(name))
        assert parsed.is_target_language
        assert parsed.is_generic_portuguese
        assert not parsed.is_brazilian_portuguese


def test_preserves_attributes() -> None:
    parsed = parse_subtitle_name(Path("Movie.en.forced.srt"))
    assert parsed.attributes == ("forced",)
    assert parsed.has_specialized_variant


def test_preserves_multiple_attributes() -> None:
    parsed = parse_subtitle_name(Path("Movie.eng.sdh.forced.srt"))
    assert parsed.attributes == ("sdh", "forced")
    assert parsed.output_path("pt-BR") == Path("Movie.pt-BR.sdh.forced.srt")


def test_unknown_token_after_language_becomes_movie_name() -> None:
    # Right-anchored parsing keeps unknown trailing tokens as part of the title
    # so two different sources cannot collide on the same output.
    parsed = parse_subtitle_name(Path("Movie.en.foobar.srt"))
    assert parsed.output_path("pt-BR") == Path("Movie.en.foobar.pt-BR.srt")


def test_different_unknown_suffixes_do_not_collide() -> None:
    one = parse_subtitle_name(Path("Movie.en.1.srt"))
    two = parse_subtitle_name(Path("Movie.en.2.srt"))
    assert one.output_path("pt-BR") != two.output_path("pt-BR")


def test_movie_name_with_periods_is_preserved() -> None:
    parsed = parse_subtitle_name(Path("My.Movie.2024.en.srt"))
    assert parsed.movie_name == "My.Movie.2024"
    assert parsed.output_path("pt-BR") == Path("My.Movie.2024.pt-BR.srt")


def test_explicit_output_path_takes_precedence_in_cli(
    tmp_path: Path,
) -> None:
    from subtranslate.cli import app
    from subtranslate.translator import TranslationStats

    srt = tmp_path / "Movie.en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    explicit = tmp_path / "custom.pt-BR.srt"

    def fake_translate(input_path, output_path, settings, **kwargs):
        assert output_path == explicit
        return TranslationStats(total_entries=1, batch_count=1)

    runner = CliRunner()
    with patch("subtranslate.cli.translate", fake_translate):
        result = runner.invoke(
            app, ["translate", str(srt), "--output", str(explicit)]
        )
    assert result.exit_code == 0


def test_existing_target_file_same_path_is_detected() -> None:
    parsed = parse_subtitle_name(Path("Movie.pt-BR.srt"))
    assert parsed.is_target_language
    assert parsed.output_path("pt-BR") == Path("Movie.pt-BR.srt")


@pytest.mark.parametrize(
    ("input_name", "expected_output"),
    [
        # Title words that look like tags must stay in the movie name.
        ("Hi.Mom.2021.en.srt", "Hi.Mom.2021.pt-BR.srt"),
        ("En.Garde.1990.srt", "En.Garde.1990.pt-BR.srt"),
        ("Por.Favor.eng.srt", "Por.Favor.pt-BR.srt"),
        ("The.Eng.Movie.en.srt", "The.Eng.Movie.pt-BR.srt"),
        ("en.srt", "en.pt-BR.srt"),
    ],
)
def test_title_words_looking_like_tags_are_preserved(
    input_name: str, expected_output: str
) -> None:
    assert default_output_path(Path(input_name), "pt-BR") == Path(expected_output)


def test_target_language_tag_is_sanitized() -> None:
    assert default_output_path(Path("Movie.en.srt"), "pt/BR") == Path("Movie.pt-BR.srt")
    assert default_output_path(Path("Movie.en.srt"), "pt\\BR") == Path("Movie.pt-BR.srt")
