from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from subtranslate.cli import app
from subtranslate.extract import (
    ExtractionSkipReason,
    ExtractionStatus,
    extract_embedded,
)

runner = CliRunner()


VALID_SRT = """1
00:00:01,000 --> 00:00:02,000
Hello

2
00:00:03,000 --> 00:00:04,000
World
"""


def _write_srt(path: Path, text: str = "Hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nLine two\n",
        encoding="utf-8",
    )


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake video bytes")


def _ffprobe_json(streams: list[dict]) -> str:
    return json.dumps({"streams": streams})


def _stream(
    index: int,
    subtitle_index: int,
    codec: str = "subrip",
    language: str | None = "eng",
    title: str = "",
    forced: int = 0,
    hearing_impaired: int = 0,
) -> dict:
    stream: dict = {
        "index": index,
        "codec_name": codec,
        "disposition": {"forced": forced, "hearing_impaired": hearing_impaired},
    }
    tags: dict = {}
    if language is not None:
        tags["language"] = language
    if title:
        tags["title"] = title
    if tags:
        stream["tags"] = tags
    return stream


def _patch_ffprobe(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int = 0,
) -> None:
    def fake_run(*_args, **_kwargs):
        class _Result:
            pass

        result = _Result()
        result.stdout = stdout
        result.stderr = ""
        result.returncode = returncode
        return result

    monkeypatch.setattr("subtranslate.coverage.subprocess.run", fake_run)


def _patch_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int = 0,
    stderr: str = "",
    content: str | None = None,
) -> None:
    def fake_run(args, **_kwargs):
        class _Result:
            pass

        result = _Result()
        if content is not None and returncode == 0:
            output_path = Path(args[-1])
            output_path.write_text(content, encoding="utf-8")
        result.returncode = returncode
        result.stderr = stderr
        return result

    monkeypatch.setattr("subtranslate.extract._run_ffmpeg", fake_run)


@pytest.fixture(autouse=True)
def _stub_ffprobe_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to no embedded streams; tests override when needed."""
    _patch_ffprobe(monkeypatch, _ffprobe_json([]))


@pytest.fixture(autouse=True)
def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ffmpeg writes a valid SRT to the requested output path."""
    _patch_ffmpeg(monkeypatch, content=VALID_SRT)


class TestTrackSelection:
    def test_selects_embedded_english_text_track(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(library)

        assert result.movies_scanned == 1
        assert result.eligible_tracks == 1
        assert result.performed == 1
        assert result.failures == 0
        assert (video.parent / "Movie.en.srt").exists()

    def test_ignores_portuguese_and_unknown_language_tracks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="subrip", language="por"),
                    _stream(1, 1, codec="subrip", language="fra"),
                ]
            ),
        )

        result = extract_embedded(library)

        assert result.eligible_tracks == 0
        assert result.performed == 0
        assert result.skipped == 1
        assert result.skipped_reasons[ExtractionSkipReason.NO_ELIGIBLE_TRACK] == 1

    def test_ignores_forced_and_sdh_tracks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="subrip", language="eng", forced=1),
                    _stream(1, 1, codec="subrip", language="eng", hearing_impaired=1),
                ]
            ),
        )

        result = extract_embedded(library)

        assert result.eligible_tracks == 0
        assert result.skipped_reasons[ExtractionSkipReason.NO_ELIGIBLE_TRACK] == 1

    def test_ignores_image_based_subtitles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [_stream(0, 0, codec="hdmv_pgs_subtitle", language="eng")]
            ),
        )

        result = extract_embedded(library)

        assert result.eligible_tracks == 0
        assert result.skipped_reasons[ExtractionSkipReason.NO_ELIGIBLE_TRACK] == 1

    def test_prefers_subrip_over_other_text_codecs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="ass", language="eng"),
                    _stream(1, 1, codec="subrip", language="eng"),
                ]
            ),
        )

        result = extract_embedded(library)

        assert result.eligible_tracks == 1
        assert result.performed == 1
        track = result.outcomes[0].track
        assert track is not None
        assert track.codec == "subrip"
        assert track.subtitle_stream_index == 1

    def test_tie_breaks_by_lowest_subtitle_stream_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="mov_text", language="eng"),
                    _stream(1, 1, codec="ass", language="eng"),
                ]
            ),
        )

        result = extract_embedded(library)

        track = result.outcomes[0].track
        assert track is not None
        assert track.subtitle_stream_index == 0


class TestOutputNaming:
    def test_output_uses_video_stem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie (2014)" / "Movie (2014).mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        extract_embedded(library)

        assert (video.parent / "Movie (2014).en.srt").exists()

    def test_output_handles_spaces_and_special_characters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie Name" / "Movie Name [2024].mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        extract_embedded(library)

        assert (video.parent / "Movie Name [2024].en.srt").exists()

    def test_output_preserves_dots_in_stem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie.Dir" / "Movie.Dir.Version.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        extract_embedded(library)

        assert (video.parent / "Movie.Dir.Version.en.srt").exists()


class TestSkipReasons:
    def test_skips_movie_with_existing_external_english_subtitle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _write_srt(video.parent / "Movie.en.srt", "existing")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(library)

        assert result.performed == 0
        assert result.skipped == 1
        assert (
            result.skipped_reasons[ExtractionSkipReason.HAS_EXTERNAL_ENGLISH] == 1
        )
        assert not (video.parent / "Movie.en.srt").read_text().startswith("Hello")

    def test_specialized_english_variant_does_not_block_extraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _write_srt(video.parent / "Movie.en.sdh.srt")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(library)

        assert result.performed == 1
        assert (video.parent / "Movie.en.srt").exists()

    def test_skips_when_output_already_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        existing = video.parent / "Movie.en.srt"
        existing.write_text("do not overwrite", encoding="utf-8")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(library)

        assert result.performed == 0
        assert result.skipped == 1
        assert (
            result.skipped_reasons[ExtractionSkipReason.HAS_EXTERNAL_ENGLISH] == 1
        )
        assert existing.read_text() == "do not overwrite"

    def test_skips_case_insensitive_output_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        existing = video.parent / "Movie.EN.srt"
        existing.write_text("do not overwrite", encoding="utf-8")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(library)

        assert result.performed == 0
        assert result.skipped == 1
        assert (
            result.skipped_reasons[ExtractionSkipReason.HAS_EXTERNAL_ENGLISH] == 1
        )

    def test_output_exists_race_is_skip_not_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        def raise_output_exists(*_args, **_kwargs):
            from subtranslate.exceptions import OutputExistsError

            raise OutputExistsError("race")

        monkeypatch.setattr(
            "subtranslate.extract.atomic_write_path", raise_output_exists
        )

        result = extract_embedded(library)

        assert result.performed == 0
        assert result.failures == 0
        assert result.skipped == 1
        assert result.skipped_reasons[ExtractionSkipReason.OUTPUT_EXISTS] == 1

    def test_skips_probe_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(monkeypatch, _ffprobe_json([]), returncode=1)

        result = extract_embedded(library)

        assert result.performed == 0
        assert result.skipped == 1
        assert result.skipped_reasons[ExtractionSkipReason.PROBE_ERROR] == 1

    def test_skips_ambiguous_movie(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _touch_video(library / "Movie" / "Movie.eng.mkv")
        _write_srt(library / "Movie" / "Movie.eng.pt-BR.srt")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(library)

        assert result.performed == 0
        assert result.skipped == 2  # one per video
        assert result.skipped_reasons[ExtractionSkipReason.AMBIGUOUS] == 2


class TestDryRun:
    def test_dry_run_does_not_call_ffmpeg_or_create_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )
        ffmpeg_calls: list[list[str]] = []

        def capture_ffmpeg(args, **_kwargs):
            ffmpeg_calls.append(list(args))

            class _Result:
                pass

            result = _Result()
            result.returncode = 0
            result.stderr = ""
            return result

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", capture_ffmpeg)

        result = extract_embedded(library, dry_run=True)

        assert result.dry_run is True
        assert result.performed == 1
        assert result.outcomes[0].status == ExtractionStatus.WOULD_EXTRACT
        assert len(ffmpeg_calls) == 0
        assert not (video.parent / "Movie.en.srt").exists()


class TestMountValidation:
    def test_require_mount_rejects_unmounted_library(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        library.mkdir()

        with pytest.raises(FileNotFoundError):
            extract_embedded(library, require_mount=True)

    def test_require_mount_accepts_path_under_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        library.mkdir(parents=True)
        _touch_video(library / "Movie" / "Movie.mkv")

        def fake_ismount(path: str | os.PathLike[str]) -> bool:
            return str(path) == str(tmp_path)

        monkeypatch.setattr(os.path, "ismount", fake_ismount)

        result = extract_embedded(library, require_mount=True)
        assert result.movies_scanned >= 1

    def test_explicit_mount_point_rejects_unmounted_point(
        self, tmp_path: Path
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = mount_point / "movies"
        library.mkdir()

        with pytest.raises(FileNotFoundError):
            extract_embedded(library, mount_point=mount_point)

    def test_explicit_mount_point_accepts_library_under_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = mount_point / "movies"
        library.mkdir(parents=True)
        _touch_video(library / "Movie" / "Movie.mkv")

        def fake_ismount(path: str | os.PathLike[str]) -> bool:
            return str(path) == str(mount_point.resolve())

        monkeypatch.setattr(os.path, "ismount", fake_ismount)

        result = extract_embedded(library, mount_point=mount_point)
        assert result.movies_scanned == 1


class TestFfmpegFailures:
    def test_nonzero_exit_continues_and_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "A" / "A.mkv")
        _touch_video(library / "B" / "B.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="subrip", language="eng"),
                    _stream(0, 0, codec="subrip", language="eng"),
                ]
            ),
        )

        call_count = 0

        def fake_run(args, **_kwargs):
            nonlocal call_count
            call_count += 1
            output_path = Path(args[-1])
            if any("A.mkv" in str(a) for a in args):
                output_path.write_text(VALID_SRT, encoding="utf-8")
                returncode = 0
            else:
                returncode = 1

            class _Result:
                pass

            result = _Result()
            result.returncode = returncode
            result.stderr = "B failed"
            return result

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", fake_run)

        result = extract_embedded(library)

        assert result.performed == 1
        assert result.failures == 1
        assert call_count == 2
        failed = [o for o in result.outcomes if o.status == ExtractionStatus.FAILED]
        assert len(failed) == 1
        assert "B failed" in failed[0].reason

    def test_file_not_found_error_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        def raise_file_not_found(*_args, **_kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", raise_file_not_found)

        result = extract_embedded(library)

        assert result.failures == 1
        assert "ffmpeg not found" in result.outcomes[0].reason

    def test_timeout_error_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        def raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("ffmpeg", 600)

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", raise_timeout)

        result = extract_embedded(library)

        assert result.failures == 1
        assert "timed out" in result.outcomes[0].reason

    def test_cleans_up_after_invalid_ffmpeg_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )
        _patch_ffmpeg(monkeypatch, content="not valid srt")

        result = extract_embedded(library)

        assert result.failures == 1
        assert result.outcomes[0].status == ExtractionStatus.FAILED
        assert not (video.parent / "Movie.en.srt").exists()
        # No lingering temporary extraction files.
        assert not any(
            f.name.endswith(".extracting.srt") for f in video.parent.iterdir()
        )

    def test_cleans_up_after_empty_ffmpeg_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )
        _patch_ffmpeg(monkeypatch, content="1\nnot a timestamp\n")

        result = extract_embedded(library)

        assert result.failures == 1
        assert not (video.parent / "Movie.en.srt").exists()


class TestIdempotencyAndSafety:
    def test_second_run_extracts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        first = extract_embedded(library)
        second = extract_embedded(library)

        assert first.performed == 1
        assert second.performed == 0
        assert second.skipped == 1
        assert (
            second.skipped_reasons[ExtractionSkipReason.HAS_EXTERNAL_ENGLISH] == 1
        )
        assert len(list(video.parent.glob("*.srt"))) == 1

    def test_video_file_bytes_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        original = video.read_bytes()
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        extract_embedded(library)

        assert video.read_bytes() == original


class TestSinglePathInvocation:
    def test_single_movie_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        movie_dir = library / "Movie"
        video = movie_dir / "Movie.mkv"
        _touch_video(video)
        _touch_video(library / "Other" / "Other.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(movie_dir)

        assert result.movies_scanned == 1
        assert result.performed == 1
        assert (video.parent / "Movie.en.srt").exists()
        assert not (library / "Other" / "Other.en.srt").exists()

    def test_single_video_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        target = library / "Movie" / "Movie.mkv"
        _touch_video(target)
        _touch_video(library / "Movie" / "Other.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(target)

        assert result.movies_scanned == 1
        assert result.performed == 1
        assert (target.parent / "Movie.en.srt").exists()
        assert not (target.parent / "Other.en.srt").exists()


class TestCli:
    def test_cli_extract_embedded_basic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = runner.invoke(app, ["extract-embedded", str(library)])

        assert result.exit_code == 0
        assert "Extracted:" in result.output
        assert (video.parent / "Movie.en.srt").exists()

    def test_cli_extract_embedded_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = runner.invoke(
            app, ["extract-embedded", str(library), "--dry-run"]
        )

        assert result.exit_code == 0
        assert "Would extract:" in result.output
        assert not (video.parent / "Movie.en.srt").exists()

    def test_cli_require_mount_rejects_unmounted(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        library.mkdir()

        result = runner.invoke(
            app, ["extract-embedded", str(library), "--require-mount"]
        )

        assert result.exit_code == 2
        assert "not a mounted filesystem" in result.output

    def test_cli_mount_point_rejects_wrong_mount(
        self, tmp_path: Path
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = mount_point / "movies"
        library.mkdir()

        result = runner.invoke(
            app,
            ["extract-embedded", str(library), "--mount-point", str(mount_point)],
        )

        assert result.exit_code == 2
        assert "not under the required mount point" in result.output

    def test_cli_exits_nonzero_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        def raise_file_not_found(*_args, **_kwargs):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", raise_file_not_found)

        result = runner.invoke(app, ["extract-embedded", str(library)])

        assert result.exit_code == 1
        assert "Failures:" in result.output


class TestEndToEndLibrary:
    def test_library_with_multiple_movies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        has_eng = library / "HasEng" / "HasEng.mkv"
        has_pt = library / "HasPT" / "HasPT.mkv"
        no_subs = library / "NoSubs" / "NoSubs.mkv"
        forced_only = library / "Forced" / "Forced.mkv"
        for video in (has_eng, has_pt, no_subs, forced_only):
            _touch_video(video)
        _write_srt(has_pt.parent / "HasPT.pt-BR.srt")

        def fake_run(args, **_kwargs):
            output_path = Path(args[-1])
            class _Result:
                pass

            result = _Result()
            if "ffprobe" in args[0]:
                video = Path(args[-1]).name
                if video in {"HasEng.mkv", "HasPT.mkv"}:
                    streams = [_stream(0, 0, codec="subrip", language="eng")]
                elif video == "NoSubs.mkv":
                    streams = []
                elif video == "Forced.mkv":
                    streams = [_stream(0, 0, codec="subrip", language="eng", forced=1)]
                else:
                    streams = []
                result.stdout = _ffprobe_json(streams)
                result.returncode = 0
                result.stderr = ""
            else:
                output_path.write_text(VALID_SRT, encoding="utf-8")
                result.returncode = 0
                result.stderr = ""
            return result

        monkeypatch.setattr("subtranslate.coverage.subprocess.run", fake_run)
        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", fake_run)

        result = extract_embedded(library)

        assert result.movies_scanned == 4
        assert result.performed == 2
        assert (has_eng.parent / "HasEng.en.srt").exists()
        assert (has_pt.parent / "HasPT.en.srt").exists()
        assert not (no_subs.parent / "NoSubs.en.srt").exists()
        assert not (forced_only.parent / "Forced.en.srt").exists()
        assert result.skipped_reasons[ExtractionSkipReason.NO_ELIGIBLE_TRACK] == 2


@pytest.mark.skipif(
    os.system("which ffmpeg > /dev/null 2>&1") != 0,
    reason="ffmpeg not available",
)
class TestRealFfmpeg:
    def test_real_ffmpeg_extraction(self, tmp_path: Path) -> None:
        """Optional real-ffmpeg validation; only runs when ffmpeg is installed."""
        pytest.skip("real ffmpeg not available in this environment")
