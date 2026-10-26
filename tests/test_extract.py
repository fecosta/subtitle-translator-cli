from __future__ import annotations

import hashlib
import json
import os
import shutil
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

# Captured before any fixture can patch subtranslate.coverage.subprocess.run,
# which is the same object as the global subprocess.run. Real-ffmpeg tests must
# call this reference explicitly, otherwise they would invoke a stubbed runner.
_REAL_RUN = subprocess.run

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

    def test_skips_when_external_english_exists(
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

    def test_skips_case_insensitive_external_english_collision(
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

    def test_skips_second_same_stem_video_when_output_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        mkv = library / "Movie" / "Movie.mkv"
        mp4 = library / "Movie" / "Movie.mp4"
        _touch_video(mkv)
        _touch_video(mp4)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )
        ffmpeg_calls: list[list[str]] = []

        def capture_ffmpeg(args, **_kwargs):
            ffmpeg_calls.append(list(args))
            output_path = Path(args[-1])
            output_path.write_text(VALID_SRT, encoding="utf-8")

            class _Result:
                pass

            result = _Result()
            result.returncode = 0
            result.stderr = ""
            return result

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", capture_ffmpeg)

        result = extract_embedded(library)

        assert result.performed == 1
        assert result.skipped == 1
        assert (
            result.skipped_reasons[ExtractionSkipReason.OUTPUT_EXISTS] == 1
        )
        assert len(ffmpeg_calls) == 1
        assert "Movie.mkv" in ffmpeg_calls[0][ffmpeg_calls[0].index("-i") + 1]

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


class TestSingleFileMode:
    def test_single_video_file_only_probes_target_video(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        target = library / "Movie" / "Movie.mkv"
        sibling = library / "Movie" / "Sibling.mkv"
        nested = library / "Movie" / "sub" / "Nested.mkv"
        _touch_video(target)
        _touch_video(sibling)
        _touch_video(nested)
        probe_calls: list[Path] = []

        def fake_run(args, **_kwargs):
            if args[0] == "ffprobe":
                probe_calls.append(Path(args[-1]))
                stdout = _ffprobe_json(
                    [_stream(0, 0, codec="subrip", language="eng")]
                )
            else:
                output_path = Path(args[-1])
                output_path.write_text(VALID_SRT, encoding="utf-8")
                stdout = ""

            class _Result:
                pass

            result = _Result()
            result.stdout = stdout
            result.stderr = ""
            result.returncode = 0
            return result

        monkeypatch.setattr("subtranslate.coverage.subprocess.run", fake_run)
        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", fake_run)

        result = extract_embedded(target)

        assert result.movies_scanned == 1
        assert result.performed == 1
        assert probe_calls == [target.resolve()]
        assert not (sibling.parent / "Sibling.en.srt").exists()
        assert not (nested.parent / "Nested.en.srt").exists()

    def test_single_video_file_skips_with_same_dir_external_english(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "Movie" / "Movie.mkv"
        _touch_video(target)
        _write_srt(target.parent / "Movie.en.srt", "existing")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(target)

        assert result.performed == 0
        assert result.skipped == 1
        assert (
            result.skipped_reasons[ExtractionSkipReason.HAS_EXTERNAL_ENGLISH]
            == 1
        )

    def test_single_video_file_unsupported_extension_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "Movie" / "Movie.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not a video", encoding="utf-8")
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

        with pytest.raises(ValueError, match="Unsupported video extension"):
            extract_embedded(target)

        assert len(ffmpeg_calls) == 0

        cli_result = runner.invoke(app, ["extract-embedded", str(target)])
        assert cli_result.exit_code == 2
        assert "Unsupported video extension" in cli_result.output

    def test_single_video_file_uppercase_extension_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "Movie" / "Movie.MKV"
        _touch_video(target)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = extract_embedded(target)

        assert result.performed == 1
        assert (target.parent / "Movie.en.srt").exists()

    def test_single_video_file_nonexistent_path_errors(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "Movie" / "Movie.mkv"

        result = runner.invoke(app, ["extract-embedded", str(target)])
        assert result.exit_code == 2

    def test_single_video_file_require_mount_rejects_unmounted(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "Movie" / "Movie.mkv"
        _touch_video(target)

        result = runner.invoke(
            app, ["extract-embedded", str(target), "--require-mount"]
        )
        assert result.exit_code == 2
        assert "not a mounted filesystem" in result.output

    def test_single_video_file_mount_point_rejects_wrong_mount(
        self, tmp_path: Path
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        target = mount_point / "Movie" / "Movie.mkv"
        _touch_video(target)

        result = runner.invoke(
            app,
            [
                "extract-embedded",
                str(target),
                "--mount-point",
                str(mount_point),
            ],
        )
        assert result.exit_code == 2
        assert "not under the required mount point" in result.output


class TestFfmpegArgv:
    def test_ffmpeg_argv_for_preferred_nonzero_stream_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="ass", language="eng"),
                    _stream(1, 1, codec="subrip", language="eng"),
                ]
            ),
        )
        calls: list[list[str]] = []

        def capture_ffmpeg(args, **_kwargs):
            calls.append(list(args))
            output_path = Path(args[-1])
            output_path.write_text(VALID_SRT, encoding="utf-8")

            class _Result:
                pass

            result = _Result()
            result.returncode = 0
            result.stderr = ""
            return result

        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", capture_ffmpeg)

        extract_embedded(library)

        assert len(calls) == 1
        argv = calls[0]
        assert argv[0] == "ffmpeg"
        assert "-nostdin" in argv
        assert "-i" in argv
        assert str(video.resolve()) in argv
        assert "-map" in argv
        assert "0:s:1" in argv
        assert "-c:s" in argv
        assert "srt" in argv
        output_arg = Path(argv[-1])
        assert output_arg.suffix == ".srt"
        assert output_arg != (video.parent / "Movie.en.srt").resolve()
        assert output_arg.parent == video.parent
        assert "-y" not in argv
        assert "copy" not in argv
        assert "remux" not in argv


class TestFfmpegOverwriteSemantics:
    def test_extract_honors_ffmpeg_no_clobber_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for R1-1: ffmpeg -n must not see a pre-existing temp."""
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        def fake_ffmpeg_honors_n(args, **_kwargs):
            output_path = Path(args[-1])
            has_n = "-n" in args
            already_exists = output_path.exists()

            class _Result:
                pass

            result = _Result()
            if has_n and already_exists:
                result.returncode = 1
                result.stderr = (
                    f"File {output_path} already exists. Exiting."
                )
            else:
                output_path.write_text(VALID_SRT, encoding="utf-8")
                result.returncode = 0
                result.stderr = ""
            return result

        monkeypatch.setattr(
            "subtranslate.extract._run_ffmpeg", fake_ffmpeg_honors_n
        )

        result = extract_embedded(library)

        assert result.performed == 1
        assert result.failures == 0
        assert (video.parent / "Movie.en.srt").exists()


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


def test_real_run_reference_is_unpatched_subprocess_run() -> None:
    """The real-ffmpeg integration test must not accidentally use a stub runner."""
    assert _REAL_RUN is not None
    assert _REAL_RUN.__module__ == "subprocess"
    assert _REAL_RUN.__name__ == "run"


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg/ffprobe not available",
)
class TestRealFfmpeg:
    def test_real_ffmpeg_extraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real-ffmpeg validation: mux, extract, idempotency, safety."""
        monkeypatch.setattr("subtranslate.coverage.subprocess.run", _REAL_RUN)
        monkeypatch.setattr("subtranslate.extract._run_ffmpeg", _REAL_RUN)

        movie_dir = tmp_path / "Movie"
        movie_dir.mkdir()
        sub_path = movie_dir / "subs.srt"
        mkv_path = movie_dir / "Movie.mkv"
        sub_path.write_text(VALID_SRT, encoding="utf-8")

        _REAL_RUN(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=1",
                "-f",
                "srt",
                "-i",
                str(sub_path),
                "-map",
                "0:v",
                "-map",
                "1:s",
                "-c:v",
                "mpeg4",
                "-c:s",
                "srt",
                "-metadata:s:s:0",
                "language=eng",
                "-t",
                "1",
                str(mkv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert mkv_path.exists()

        # The fixture must really contain one English SubRip stream, otherwise
        # the extraction assertions below would pass for the wrong reason.
        probe = _REAL_RUN(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=codec_name:stream_tags=language",
                "-of",
                "json",
                str(mkv_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout)["streams"]
        assert len(streams) == 1
        assert streams[0]["codec_name"] == "subrip"
        assert streams[0]["tags"]["language"] == "eng"

        original_sha = hashlib.sha256(mkv_path.read_bytes()).hexdigest()
        original_mtime = mkv_path.stat().st_mtime

        result = extract_embedded(mkv_path)
        assert result.performed == 1
        assert result.failures == 0

        output_path = movie_dir / "Movie.en.srt"
        assert output_path.exists()
        assert "Hello" in output_path.read_text(encoding="utf-8")
        assert (
            hashlib.sha256(mkv_path.read_bytes()).hexdigest() == original_sha
        )
        assert mkv_path.stat().st_mtime == original_mtime

        # No temp leftovers: the private temp files are hidden and named
        # ".Movie.en.extract-*.srt".
        listing = [f.name for f in movie_dir.iterdir()]
        assert not [name for name in listing if name.startswith(".")]
        assert not [name for name in listing if ".extract-" in name]

        second = extract_embedded(mkv_path)
        assert second.performed == 0
        assert second.skipped == 1
        assert (
            second.skipped_reasons[ExtractionSkipReason.HAS_EXTERNAL_ENGLISH]
            == 1
        )

        output_path.write_text("do not overwrite", encoding="utf-8")
        third = extract_embedded(mkv_path)
        assert third.performed == 0
        assert third.skipped == 1
        assert output_path.read_text(encoding="utf-8") == "do not overwrite"

        output_path.unlink()
        dry = extract_embedded(mkv_path, dry_run=True)
        assert dry.performed == 1
        assert not output_path.exists()

        output_path.unlink(missing_ok=True)
        cli_result = runner.invoke(
            app, ["extract-embedded", str(mkv_path)]
        )
        assert cli_result.exit_code == 0
        assert output_path.exists()
        assert "Hello" in output_path.read_text(encoding="utf-8")
