from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from subtranslate.cli import app
from subtranslate.coverage import (
    EMBEDDED_CODEC_IMAGE,
    EMBEDDED_CODEC_TEXT,
    EMBEDDED_CODEC_UNKNOWN,
    VIDEO_EXTENSIONS,
    CoverageStatus,
    classify_embedded_codec,
    coverage_result_to_dict,
    normalize_embedded_language,
    scan_coverage,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _stub_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests do not depend on ffprobe being installed locally."""

    def fake_run(*args, **kwargs):
        class _Result:
            pass

        result = _Result()
        result.stdout = json.dumps({"streams": []})
        result.stderr = ""
        result.returncode = 0
        return result

    monkeypatch.setattr("subtranslate.coverage.subprocess.run", fake_run)


def _write_srt(path: Path, text: str = "Hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nLine two\n",
        encoding="utf-8",
    )


def _touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class TestConstants:
    def test_video_extensions_are_configurable_constants(self) -> None:
        expected = {".mkv", ".mp4", ".avi", ".m4v"}
        assert expected == VIDEO_EXTENSIONS
        assert all(ext.startswith(".") for ext in VIDEO_EXTENSIONS)


class TestMovieDiscoveryAndClassification:
    def test_coverage_has_brazilian_portuguese(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.srt")

        result = scan_coverage(library)

        assert len(result.movies) == 1
        assert result.movies[0].category == CoverageStatus.HAS_PT_BR
        assert result.summary[CoverageStatus.HAS_PT_BR] == 1

    def test_coverage_has_generic_portuguese(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt.srt")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.HAS_GENERIC_PT

    def test_coverage_has_english_only(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.en.srt")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.HAS_EN_ONLY

    def test_coverage_has_unlabeled_subtitle(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.srt")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.HAS_UNLABELED

    def test_coverage_no_external_subtitles(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.NO_EXTERNAL_SUBS

    def test_coverage_embedded_only_is_no_external_subs(self, tmp_path: Path) -> None:
        # The scanner inspects only external SRT files, not media containers.
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.NO_EXTERNAL_SUBS
        assert len(result.movies[0].subtitles) == 0

    def test_coverage_multiple_videos_in_one_directory(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "A" / "A.mkv")
        _touch_video(library / "A" / "B.mkv")
        _write_srt(library / "A" / "A.en.srt")
        _write_srt(library / "A" / "B.pt-BR.srt")

        result = scan_coverage(library)

        categories = {m.movie_name: m.category for m in result.movies}
        assert categories["A"] == CoverageStatus.HAS_EN_ONLY
        assert categories["B"] == CoverageStatus.HAS_PT_BR

    def test_coverage_legacy_eng_pt_br_naming(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.eng.srt")
        _write_srt(library / "Movie" / "Movie.eng.pt-BR.srt")

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_PT_BR
        tags = {sub.language_tag: sub.is_brazilian_portuguese for sub in movie.subtitles}
        assert tags.get("eng") is False
        assert tags.get("pt-BR") is True

    def test_coverage_specialized_variants_do_not_count_as_full_coverage(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.forced.srt")
        _write_srt(library / "Movie" / "Movie.en.sdh.srt")

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_SPECIALIZED_ONLY
        assert len(movie.subtitles) == 2
        assert all(sub.is_specialized_variant for sub in movie.subtitles)

    def test_coverage_case_insensitive_extensions_and_tags(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.MP4")
        _write_srt(library / "Movie" / "Movie.PT-BR.SRT")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.HAS_PT_BR

    def test_coverage_preserves_full_coverage_with_variant(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.srt")
        _write_srt(library / "Movie" / "Movie.pt-BR.forced.srt")

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_PT_BR
        assert len(movie.subtitles) == 2

    @pytest.mark.parametrize("attribute", ["forced", "sdh", "hi", "cc"])
    def test_coverage_individual_specialized_attributes(
        self, tmp_path: Path, attribute: str
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / f"Movie.pt-BR.{attribute}.srt")

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_SPECIALIZED_ONLY
        assert movie.subtitles[0].is_specialized_variant

    def test_coverage_case_insensitive_stem_matching(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.MKV")
        _write_srt(library / "Movie" / "movie.pt-BR.srt")

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.HAS_PT_BR


class TestEdgeCasesAndSafety:
    def test_coverage_ignores_video_symlinks(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        real_video = tmp_path / "outside.mkv"
        _touch_video(real_video)
        movie_dir = library / "Movie"
        movie_dir.mkdir(parents=True)
        link = movie_dir / "Movie.mkv"
        link.symlink_to(real_video)

        result = scan_coverage(library)

        assert len(result.movies) == 0

    def test_coverage_ignores_subtitle_symlinks(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        real_srt = tmp_path / "outside.srt"
        _write_srt(real_srt)
        link = library / "Movie" / "Movie.pt-BR.srt"
        link.symlink_to(real_srt)

        result = scan_coverage(library)

        assert result.movies[0].category == CoverageStatus.NO_EXTERNAL_SUBS

    def test_coverage_reports_unreadable_directory(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        inaccessible = library / "Movie"
        inaccessible.mkdir(parents=True)
        _touch_video(inaccessible / "Movie.mkv")
        try:
            os.chmod(inaccessible, 0o000)
            result = scan_coverage(library)
        finally:
            os.chmod(inaccessible, 0o755)

        assert len(result.errors) >= 1

    def test_coverage_rejects_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError):
            scan_coverage(missing)

    def test_coverage_rejects_symlinked_root(self, tmp_path: Path) -> None:
        real_library = tmp_path / "real_library"
        _touch_video(real_library / "Movie" / "Movie.mkv")
        link = tmp_path / "link_library"
        link.symlink_to(real_library)

        with pytest.raises(ValueError):
            scan_coverage(link)

    def test_coverage_require_mount_rejects_unmounted(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        library.mkdir()

        with pytest.raises(FileNotFoundError):
            scan_coverage(library, require_mount=True)

    def test_coverage_require_mount_accepts_path_under_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        library.mkdir(parents=True)

        def fake_ismount(path: str | os.PathLike[str]) -> bool:
            return str(path) == str(tmp_path)

        monkeypatch.setattr(os.path, "ismount", fake_ismount)

        result = scan_coverage(library, require_mount=True)
        assert result.directories_scanned >= 1

    def test_coverage_ambiguous_subtitle_matches_multiple_videos(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _touch_video(library / "Movie" / "Movie.eng.mkv")
        _write_srt(library / "Movie" / "Movie.eng.pt-BR.srt")

        result = scan_coverage(library)

        categories = {m.movie_name: m.category for m in result.movies}
        assert all(c == CoverageStatus.AMBIGUOUS for c in categories.values())
        assert any(
            "matches multiple videos" in reason
            for m in result.movies
            for reason in m.ambiguous_reasons
        )


class TestResultSerialization:
    def test_coverage_json_schema(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.srt")

        result = scan_coverage(library)
        payload = coverage_result_to_dict(result, library)

        assert payload["library_path"] == str(library.resolve())
        assert isinstance(payload["directories_scanned"], int)
        assert isinstance(payload["movies_discovered"], int)
        assert "summary" in payload
        assert set(payload["summary"]) == {
            CoverageStatus.HAS_PT_BR,
            CoverageStatus.HAS_GENERIC_PT,
            CoverageStatus.HAS_EN_ONLY,
            CoverageStatus.HAS_UNLABELED,
            CoverageStatus.HAS_SPECIALIZED_ONLY,
            CoverageStatus.NO_EXTERNAL_SUBS,
            CoverageStatus.AMBIGUOUS,
        }
        assert len(payload["movies"]) == 1
        movie = payload["movies"][0]
        assert set(movie) == {
            "path",
            "movie_name",
            "category",
            "video_files",
            "subtitles",
            "embedded_subtitles",
            "probe_error",
            "ambiguous_reasons",
        }
        assert "embedded_summary" in payload
        assert movie["category"] == CoverageStatus.HAS_PT_BR
        assert movie["probe_error"] is not None or movie["embedded_subtitles"] == []

    def test_coverage_json_is_deterministic(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "B" / "B.mkv")
        _touch_video(library / "A" / "A.mkv")
        _write_srt(library / "A" / "A.pt-BR.srt")
        _write_srt(library / "B" / "B.en.srt")

        result = scan_coverage(library)
        first = json.dumps(coverage_result_to_dict(result, library), sort_keys=True)
        second_result = scan_coverage(library)
        second = json.dumps(
            coverage_result_to_dict(second_result, library), sort_keys=True
        )

        assert first == second


class TestCliCoverage:
    def test_cli_coverage_default_output(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.srt")

        result = runner.invoke(app, ["coverage", str(library)])

        assert result.exit_code == 0
        assert "HAS_PT_BR" in result.output
        assert "external SRT" in result.output

    def test_cli_coverage_json_output(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.en.srt")

        result = runner.invoke(app, ["coverage", str(library), "--format", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["movies"][0]["category"] == CoverageStatus.HAS_EN_ONLY
        assert "[" not in result.output or "HAS_EN_ONLY" in result.output

    def test_cli_coverage_missing_only(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        _touch_video(library / "HasPT" / "HasPT.mkv")
        _write_srt(library / "HasPT" / "HasPT.pt-BR.srt")
        _touch_video(library / "Missing" / "Missing.mkv")

        result = runner.invoke(app, ["coverage", str(library), "--missing-only"])

        assert result.exit_code == 0
        assert "Missing" in result.output
        assert "HasPT" not in result.output

    def test_cli_coverage_require_mount(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        library.mkdir()

        result = runner.invoke(app, ["coverage", str(library), "--require-mount"])

        assert result.exit_code == 2
        assert "not a mounted filesystem" in result.output

    def test_cli_coverage_json_missing_only_preserves_totals(
        self, tmp_path: Path
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "HasPT" / "HasPT.mkv")
        _write_srt(library / "HasPT" / "HasPT.pt-BR.srt")
        _touch_video(library / "Missing" / "Missing.mkv")

        result = runner.invoke(
            app, ["coverage", str(library), "--missing-only", "--format", "json"]
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["movies_discovered"] == 2
        assert payload["summary"][CoverageStatus.HAS_PT_BR] == 1
        assert payload["summary"][CoverageStatus.NO_EXTERNAL_SUBS] == 1
        assert len(payload["movies"]) == 1
        assert payload["movies"][0]["category"] == CoverageStatus.NO_EXTERNAL_SUBS


class TestMountPointValidation:
    def _fake_ismount_for(self, path: Path) -> callable:
        def fake_ismount(candidate: str | os.PathLike[str]) -> bool:
            return str(candidate) == str(path.resolve())

        return fake_ismount

    def test_coverage_accepts_library_under_explicit_mount_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = mount_point / "media" / "movies"
        library.mkdir(parents=True)
        _touch_video(library / "Movie" / "Movie.mkv")

        monkeypatch.setattr(
            os.path, "ismount", self._fake_ismount_for(mount_point)
        )

        result = scan_coverage(library, mount_point=mount_point)
        assert result.directories_scanned >= 1
        assert len(result.movies) == 1

    def test_coverage_rejects_unmounted_explicit_mount_point(
        self, tmp_path: Path
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = mount_point / "media" / "movies"
        library.mkdir(parents=True)
        _touch_video(library / "Movie" / "Movie.mkv")

        with pytest.raises(FileNotFoundError):
            scan_coverage(library, mount_point=mount_point)

    def test_coverage_rejects_library_outside_explicit_mount_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = tmp_path / "other" / "movies"
        library.mkdir(parents=True)
        _touch_video(library / "Movie" / "Movie.mkv")

        monkeypatch.setattr(
            os.path, "ismount", self._fake_ismount_for(mount_point)
        )

        with pytest.raises(FileNotFoundError):
            scan_coverage(library, mount_point=mount_point)

    def test_cli_coverage_rejects_unmounted_mount_point(
        self, tmp_path: Path
    ) -> None:
        mount_point = tmp_path / "nas"
        mount_point.mkdir()
        library = mount_point / "movies"
        library.mkdir(parents=True)

        result = runner.invoke(
            app,
            ["coverage", str(library), "--mount-point", str(mount_point)],
        )

        assert result.exit_code == 2
        assert "not under the required mount point" in result.output


class TestRegressionAgainstExistingCommands:
    def test_scan_still_finds_eligible_movie(self, tmp_path: Path) -> None:
        from subtranslate.config import get_scan_settings
        from subtranslate.scanner import ScanStatus, scan_library

        library = tmp_path / "library"
        _write_srt(library / "Movie" / "Movie.en.srt")

        settings = get_scan_settings()
        result = scan_library(library, settings)

        assert result.eligible_count == 1
        assert result.items[0].status == ScanStatus.TRANSLATE

    def test_scan_require_mount_still_rejects_unmounted(
        self, tmp_path: Path
    ) -> None:
        from subtranslate.config import get_scan_settings
        from subtranslate.scanner import scan_library

        library = tmp_path / "library"
        library.mkdir()

        settings = get_scan_settings()
        with pytest.raises(FileNotFoundError):
            scan_library(library, settings, require_mount=True)

    def test_scan_require_mount_accepts_path_under_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from subtranslate.config import get_scan_settings
        from subtranslate.scanner import scan_library

        library = tmp_path / "library"
        library.mkdir(parents=True)

        def fake_ismount(path: str | os.PathLike[str]) -> bool:
            return str(path) == str(tmp_path)

        monkeypatch.setattr(os.path, "ismount", fake_ismount)

        settings = get_scan_settings()
        result = scan_library(library, settings, require_mount=True)
        assert result.directories_scanned >= 1


def _ffprobe_json(streams: list[dict]) -> str:
    return json.dumps({"streams": streams})


def _patch_ffprobe(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> None:
    def fake_run(*_args, **_kwargs):
        class _Result:
            pass

        result = _Result()
        result.stdout = stdout
        result.stderr = ""
        result.returncode = returncode
        return result

    monkeypatch.setattr("subtranslate.coverage.subprocess.run", fake_run)


def _patch_ffprobe_side_effect(monkeypatch: pytest.MonkeyPatch, side_effect) -> None:
    monkeypatch.setattr("subtranslate.coverage.subprocess.run", side_effect)


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


class TestEmbeddedLanguageNormalization:
    def test_brazilian_portuguese_codes(self) -> None:
        for code in ["pt-BR", "pt_BR", "pob", "por-BR", "ptbr"]:
            assert normalize_embedded_language(code) == "pt-BR"

    def test_generic_portuguese_codes(self) -> None:
        for code in ["pt", "por", "pt-PT", "por-PT"]:
            assert normalize_embedded_language(code) == "pt"

    def test_english_codes(self) -> None:
        for code in ["en", "eng", "english"]:
            assert normalize_embedded_language(code) == "en"

    def test_unknown_language_returns_none(self) -> None:
        assert normalize_embedded_language("fra") is None
        assert normalize_embedded_language("") is None
        assert normalize_embedded_language(None) is None


class TestEmbeddedCodecClassification:
    def test_text_codecs(self) -> None:
        for codec in ["subrip", "srt", "ass", "ssa", "webvtt", "text", "mov_text"]:
            kind, is_text, is_image, supported = classify_embedded_codec(codec)
            assert kind == EMBEDDED_CODEC_TEXT
            assert is_text is True
            assert is_image is False
            assert supported is True

    def test_image_codecs(self) -> None:
        for codec in [
            "hdmv_pgs_subtitle",
            "dvd_subtitle",
            "vobsub",
            "pgs",
            "dvb_subtitle",
        ]:
            kind, is_text, is_image, supported = classify_embedded_codec(codec)
            assert kind == EMBEDDED_CODEC_IMAGE
            assert is_text is False
            assert is_image is True
            assert supported is False

    def test_unknown_codec(self) -> None:
        kind, is_text, is_image, supported = classify_embedded_codec("unknown_codec")
        assert kind == EMBEDDED_CODEC_UNKNOWN
        assert is_text is False
        assert is_image is False
        assert supported is False


class TestEmbeddedSubtitleDiscovery:
    def test_embedded_brazilian_portuguese_subtitle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(2, 0, codec="subrip", language="pt-BR")]),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert len(movie.embedded_subtitles) == 1
        assert movie.embedded_subtitles[0].is_brazilian_portuguese is True
        assert movie.category == CoverageStatus.NO_EXTERNAL_SUBS
        assert result.embedded_summary["brazilian_portuguese"] == 1

    def test_embedded_generic_portuguese_subtitle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(1, 0, codec="subrip", language="por")]),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.embedded_subtitles[0].is_generic_portuguese is True
        assert result.embedded_summary["generic_portuguese"] == 1

    def test_embedded_english_text_subtitle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="mov_text", language="eng")]),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.embedded_subtitles[0].is_english is True
        assert movie.embedded_subtitles[0].is_supported_text is True
        assert result.embedded_summary["english"] == 1
        assert result.embedded_summary["supported_text"] == 1

    def test_multiple_embedded_tracks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="subrip", language="eng"),
                    _stream(1, 1, codec="subrip", language="pt-BR"),
                ]
            ),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert len(movie.embedded_subtitles) == 2
        assert result.embedded_summary["tracks_discovered"] == 2

    def test_forced_only_and_sdh_only_tracks(
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

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.embedded_subtitles[0].is_forced is True
        assert movie.embedded_subtitles[1].is_sdh is True
        assert result.embedded_summary["forced"] == 1
        assert result.embedded_summary["sdh"] == 1

    def test_image_based_subtitles(
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

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.embedded_subtitles[0].is_image is True
        assert movie.embedded_subtitles[0].is_supported_text is False
        assert result.embedded_summary["image"] == 1

    def test_missing_language_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language=None)]),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.embedded_subtitles[0].is_unknown_language is True
        assert movie.embedded_subtitles[0].language_tag is None
        assert result.embedded_summary["unknown_language"] == 1

    def test_no_embedded_subtitles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(monkeypatch, _ffprobe_json([]))

        result = scan_coverage(library)

        assert result.movies[0].embedded_subtitles == ()
        assert result.movies[0].probe_error is None
        assert result.embedded_summary["tracks_discovered"] == 0

    def test_missing_ffprobe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")

        def raise_file_not_found(*_args, **_kwargs):
            raise FileNotFoundError("ffprobe")

        _patch_ffprobe_side_effect(monkeypatch, raise_file_not_found)

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.probe_error is not None
        assert "ffprobe not found" in movie.probe_error
        assert len(result.errors) == 1
        assert result.embedded_summary["movies_with_probe_error"] == 1

    def test_probe_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")

        def raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("ffprobe", 60)

        _patch_ffprobe_side_effect(monkeypatch, raise_timeout)

        result = scan_coverage(library)

        assert "timed out" in result.movies[0].probe_error
        assert result.embedded_summary["movies_with_probe_error"] == 1

    def test_probe_malformed_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(monkeypatch, "not valid json")

        result = scan_coverage(library)

        assert "valid JSON" in result.movies[0].probe_error
        assert result.embedded_summary["movies_with_probe_error"] == 1

    def test_probe_failure_preserves_external_subtitles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.srt")

        def raise_file_not_found(*_args, **_kwargs):
            raise FileNotFoundError("ffprobe")

        _patch_ffprobe_side_effect(monkeypatch, raise_file_not_found)

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_PT_BR
        assert len(movie.subtitles) == 1
        assert movie.probe_error is not None


class TestCombinedCoverage:
    def test_external_pt_br_plus_embedded_english(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.pt-BR.srt")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng")]),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_PT_BR
        assert len(movie.embedded_subtitles) == 1
        assert movie.embedded_subtitles[0].is_english is True

    def test_external_en_only_plus_embedded_pt_br_reported_separately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _write_srt(library / "Movie" / "Movie.en.srt")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="pt-BR")]),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.category == CoverageStatus.HAS_EN_ONLY
        assert movie.embedded_subtitles[0].is_brazilian_portuguese is True


class TestEmbeddedSerialization:
    def test_embedded_inventory_appears_in_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(2, 0, codec="subrip", language="pt-BR")]),
        )

        result = scan_coverage(library)
        payload = coverage_result_to_dict(result, library)
        movie = payload["movies"][0]

        assert movie["embedded_subtitles"][0]["language_tag"] == "pt-BR"
        assert movie["embedded_subtitles"][0]["codec_kind"] == EMBEDDED_CODEC_TEXT
        assert movie["embedded_subtitles"][0]["is_supported_text"] is True
        assert payload["embedded_summary"]["tracks_discovered"] == 1


class TestCliEmbeddedCoverage:
    def test_title_sdh_false_positive_avoided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json(
                [
                    _stream(0, 0, codec="subrip", language="eng", title="Chinese (Simplified)"),
                    _stream(1, 1, codec="subrip", language="eng", title="SDH"),
                ]
            ),
        )

        result = scan_coverage(library)

        movie = result.movies[0]
        assert movie.embedded_subtitles[0].is_sdh is False
        assert movie.embedded_subtitles[1].is_sdh is True

    def test_probe_nonzero_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(monkeypatch, "error", returncode=1)

        result = scan_coverage(library)

        assert "ffprobe failed" in result.movies[0].probe_error
        assert result.embedded_summary["movies_with_probe_error"] == 1

    def test_probe_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")

        def raise_oserror(*_args, **_kwargs):
            raise OSError("permission denied")

        _patch_ffprobe_side_effect(monkeypatch, raise_oserror)

        result = scan_coverage(library)

        assert "could not be executed" in result.movies[0].probe_error
        assert result.embedded_summary["movies_with_probe_error"] == 1

    def test_probe_valid_json_wrong_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(monkeypatch, json.dumps([]))

        result = scan_coverage(library)

        assert "unexpected shape" in result.movies[0].probe_error
        assert result.embedded_summary["movies_with_probe_error"] == 1

    def test_ffprobe_argv_uses_argument_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        video = library / "Movie" / "Movie.mkv"
        _touch_video(video)
        calls: list[list[str]] = []

        def capture_run(args, **_kwargs):
            calls.append(list(args))
            class _Result:
                pass
            result = _Result()
            result.stdout = json.dumps({"streams": []})
            result.stderr = ""
            result.returncode = 0
            return result

        monkeypatch.setattr("subtranslate.coverage.subprocess.run", capture_run)
        scan_coverage(library)

        assert len(calls) == 1
        argv = calls[0]
        assert argv[0] == "ffprobe"
        assert "-select_streams" in argv
        assert "s" in argv
        assert "-of" in argv
        assert "json" in argv
        assert str(video) in argv

    def test_cli_shows_embedded_subtitles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")
        _patch_ffprobe(
            monkeypatch,
            _ffprobe_json([_stream(0, 0, codec="subrip", language="eng", forced=1)]),
        )

        result = runner.invoke(app, ["coverage", str(library)])

        assert result.exit_code == 0
        assert "en (subrip)" in result.output
        assert "(forced)" in result.output

    def test_cli_shows_probe_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "Movie" / "Movie.mkv")

        def raise_file_not_found(*_args, **_kwargs):
            raise FileNotFoundError("ffprobe")

        _patch_ffprobe_side_effect(monkeypatch, raise_file_not_found)

        result = runner.invoke(app, ["coverage", str(library)])

        assert result.exit_code == 0
        assert "probe error" in result.output or "ffprobe" in result.output

    def test_cli_long_movie_name_is_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        long_name = "A" * 80
        _touch_video(library / long_name / f"{long_name}.mkv")
        _patch_ffprobe(monkeypatch, _ffprobe_json([]))

        result = runner.invoke(app, ["coverage", str(library)])

        assert result.exit_code == 0
        # Rich folds long cell text across lines, so the full raw string may not
        # be contiguous. Verify a clear prefix is visible and readable.
        assert long_name[:20] in result.output
        assert "NO_EXTERNAL_SUBS" in result.output

    def test_cli_missing_only_with_embedded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = tmp_path / "library"
        _touch_video(library / "HasPT" / "HasPT.mkv")
        _write_srt(library / "HasPT" / "HasPT.pt-BR.srt")
        _touch_video(library / "Missing" / "Missing.mkv")
        _patch_ffprobe(monkeypatch, _ffprobe_json([]))

        result = runner.invoke(app, ["coverage", str(library), "--missing-only"])

        assert result.exit_code == 0
        assert "Missing" in result.output
        assert "HasPT" not in result.output
