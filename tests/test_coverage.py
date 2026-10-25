from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from subtranslate.cli import app
from subtranslate.coverage import (
    VIDEO_EXTENSIONS,
    CoverageStatus,
    coverage_result_to_dict,
    scan_coverage,
)

runner = CliRunner()


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
            "ambiguous_reasons",
        }
        assert movie["category"] == CoverageStatus.HAS_PT_BR

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
