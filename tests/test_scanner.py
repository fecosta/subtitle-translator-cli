from __future__ import annotations

import os
from pathlib import Path

import pytest

from subtranslate.config import Settings, get_scan_settings
from subtranslate.scanner import ScanStatus, scan_library


@pytest.fixture
def scan_settings() -> Settings:
    return get_scan_settings(
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


def test_scan_finds_eligible_movie(tmp_path: Path, scan_settings: Settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 1
    assert result.english_files == 1
    assert result.portuguese_files == 0
    assert result.items[0].status == ScanStatus.TRANSLATE
    assert result.items[0].output_path == library / "Movie A" / "Movie.A.pt-BR.srt"


def test_scan_skips_existing_brazilian_portuguese(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie A" / "Movie.A.pt-BR.srt")

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 0
    assert result.skipped_count == 1
    assert result.portuguese_files == 1
    assert "Brazilian Portuguese" in result.items[0].reason


def test_scan_skips_existing_generic_portuguese(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie A" / "Movie.A.pt.srt")

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 0
    assert result.skipped_count == 1
    assert "Generic Portuguese" in result.items[0].reason


def test_scan_marks_multiple_english_sources_ambiguous(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie A" / "Movie.A.eng.srt")

    result = scan_library(library, scan_settings)

    assert result.ambiguous_count == 1
    assert result.eligible_count == 0
    assert result.items[0].status == ScanStatus.AMBIGUOUS


def test_scan_excludes_specialized_variants_by_default(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.forced.srt")
    _write_srt(library / "Movie A" / "Movie.A.eng.sdh.srt")

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 0
    assert result.skipped_count == 1
    assert "specialized variant" in result.items[0].reason


def test_scan_can_include_variants(tmp_path: Path, scan_settings: Settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.forced.srt")

    result = scan_library(library, scan_settings, include_variants=True)

    assert result.eligible_count == 1
    assert result.items[0].output_path == library / "Movie A" / "Movie.A.pt-BR.forced.srt"


def test_scan_skips_unlabeled_srt(tmp_path: Path, scan_settings: Settings) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.srt")

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 0
    assert result.skipped_count == 1
    assert "No recognized English subtitle" in result.items[0].reason


def test_scan_detects_invalid_srt(tmp_path: Path, scan_settings: Settings) -> None:
    library = tmp_path / "library"
    srt = library / "Movie A" / "Movie.A.en.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text("not a valid srt file", encoding="utf-8")

    result = scan_library(library, scan_settings)

    assert result.error_count == 1
    assert result.items[0].status == ScanStatus.ERROR


def test_scan_handles_multiple_movie_directories(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie B" / "Movie.B.en.srt")
    _write_srt(library / "Movie C" / "Movie.C.pt-BR.srt")

    result = scan_library(library, scan_settings)

    assert result.directories_scanned == 4  # library root + three movie dirs
    assert result.eligible_count == 2
    assert result.skipped_count == 1


def test_scan_ignores_symlinks(tmp_path: Path, scan_settings: Settings) -> None:
    library = tmp_path / "library"
    real_srt = tmp_path / "outside.srt"
    _write_srt(real_srt)
    link_dir = library / "Movie A"
    link_dir.mkdir(parents=True)
    link = link_dir / "Movie.A.en.srt"
    link.symlink_to(real_srt)

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 0
    assert result.english_files == 0


def test_scan_reports_permission_errors_gracefully(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    inaccessible = library / "Movie A"
    inaccessible.mkdir(parents=True)
    try:
        os.chmod(inaccessible, 0o000)
        result = scan_library(library, scan_settings)
    finally:
        os.chmod(inaccessible, 0o755)

    assert result.error_count >= 1


def test_scan_refers_missing_directory(tmp_path: Path, scan_settings: Settings) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        scan_library(missing, scan_settings)


def test_scan_require_mount_rejects_unmounted_directory(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    library.mkdir()
    with pytest.raises(FileNotFoundError):
        scan_library(library, scan_settings, require_mount=True)


def test_scan_estimates_cost_for_known_model(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")

    result = scan_library(library, scan_settings)

    assert result.total_estimated_input_tokens > 0
    assert result.total_estimated_output_tokens > 0
    assert result.total_estimated_cost_usd is not None
    assert result.total_estimated_cost_usd > 0


def test_scan_reports_unknown_cost_for_unknown_model(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    settings = get_scan_settings(model="unknown-model", target_lang="pt-BR")

    result = scan_library(library, settings)

    assert result.eligible_count == 1
    assert result.total_estimated_cost_usd is None


def test_scan_groups_by_movie_name_within_directory(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie A" / "Movie.A.pt-BR.srt")
    _write_srt(library / "Movie A" / "Other.en.srt")

    result = scan_library(library, scan_settings)

    statuses = {item.movie_name: item.status for item in result.items}
    assert statuses["Movie.A"] == ScanStatus.SKIP
    assert statuses["Other"] == ScanStatus.TRANSLATE


def test_scan_variant_portuguese_does_not_block_full_translation(
    tmp_path: Path, scan_settings: Settings
) -> None:
    library = tmp_path / "library"
    _write_srt(library / "Movie A" / "Movie.A.en.srt")
    _write_srt(library / "Movie A" / "Movie.A.pt-BR.forced.srt")

    result = scan_library(library, scan_settings)

    assert result.eligible_count == 1
    assert result.portuguese_files == 1
    assert result.items[0].status == ScanStatus.TRANSLATE


def test_scan_rejects_symlinked_root(tmp_path: Path, scan_settings: Settings) -> None:
    real_library = tmp_path / "real_library"
    _write_srt(real_library / "Movie A" / "Movie.A.en.srt")
    link = tmp_path / "link_library"
    link.symlink_to(real_library)

    with pytest.raises(FileNotFoundError):
        scan_library(link, scan_settings)
