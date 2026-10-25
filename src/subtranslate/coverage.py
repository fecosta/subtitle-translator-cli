from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .naming import (
    SOURCE_LANG_TAGS,
    TARGET_LANG_TAGS,
    SubtitleName,
    parse_subtitle_name,
)

VIDEO_EXTENSIONS = frozenset({".mkv", ".mp4", ".avi", ".m4v"})


class CoverageStatus:
    HAS_PT_BR = "HAS_PT_BR"
    HAS_GENERIC_PT = "HAS_GENERIC_PT"
    HAS_EN_ONLY = "HAS_EN_ONLY"
    HAS_UNLABELED = "HAS_UNLABELED"
    HAS_SPECIALIZED_ONLY = "HAS_SPECIALIZED_ONLY"
    NO_EXTERNAL_SUBS = "NO_EXTERNAL_SUBS"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class SubtitleInventory:
    """Observed external subtitle for a movie."""

    path: Path
    language_tag: str | None
    attributes: tuple[str, ...]
    is_brazilian_portuguese: bool
    is_generic_portuguese: bool
    is_english: bool
    is_unlabeled: bool
    is_specialized_variant: bool


@dataclass(slots=True)
class MovieCoverage:
    """External subtitle coverage for a single discovered movie."""

    movie_dir: Path
    movie_name: str
    video_paths: tuple[Path, ...]
    category: str
    subtitles: tuple[SubtitleInventory, ...]
    ambiguous_reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class CoverageResult:
    """Aggregate results of a coverage scan."""

    movies: tuple[MovieCoverage, ...]
    directories_scanned: int
    errors: tuple[tuple[Path, str], ...]

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {
            CoverageStatus.HAS_PT_BR: 0,
            CoverageStatus.HAS_GENERIC_PT: 0,
            CoverageStatus.HAS_EN_ONLY: 0,
            CoverageStatus.HAS_UNLABELED: 0,
            CoverageStatus.HAS_SPECIALIZED_ONLY: 0,
            CoverageStatus.NO_EXTERNAL_SUBS: 0,
            CoverageStatus.AMBIGUOUS: 0,
        }
        for movie in self.movies:
            counts[movie.category] = counts.get(movie.category, 0) + 1
        return counts


def is_under_mount(path: Path) -> bool:
    """Return True when path is mounted or resides under a non-root mount point."""
    resolved = path.resolve()
    if os.path.ismount(resolved):
        return True
    root = Path(resolved.root)
    for parent in resolved.parents:
        if parent == root:
            break
        if os.path.ismount(parent):
            return True
    return False


def is_under_specific_mount(library_path: Path, mount_point: Path) -> bool:
    """Return True when mount_point is mounted and library_path is under it."""
    resolved_library = library_path.resolve()
    resolved_mount = mount_point.resolve()
    if not resolved_mount.exists():
        return False
    if not resolved_mount.is_dir():
        return False
    if not os.path.ismount(resolved_mount):
        return False
    if resolved_mount == resolved_library:
        return True
    return resolved_library.is_relative_to(resolved_mount)


def _candidate_movie_names(parsed: SubtitleName) -> frozenset[str]:
    """Possible video basenames a subtitle may belong to.

    Includes the parsed movie name plus, for Portuguese subtitles only, the
    base name left after stripping a trailing source-language tag. This covers
    legacy translations such as ``Movie.eng.pt-BR.srt``.
    """
    names: set[str] = {parsed.movie_name}
    if parsed.language_tag is None or parsed.language_tag.lower() not in TARGET_LANG_TAGS:
        return frozenset(names)

    lower_movie = parsed.movie_name.lower()
    for tag in SOURCE_LANG_TAGS:
        suffix = "." + tag
        if lower_movie.endswith(suffix):
            stripped = parsed.movie_name[: -len(suffix)]
            if stripped:
                names.add(stripped)
            break
    return frozenset(names)


def _inventory(parsed: SubtitleName) -> SubtitleInventory:
    return SubtitleInventory(
        path=parsed.path,
        language_tag=parsed.language_tag,
        attributes=parsed.attributes,
        is_brazilian_portuguese=parsed.is_brazilian_portuguese,
        is_generic_portuguese=parsed.is_generic_portuguese,
        is_english=parsed.is_source_language,
        is_unlabeled=parsed.is_unlabeled,
        is_specialized_variant=parsed.has_specialized_variant,
    )


def _associate_subtitles(
    video_stems: set[str],
    srts: list[tuple[Path, SubtitleName, frozenset[str]]],
) -> tuple[dict[str, list[SubtitleInventory]], dict[str, list[str]]]:
    associated: dict[str, list[SubtitleInventory]] = {stem: [] for stem in video_stems}
    ambiguous: dict[str, list[str]] = {stem: [] for stem in video_stems}

    for path, parsed, candidates in srts:
        inventory = _inventory(parsed)
        candidate_keys = {candidate.lower() for candidate in candidates}
        matches = candidate_keys & video_stems
        if len(matches) == 1:
            stem = next(iter(matches))
            associated[stem].append(inventory)
        elif len(matches) > 1:
            reason = f"subtitle {path.name} matches multiple videos"
            for stem in sorted(matches):
                ambiguous[stem].append(reason)
        # Unmatched subtitles are ignored; they do not safely belong to any video.

    return associated, ambiguous


def _classify(
    subtitles: tuple[SubtitleInventory, ...],
    ambiguous_reasons: tuple[str, ...],
) -> str:
    if ambiguous_reasons:
        return CoverageStatus.AMBIGUOUS

    full_subtitles = [s for s in subtitles if not s.is_specialized_variant]
    if any(s.is_brazilian_portuguese for s in full_subtitles):
        return CoverageStatus.HAS_PT_BR
    if any(s.is_generic_portuguese for s in full_subtitles):
        return CoverageStatus.HAS_GENERIC_PT
    if any(s.is_english for s in full_subtitles):
        return CoverageStatus.HAS_EN_ONLY
    if any(s.is_unlabeled for s in full_subtitles):
        return CoverageStatus.HAS_UNLABELED
    if subtitles:
        return CoverageStatus.HAS_SPECIALIZED_ONLY
    return CoverageStatus.NO_EXTERNAL_SUBS


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS and not path.is_symlink()


def _is_srt_file(path: Path) -> bool:
    return path.suffix.lower() == ".srt" and not path.is_symlink()


def scan_coverage(
    library_path: Path,
    *,
    require_mount: bool = False,
    mount_point: Path | None = None,
) -> CoverageResult:
    """Recursively inspect a movie library and report external SRT coverage."""
    if library_path.is_symlink():
        raise ValueError(f"Library path must not be a symlink: {library_path}")

    library_path = library_path.resolve()
    if not library_path.exists():
        raise FileNotFoundError(f"Library path does not exist: {library_path}")
    if not library_path.is_dir():
        raise NotADirectoryError(f"Library path is not a directory: {library_path}")
    if mount_point is not None and not is_under_specific_mount(library_path, mount_point):
        raise FileNotFoundError(
            f"Library path is not under the required mount point: {mount_point}"
        )
    if require_mount and not is_under_mount(library_path):
        raise FileNotFoundError(
            f"Library path is not a mounted filesystem: {library_path}"
        )

    walk_errors: list[tuple[Path, str]] = []

    def on_error(error: OSError) -> None:
        walk_errors.append((Path(error.filename or ""), str(error)))

    movies: dict[tuple[Path, str], MovieCoverage] = {}
    directories_scanned = 0

    for root, _dirs, files in os.walk(
        library_path, followlinks=False, onerror=on_error
    ):
        root_path = Path(root)
        directories_scanned += 1

        videos: dict[str, list[Path]] = defaultdict(list)
        display_names: dict[str, str] = {}
        srts: list[tuple[Path, SubtitleName, frozenset[str]]] = []

        for filename in files:
            file_path = root_path / filename
            if _is_video_file(file_path):
                key = file_path.stem.lower()
                videos[key].append(file_path)
                if key not in display_names:
                    display_names[key] = file_path.stem
            elif _is_srt_file(file_path):
                try:
                    parsed = parse_subtitle_name(file_path)
                except Exception as exc:  # noqa: BLE001
                    walk_errors.append((file_path, str(exc)))
                    continue
                srts.append((file_path, parsed, _candidate_movie_names(parsed)))

        if not videos:
            continue

        video_stems = set(videos)
        associated, ambiguous = _associate_subtitles(video_stems, srts)

        for stem in sorted(video_stems):
            subtitles = tuple(associated.get(stem, ()))
            ambiguous_reasons = tuple(ambiguous.get(stem, ()))
            movies[(root_path, display_names[stem])] = MovieCoverage(
                movie_dir=root_path,
                movie_name=display_names[stem],
                video_paths=tuple(sorted(videos[stem])),
                category=_classify(subtitles, ambiguous_reasons),
                subtitles=subtitles,
                ambiguous_reasons=ambiguous_reasons,
            )

    sorted_movies = tuple(
        sorted(movies.values(), key=lambda m: (m.movie_dir, m.movie_name))
    )
    return CoverageResult(
        movies=sorted_movies,
        directories_scanned=directories_scanned,
        errors=tuple(walk_errors),
    )


def coverage_result_to_dict(
    result: CoverageResult,
    library_path: Path,
) -> dict:
    """Convert a CoverageResult to a JSON-serializable dictionary."""
    resolved = library_path.resolve()

    def _relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(resolved))
        except ValueError:
            return str(path)

    return {
        "library_path": str(resolved),
        "directories_scanned": result.directories_scanned,
        "movies_discovered": len(result.movies),
        "errors": [
            {"path": _relative(path), "reason": reason}
            for path, reason in result.errors
        ],
        "summary": result.summary,
        "movies": [
            {
                "path": _relative(movie.movie_dir),
                "movie_name": movie.movie_name,
                "category": movie.category,
                "video_files": [path.name for path in movie.video_paths],
                "subtitles": [
                    {
                        "filename": sub.path.name,
                        "language_tag": sub.language_tag,
                        "attributes": list(sub.attributes),
                        "is_brazilian_portuguese": sub.is_brazilian_portuguese,
                        "is_generic_portuguese": sub.is_generic_portuguese,
                        "is_english": sub.is_english,
                        "is_unlabeled": sub.is_unlabeled,
                        "is_specialized_variant": sub.is_specialized_variant,
                    }
                    for sub in movie.subtitles
                ],
                "ambiguous_reasons": list(movie.ambiguous_reasons),
            }
            for movie in result.movies
        ],
    }
