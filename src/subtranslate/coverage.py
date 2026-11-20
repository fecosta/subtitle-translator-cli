from __future__ import annotations

import json
import os
import re
import subprocess
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

FFPROBE_TIMEOUT_SECONDS = 60

EMBEDDED_CODEC_TEXT = "text"
EMBEDDED_CODEC_IMAGE = "image"
EMBEDDED_CODEC_UNKNOWN = "unknown"

TEXT_SUBTITLE_CODECS = frozenset(
    {"subrip", "srt", "ass", "ssa", "webvtt", "text", "mov_text"}
)
IMAGE_SUBTITLE_CODECS = frozenset(
    {"hdmv_pgs_subtitle", "dvb_subtitle", "dvd_subtitle", "vobsub", "pgs"}
)

BRAZILIAN_PORTUGUESE_CODES = frozenset(
    {"pt-br", "pt_br", "pob", "ptbr", "por-br", "porbr"}
)
GENERIC_PORTUGUESE_CODES = frozenset({"pt", "por", "pt-pt", "por-pt"})
ENGLISH_LANGUAGE_CODES = frozenset({"en", "eng", "english"})


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


@dataclass(frozen=True, slots=True)
class EmbeddedSubtitleInventory:
    """Observed embedded subtitle stream for a movie."""

    stream_index: int
    subtitle_stream_index: int
    video_file: str
    codec: str
    codec_kind: str
    language_tag: str | None
    raw_language_tag: str | None
    title: str
    is_brazilian_portuguese: bool
    is_generic_portuguese: bool
    is_english: bool
    is_unknown_language: bool
    is_forced: bool
    is_sdh: bool
    is_text: bool
    is_image: bool
    is_supported_text: bool


@dataclass(slots=True)
class MovieCoverage:
    """Subtitle coverage for a single discovered movie."""

    movie_dir: Path
    movie_name: str
    video_paths: tuple[Path, ...]
    category: str
    subtitles: tuple[SubtitleInventory, ...]
    embedded_subtitles: tuple[EmbeddedSubtitleInventory, ...] = field(default_factory=tuple)
    ambiguous_reasons: tuple[str, ...] = field(default_factory=tuple)
    probe_error: str | None = None


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

    @property
    def embedded_summary(self) -> dict[str, int]:
        """Aggregate counts across all discovered embedded subtitle streams."""
        counts: dict[str, int] = {
            "tracks_discovered": 0,
            "movies_with_embedded_subtitles": 0,
            "movies_with_probe_error": 0,
            "brazilian_portuguese": 0,
            "generic_portuguese": 0,
            "english": 0,
            "unknown_language": 0,
            "forced": 0,
            "sdh": 0,
            "text": 0,
            "image": 0,
            "supported_text": 0,
            "unknown_codec": 0,
        }
        for movie in self.movies:
            if movie.probe_error:
                counts["movies_with_probe_error"] += 1
            if movie.embedded_subtitles:
                counts["movies_with_embedded_subtitles"] += 1
            for stream in movie.embedded_subtitles:
                counts["tracks_discovered"] += 1
                if stream.is_brazilian_portuguese:
                    counts["brazilian_portuguese"] += 1
                if stream.is_generic_portuguese:
                    counts["generic_portuguese"] += 1
                if stream.is_english:
                    counts["english"] += 1
                if stream.is_unknown_language:
                    counts["unknown_language"] += 1
                if stream.is_forced:
                    counts["forced"] += 1
                if stream.is_sdh:
                    counts["sdh"] += 1
                if stream.is_text:
                    counts["text"] += 1
                if stream.is_image:
                    counts["image"] += 1
                if stream.is_supported_text:
                    counts["supported_text"] += 1
                if stream.codec_kind == EMBEDDED_CODEC_UNKNOWN:
                    counts["unknown_codec"] += 1
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


def normalize_embedded_language(value: str | None) -> str | None:
    """Normalize an ffprobe language tag to a canonical tag.

    Returns ``pt-BR``, ``pt``, ``en``, or ``None`` when the language cannot be
    confidently identified as one of those three.
    """
    if not value:
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized in BRAZILIAN_PORTUGUESE_CODES:
        return "pt-BR"
    if normalized in GENERIC_PORTUGUESE_CODES:
        return "pt"
    if normalized in ENGLISH_LANGUAGE_CODES:
        return "en"
    return None


def classify_embedded_codec(codec: str | None) -> tuple[str, bool, bool, bool]:
    """Return (codec_kind, is_text, is_image, is_supported_text) for a codec name."""
    name = (codec or "").lower()
    if name in TEXT_SUBTITLE_CODECS:
        return EMBEDDED_CODEC_TEXT, True, False, True
    if name in IMAGE_SUBTITLE_CODECS:
        return EMBEDDED_CODEC_IMAGE, False, True, False
    return EMBEDDED_CODEC_UNKNOWN, False, False, False


_FORCED_TITLE_RE = re.compile(r"\bforced\b", re.IGNORECASE)
_SDH_TITLE_RE = re.compile(r"\b(sdh|hi|cc)\b|\bhearing\b|\bdeaf\b", re.IGNORECASE)


def _title_implies_forced(title: str) -> bool:
    return bool(_FORCED_TITLE_RE.search(title))


def _title_implies_sdh(title: str) -> bool:
    return bool(_SDH_TITLE_RE.search(title))


def _parse_embedded_stream(
    stream: dict, subtitle_stream_index: int, video_path: Path
) -> EmbeddedSubtitleInventory:
    tags = stream.get("tags", {}) or {}
    raw_language = tags.get("language")
    language_tag = normalize_embedded_language(raw_language)
    title = (tags.get("title", "") or "").strip()
    codec = (stream.get("codec_name") or "").lower()
    codec_kind, is_text, is_image, is_supported_text = classify_embedded_codec(codec)
    disposition = stream.get("disposition", {}) or {}
    is_forced = bool(disposition.get("forced")) or _title_implies_forced(title)
    is_sdh = bool(disposition.get("hearing_impaired")) or _title_implies_sdh(title)

    return EmbeddedSubtitleInventory(
        stream_index=int(stream.get("index", subtitle_stream_index)),
        subtitle_stream_index=subtitle_stream_index,
        video_file=video_path.name,
        codec=codec,
        codec_kind=codec_kind,
        language_tag=language_tag,
        raw_language_tag=raw_language if raw_language else None,
        title=title,
        is_brazilian_portuguese=language_tag == "pt-BR",
        is_generic_portuguese=language_tag == "pt",
        is_english=language_tag == "en",
        is_unknown_language=language_tag is None,
        is_forced=is_forced,
        is_sdh=is_sdh,
        is_text=is_text,
        is_image=is_image,
        is_supported_text=is_supported_text,
    )


def probe_video_subtitles(video_path: Path) -> tuple[list[EmbeddedSubtitleInventory], str | None]:
    """Probe a single video for embedded subtitle streams.

    Returns a list of discovered streams and an optional error message. An error
    is returned when ffprobe is missing, the probe fails, or the output cannot
    be parsed. A successful probe with no subtitle streams returns an empty list
    and no error.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name:stream_tags=language,title:stream_disposition=forced,hearing_impaired",
                "-of",
                "json",
                str(video_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        return [], f"ffprobe not found: {exc}"
    except subprocess.TimeoutExpired:
        return [], f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS}s for {video_path.name}"
    except OSError as exc:
        return [], f"ffprobe could not be executed: {exc}"

    if result.returncode != 0:
        return [], f"ffprobe failed for {video_path.name}: {result.stderr.strip()}"

    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", []) or []
        return [
            _parse_embedded_stream(stream, index, video_path)
            for index, stream in enumerate(streams)
        ], None
    except json.JSONDecodeError as exc:
        return [], f"ffprobe output is not valid JSON for {video_path.name}: {exc}"
    except (AttributeError, TypeError, ValueError) as exc:
        return [], f"ffprobe output has an unexpected shape for {video_path.name}: {exc}"


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


def scan_single_video(
    video_path: Path,
    *,
    require_mount: bool = False,
    mount_point: Path | None = None,
) -> CoverageResult:
    """Inspect a single video file and its same-directory external subtitles.

    Only the given video is probed, and only SRT files in the same directory
    are considered for subtitle association. Sibling videos and
    subdirectories are never touched.
    """
    if video_path.is_symlink():
        raise ValueError(f"Video path must not be a symlink: {video_path}")

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    if not video_path.is_file():
        raise NotADirectoryError(f"Video path is not a file: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension: {video_path.suffix} "
            f"(supported: {', '.join(sorted(VIDEO_EXTENSIONS))})"
        )

    movie_dir = video_path.parent
    if mount_point is not None and not is_under_specific_mount(movie_dir, mount_point):
        raise FileNotFoundError(
            f"Library path is not under the required mount point: {mount_point}"
        )
    if require_mount and not is_under_mount(movie_dir):
        raise FileNotFoundError(
            f"Library path is not a mounted filesystem: {movie_dir}"
        )

    video_stem = video_path.stem.lower()
    videos: dict[str, list[Path]] = {video_stem: [video_path]}
    display_names: dict[str, str] = {video_stem: video_path.stem}
    srts: list[tuple[Path, SubtitleName, frozenset[str]]] = []

    try:
        for entry in os.scandir(movie_dir):
            if not entry.is_file(follow_symlinks=False):
                continue
            file_path = Path(entry.path)
            if _is_srt_file(file_path):
                try:
                    parsed = parse_subtitle_name(file_path)
                except Exception:  # noqa: BLE001
                    continue
                srts.append((file_path, parsed, _candidate_movie_names(parsed)))
    except OSError:
        pass

    video_stems = set(videos)
    associated, ambiguous = _associate_subtitles(video_stems, srts)

    subtitles = tuple(associated.get(video_stem, ()))
    ambiguous_reasons = tuple(ambiguous.get(video_stem, ()))
    embedded, probe_error = probe_video_subtitles(video_path)

    errors: list[tuple[Path, str]] = []
    if probe_error:
        errors.append((video_path, probe_error))

    movie = MovieCoverage(
        movie_dir=movie_dir,
        movie_name=display_names[video_stem],
        video_paths=tuple(sorted(videos[video_stem])),
        category=_classify(subtitles, ambiguous_reasons),
        subtitles=subtitles,
        embedded_subtitles=tuple(embedded),
        ambiguous_reasons=ambiguous_reasons,
        probe_error=probe_error,
    )

    return CoverageResult(
        movies=(movie,),
        directories_scanned=1,
        errors=tuple(errors),
    )


def scan_coverage(
    library_path: Path,
    *,
    require_mount: bool = False,
    mount_point: Path | None = None,
) -> CoverageResult:
    """Recursively inspect a movie library and report subtitle coverage.

    Discovers external SRT files and embedded subtitle streams in supported
    video files. Embedded discovery is read-only: it runs ffprobe but never
    extracts, translates, or modifies media files.
    """
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
            embedded: list[EmbeddedSubtitleInventory] = []
            probe_error: str | None = None
            for video_path in sorted(videos[stem]):
                streams, error = probe_video_subtitles(video_path)
                embedded.extend(streams)
                if error and probe_error is None:
                    probe_error = error
            if probe_error:
                walk_errors.append((root_path / display_names[stem], probe_error))
            movies[(root_path, display_names[stem])] = MovieCoverage(
                movie_dir=root_path,
                movie_name=display_names[stem],
                video_paths=tuple(sorted(videos[stem])),
                category=_classify(subtitles, ambiguous_reasons),
                subtitles=subtitles,
                embedded_subtitles=tuple(embedded),
                ambiguous_reasons=ambiguous_reasons,
                probe_error=probe_error,
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
        "embedded_summary": result.embedded_summary,
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
                "embedded_subtitles": [
                    {
                        "stream_index": emb.stream_index,
                        "subtitle_stream_index": emb.subtitle_stream_index,
                        "video_file": emb.video_file,
                        "codec": emb.codec,
                        "codec_kind": emb.codec_kind,
                        "language_tag": emb.language_tag,
                        "raw_language_tag": emb.raw_language_tag,
                        "title": emb.title,
                        "is_brazilian_portuguese": emb.is_brazilian_portuguese,
                        "is_generic_portuguese": emb.is_generic_portuguese,
                        "is_english": emb.is_english,
                        "is_unknown_language": emb.is_unknown_language,
                        "is_forced": emb.is_forced,
                        "is_sdh": emb.is_sdh,
                        "is_text": emb.is_text,
                        "is_image": emb.is_image,
                        "is_supported_text": emb.is_supported_text,
                    }
                    for emb in movie.embedded_subtitles
                ],
                "probe_error": movie.probe_error,
                "ambiguous_reasons": list(movie.ambiguous_reasons),
            }
            for movie in result.movies
        ],
    }
