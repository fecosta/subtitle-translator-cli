"""Extract eligible embedded English subtitles to external .en.srt files.

Selection rules:
- Eligible: is_english AND is_supported_text AND NOT is_forced AND NOT is_sdh.
- Tie-break: prefer subrip/srt codec, then lowest subtitle_stream_index.

Skip reasons:
- Existing non-specialized external English subtitle for the movie.
- ffprobe error for the movie.
- Ambiguous subtitle-to-video association for the movie.
- No eligible embedded English text track for a video.
- Target output path already exists (race protection).

Output filename: <video stem>.en.srt in the video's directory.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .coverage import (
    VIDEO_EXTENSIONS,
    EmbeddedSubtitleInventory,
    MovieCoverage,
    scan_coverage,
    scan_single_video,
)
from .exceptions import OutputExistsError, SubtitleError
from .fileio import atomic_write_path
from .srt import load_subtitles

FFMPEG_TIMEOUT_SECONDS = 600

PREFERRED_TEXT_CODECS = frozenset({"subrip", "srt"})


def _run_ffmpeg(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    """Thin wrapper around subprocess.run for ffmpeg calls.

    Exists so tests can monkeypatch ffmpeg independently of ffprobe.
    """
    return subprocess.run(args, **kwargs)


class ExtractionStatus:
    EXTRACTED = "extracted"
    WOULD_EXTRACT = "would_extract"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExtractionSkipReason:
    HAS_EXTERNAL_ENGLISH = "has external English subtitle"
    PROBE_ERROR = "probe error"
    AMBIGUOUS = "ambiguous"
    NO_ELIGIBLE_TRACK = "no eligible embedded English text track"
    OUTPUT_EXISTS = "output exists"


class ExtractionFailedError(Exception):
    """Embedded subtitle extraction failed for a single video."""


@dataclass(slots=True)
class ExtractionOutcome:
    """Result of processing one movie (or one video within a movie)."""

    movie_dir: Path
    movie_name: str
    video_path: Path | None
    status: str
    reason: str
    output_path: Path | None = None
    track: EmbeddedSubtitleInventory | None = None


@dataclass(slots=True)
class ExtractionResult:
    """Aggregate results of an extract-embedded run."""

    movies_scanned: int
    eligible_tracks: int
    performed: int
    skipped: int
    failures: int
    outcomes: list[ExtractionOutcome] = field(default_factory=list)
    skipped_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dry_run: bool = False


def _target_output_path(video_path: Path) -> Path:
    """Return <video stem>.en.srt next to the source video."""
    return video_path.with_name(video_path.stem + ".en.srt")


def _target_exists(path: Path) -> bool:
    """Check for an existing file, including case-insensitive matches."""
    if path.exists():
        return True
    try:
        target_name = path.name.lower()
        return any(
            existing.name.lower() == target_name for existing in path.parent.iterdir()
        )
    except OSError:
        return False


def _has_appropriate_external_english(movie: MovieCoverage) -> bool:
    """Return True when the movie already has a full (non-variant) English sub."""
    return any(
        sub.is_english and not sub.is_specialized_variant for sub in movie.subtitles
    )


def select_eligible_track(
    embedded_subtitles: tuple[EmbeddedSubtitleInventory, ...],
    video_file: str,
) -> EmbeddedSubtitleInventory | None:
    """Choose the best eligible embedded English text track for a video file.

    Eligible tracks are English, supported text codecs, and not forced or SDH.
    Preference: subrip/srt codec first, then lowest subtitle_stream_index.
    """
    eligible = [
        track
        for track in embedded_subtitles
        if track.video_file == video_file
        and track.is_english
        and track.is_supported_text
        and not track.is_forced
        and not track.is_sdh
    ]
    if not eligible:
        return None

    def sort_key(track: EmbeddedSubtitleInventory) -> tuple[int, int]:
        preferred = 0 if track.codec in PREFERRED_TEXT_CODECS else 1
        return (preferred, track.subtitle_stream_index)

    return min(eligible, key=sort_key)


def _extract_track(
    video_path: Path,
    track: EmbeddedSubtitleInventory,
    output_path: Path,
) -> None:
    """Run ffmpeg to extract one subtitle stream and install it atomically.

    Raises ExtractionFailedError on any extraction or validation failure.
    Raises OutputExistsError when the target already exists (race).
    """
    # Create a private temporary path that does not exist when ffmpeg runs so
    # that ffmpeg's -n (no-clobber) flag succeeds. The leading dot keeps a
    # leaked temp file from looking like a final subtitle, and the .srt suffix
    # lets ffmpeg select the SRT muxer and lets pysubs2 parse the result.
    fd, tmp_path = tempfile.mkstemp(
        suffix=".srt",
        prefix=f".{output_path.stem}.extract-",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp = Path(tmp_path)

    try:
        # Ensure ffmpeg sees a non-existent output path for -n.
        with contextlib.suppress(OSError):
            tmp.unlink()
        try:
            result = _run_ffmpeg(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-i",
                    str(video_path),
                    "-map",
                    # subtitle_stream_index is the zero-based index among subtitle
                    # streams, matching ffmpeg's 0:s:N mapping.
                    f"0:s:{track.subtitle_stream_index}",
                    "-c:s",
                    "srt",
                    str(tmp),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise ExtractionFailedError(f"ffmpeg not found: {exc}") from exc
        except subprocess.TimeoutExpired:
            raise ExtractionFailedError(
                f"ffmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s"
            ) from None
        except OSError as exc:
            raise ExtractionFailedError(f"ffmpeg could not be executed: {exc}") from exc

        if result.returncode != 0:
            raise ExtractionFailedError(f"ffmpeg failed: {result.stderr.strip()}")

        try:
            entries = load_subtitles(tmp)
        except SubtitleError as exc:
            raise ExtractionFailedError(f"extracted subtitle is invalid: {exc}") from exc

        if not entries:
            raise ExtractionFailedError("extracted subtitle is empty")

        atomic_write_path(entries, output_path, force=False)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _add_outcome(
    result: ExtractionResult,
    movie: MovieCoverage,
    video_path: Path | None,
    status: str,
    reason: str,
    output_path: Path | None = None,
    track: EmbeddedSubtitleInventory | None = None,
) -> None:
    """Append an outcome and update aggregate counters."""
    result.outcomes.append(
        ExtractionOutcome(
            movie_dir=movie.movie_dir,
            movie_name=movie.movie_name,
            video_path=video_path,
            status=status,
            reason=reason,
            output_path=output_path,
            track=track,
        )
    )
    if status == ExtractionStatus.SKIPPED:
        result.skipped += 1
        result.skipped_reasons[reason] += 1
    elif status == ExtractionStatus.FAILED:
        result.failures += 1


def extract_embedded(
    path: Path,
    *,
    dry_run: bool = False,
    require_mount: bool = False,
    mount_point: Path | None = None,
) -> ExtractionResult:
    """Extract eligible embedded English subtitles to external .en.srt files.

    ``path`` may be a library root, a single movie directory, or a single video
    file. When a video file is given, only that video is probed and only
    external SRT files in the same directory are considered.
    """
    if path.is_symlink():
        raise ValueError(f"Path must not be a symlink: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension: {path.suffix} "
                f"(supported: {', '.join(sorted(VIDEO_EXTENSIONS))})"
            )
        coverage = scan_single_video(
            path,
            require_mount=require_mount,
            mount_point=mount_point,
        )
        target_video = path.resolve()
    else:
        coverage = scan_coverage(
            path,
            require_mount=require_mount,
            mount_point=mount_point,
        )
        target_video = None

    result = ExtractionResult(
        movies_scanned=0,
        eligible_tracks=0,
        performed=0,
        skipped=0,
        failures=0,
        dry_run=dry_run,
    )

    target_set = {target_video} if target_video is not None else None

    for movie in coverage.movies:
        if target_set is not None:
            if target_video not in set(movie.video_paths):
                continue
            videos_to_process = (target_video,)
        else:
            videos_to_process = movie.video_paths

        result.movies_scanned += 1

        if movie.probe_error:
            for video_path in videos_to_process:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.SKIPPED,
                    ExtractionSkipReason.PROBE_ERROR,
                )
            continue

        if movie.ambiguous_reasons:
            for video_path in videos_to_process:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.SKIPPED,
                    ExtractionSkipReason.AMBIGUOUS,
                )
            continue

        if _has_appropriate_external_english(movie):
            for video_path in videos_to_process:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.SKIPPED,
                    ExtractionSkipReason.HAS_EXTERNAL_ENGLISH,
                )
            continue

        for video_path in videos_to_process:
            track = select_eligible_track(movie.embedded_subtitles, video_path.name)
            if track is None:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.SKIPPED,
                    ExtractionSkipReason.NO_ELIGIBLE_TRACK,
                )
                continue

            result.eligible_tracks += 1
            output_path = _target_output_path(video_path)

            if _target_exists(output_path):
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.SKIPPED,
                    ExtractionSkipReason.OUTPUT_EXISTS,
                    output_path=output_path,
                    track=track,
                )
                continue

            if dry_run:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.WOULD_EXTRACT,
                    "",
                    output_path=output_path,
                    track=track,
                )
                result.performed += 1
                continue

            try:
                _extract_track(video_path, track, output_path)
            except OutputExistsError:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.SKIPPED,
                    ExtractionSkipReason.OUTPUT_EXISTS,
                    output_path=output_path,
                    track=track,
                )
            except ExtractionFailedError as exc:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.FAILED,
                    str(exc),
                    output_path=output_path,
                    track=track,
                )
            else:
                _add_outcome(
                    result,
                    movie,
                    video_path,
                    ExtractionStatus.EXTRACTED,
                    "",
                    output_path=output_path,
                    track=track,
                )
                result.performed += 1

    return result
