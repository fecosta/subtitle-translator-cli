from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .exceptions import SubtitleError
from .naming import SubtitleName, parse_subtitle_name
from .translator import DryRunInfo, get_dry_run_info


class ScanStatus:
    TRANSLATE = "TRANSLATE"
    SKIP = "SKIP"
    AMBIGUOUS = "AMBIGUOUS"
    ERROR = "ERROR"


@dataclass(slots=True)
class ScanItem:
    """A single movie-level scanning decision."""

    source: Path | None
    movie_dir: Path
    movie_name: str
    output_path: Path | None
    status: str
    reason: str
    dry_run_info: DryRunInfo | None = None


@dataclass(slots=True)
class ScanResult:
    """Aggregate results of scanning a library."""

    items: list[ScanItem] = field(default_factory=list)
    directories_scanned: int = 0
    english_files: int = 0
    portuguese_files: int = 0
    eligible_count: int = 0
    skipped_count: int = 0
    ambiguous_count: int = 0
    error_count: int = 0
    total_estimated_input_tokens: int = 0
    total_estimated_output_tokens: int = 0
    total_estimated_cost_usd: float | None = None


def _find_srt_files(
    library_path: Path,
    walk_errors: list[tuple[Path, str]],
) -> tuple[dict[tuple[Path, str], list[SubtitleName]], int]:
    """Group SRT files by (parent directory, movie name), skipping symlinks."""
    groups: dict[tuple[Path, str], list[SubtitleName]] = defaultdict(list)
    directories: set[Path] = set()

    def on_error(error: OSError) -> None:
        walk_errors.append((Path(error.filename or ""), str(error)))

    for root, _dirs, files in os.walk(library_path, followlinks=False, onerror=on_error):
        root_path = Path(root)
        directories.add(root_path)
        for filename in files:
            if not filename.lower().endswith(".srt"):
                continue
            file_path = root_path / filename
            if file_path.is_symlink():
                continue
            try:
                parsed = parse_subtitle_name(file_path)
            except Exception as exc:  # noqa: BLE001
                walk_errors.append((file_path, str(exc)))
                continue
            groups[(root_path, parsed.movie_name)].append(parsed)

    return groups, len(directories)


def _classify_group(
    names: list[SubtitleName],
    include_variants: bool,
) -> tuple[list[SubtitleName], list[SubtitleName], list[SubtitleName]]:
    """Return (english candidates, full portuguese, variant portuguese)."""
    english: list[SubtitleName] = []
    portuguese: list[SubtitleName] = []
    portuguese_variants: list[SubtitleName] = []
    for name in names:
        if name.is_target_language:
            if name.has_specialized_variant:
                portuguese_variants.append(name)
            else:
                portuguese.append(name)
        elif name.is_source_language and (
            include_variants or not name.has_specialized_variant
        ):
            english.append(name)
    return english, portuguese, portuguese_variants


def _build_skip_item(
    movie_dir: Path,
    movie_name: str,
    reason: str,
) -> ScanItem:
    return ScanItem(
        source=None,
        movie_dir=movie_dir,
        movie_name=movie_name,
        output_path=None,
        status=ScanStatus.SKIP,
        reason=reason,
    )


def _cost_from_dry_run(dry_run_info: DryRunInfo | None) -> tuple[float | None, bool]:
    if dry_run_info is None:
        return None, True
    return dry_run_info.estimated_cost_usd, dry_run_info.estimated_cost_usd is None


def scan_library(
    library_path: Path,
    settings: Settings,
    *,
    require_mount: bool = False,
    include_variants: bool = False,
) -> ScanResult:
    """Recursively scan a library and build a read-only translation plan."""
    if library_path.is_symlink():
        raise ValueError(f"Library path must not be a symlink: {library_path}")
    library_path = library_path.resolve()
    if not library_path.exists():
        raise FileNotFoundError(f"Library path does not exist: {library_path}")
    if not library_path.is_dir():
        raise NotADirectoryError(f"Library path is not a directory: {library_path}")
    if require_mount and not os.path.ismount(library_path):
        raise FileNotFoundError(
            f"Library path is not a mounted filesystem: {library_path}"
        )

    walk_errors: list[tuple[Path, str]] = []
    groups, directory_count = _find_srt_files(library_path, walk_errors)

    result = ScanResult(directories_scanned=directory_count)

    total_cost = 0.0
    cost_unknown = False

    for (movie_dir, movie_name), names in sorted(groups.items()):
        english, portuguese, portuguese_variants = _classify_group(
            names, include_variants
        )
        result.english_files += len([n for n in names if n.is_source_language])
        result.portuguese_files += len(portuguese) + len(portuguese_variants)

        if portuguese:
            pt = portuguese[0]
            reason = "Portuguese already exists"
            if pt.is_brazilian_portuguese:
                reason = "Brazilian Portuguese subtitle already exists"
            elif pt.is_generic_portuguese:
                reason = "Generic Portuguese subtitle already exists"
            result.items.append(_build_skip_item(movie_dir, movie_name, reason))
            result.skipped_count += 1
            continue

        if not english:
            if any(
                name.is_source_language and name.has_specialized_variant
                for name in names
            ):
                reason = "English source is a specialized variant (forced/sdh/hi/cc)"
            else:
                reason = "No recognized English subtitle"
            result.items.append(_build_skip_item(movie_dir, movie_name, reason))
            result.skipped_count += 1
            continue

        if len(english) > 1:
            result.items.append(
                ScanItem(
                    source=None,
                    movie_dir=movie_dir,
                    movie_name=movie_name,
                    output_path=None,
                    status=ScanStatus.AMBIGUOUS,
                    reason="Multiple English candidates",
                )
            )
            result.ambiguous_count += 1
            continue

        source = english[0]
        output_path = source.output_path(settings.target_lang)
        if output_path.exists():
            result.items.append(
                _build_skip_item(
                    movie_dir, movie_name, "Output file already exists"
                )
            )
            result.skipped_count += 1
            continue

        dry_run_info: DryRunInfo | None = None
        try:
            dry_run_info = get_dry_run_info(source.path, settings)
        except (SubtitleError, OSError) as exc:
            result.items.append(
                ScanItem(
                    source=source.path,
                    movie_dir=movie_dir,
                    movie_name=movie_name,
                    output_path=output_path,
                    status=ScanStatus.ERROR,
                    reason=f"Invalid SRT: {exc}",
                )
            )
            result.error_count += 1
            continue

        cost, unknown = _cost_from_dry_run(dry_run_info)
        if unknown:
            cost_unknown = True
        elif cost is not None:
            total_cost += cost

        result.items.append(
            ScanItem(
                source=source.path,
                movie_dir=movie_dir,
                movie_name=movie_name,
                output_path=output_path,
                status=ScanStatus.TRANSLATE,
                reason="English found, Portuguese missing",
                dry_run_info=dry_run_info,
            )
        )
        result.eligible_count += 1
        if dry_run_info is not None:
            result.total_estimated_input_tokens += dry_run_info.estimated_input_tokens
            result.total_estimated_output_tokens += dry_run_info.estimated_output_tokens

    for file_path, reason in walk_errors:
        result.items.append(
            ScanItem(
                source=file_path,
                movie_dir=file_path.parent,
                movie_name=file_path.stem,
                output_path=None,
                status=ScanStatus.ERROR,
                reason=f"Invalid or unreadable file: {reason}",
            )
        )
        result.error_count += 1

    result.total_estimated_cost_usd = None if cost_unknown else total_cost
    return result
