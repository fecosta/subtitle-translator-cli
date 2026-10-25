from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from .batch_runner import BatchResult, build_batch_plan, execute_batch
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    get_scan_settings,
    get_settings,
)
from .coverage import (
    CoverageResult,
    CoverageStatus,
    MovieCoverage,
    coverage_result_to_dict,
    scan_coverage,
)
from .debug import translate_batch_by_id, translate_batch_by_range
from .exceptions import (
    ConfigurationError,
    FileConflictError,
    OutputExistsError,
    SubtitleError,
    TranslationError,
)
from .naming import default_output_path
from .progress import (
    ProgressReporter,
    print_dry_run,
    print_header,
    print_summary,
)
from .scanner import ScanStatus, scan_library
from .translator import TranslationStats, get_dry_run_info, translate

app = typer.Typer(name="subtranslate", help="Translate SRT subtitles with OpenAI")
debug_app = typer.Typer(name="debug", help="Debug helpers for subtitle translation")
app.add_typer(debug_app)
console = Console(stderr=True)


@app.callback()
def _root() -> None:
    """Root callback to preserve the 'subtranslate translate' command form."""


@app.command("translate")
def translate_command(
    input_file: Path = typer.Argument(  # noqa: B008
        ..., exists=True, readable=True, help="Input SRT file"
    ),
    to: str = typer.Option(DEFAULT_TARGET_LANG, "--to", help="Target language"),  # noqa: B008
    from_: str = typer.Option(  # noqa: B008
        DEFAULT_SOURCE_LANG, "--from", help="Source language", show_default=True
    ),
    model: str | None = typer.Option(None, "--model", help="OpenAI model"),  # noqa: B008
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output SRT file"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without API calls"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Overwrite existing output"),  # noqa: B008
    verbose: bool = typer.Option(False, "--verbose", help="Show verbose output"),  # noqa: B008
    batch_size: int = typer.Option(  # noqa: B008
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Number of subtitles per translation batch",
        min=1,
        max=200,
    ),
) -> None:
    """Translate an SRT subtitle file into another language."""
    try:
        settings = get_settings(
            model=model,
            source_lang=from_,
            target_lang=to,
            batch_size=batch_size,
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        output_path = output if output is not None else default_output_path(input_file, to)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    print_header(input_file, from_, to, settings.model)

    if dry_run:
        info = get_dry_run_info(input_file, settings)
        print_dry_run(info)
        print_summary(
            TranslationStats(
                duration_seconds=0.0,
                total_entries=info.total_entries,
                batch_count=info.batch_count,
                dry_run=True,
            ),
            output_path,
        )
        raise typer.Exit()

    reporter = ProgressReporter()
    reporter.start(100)

    def on_batch(current: int, total: int) -> None:
        reporter.update(current, total)

    try:
        stats = translate(
            input_file,
            output_path,
            settings,
            force=force,
            verbose=verbose,
            progress_callback=on_batch,
        )
    except OutputExistsError as exc:
        reporter.stop()
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except FileConflictError as exc:
        reporter.stop()
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    except SubtitleError as exc:
        reporter.stop()
        console.print(f"[red]Subtitle error:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    except TranslationError as exc:
        reporter.stop()
        console.print(f"[red]Translation error:[/red] {exc}")
        raise typer.Exit(code=4) from exc
    except Exception as exc:
        reporter.stop()
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        reporter.stop()

    print_summary(stats, output_path)


@app.command("scan")
def scan_command(
    library_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root directory of the movie library",
    ),
    to: str = typer.Option(DEFAULT_TARGET_LANG, "--to", help="Target language"),  # noqa: B008
    from_: str = typer.Option(  # noqa: B008
        DEFAULT_SOURCE_LANG,
        "--from",
        help="Source language",
        show_default=True,
    ),
    model: str | None = typer.Option(None, "--model", help="OpenAI model"),  # noqa: B008
    batch_size: int = typer.Option(  # noqa: B008
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Number of subtitles per translation batch",
        min=1,
        max=200,
    ),
    require_mount: bool = typer.Option(  # noqa: B008
        False,
        "--require-mount",
        help="Require the library path to be under a mounted filesystem",
    ),
    include_variants: bool = typer.Option(  # noqa: B008
        False,
        "--include-variants",
        help="Include forced/sdh/hi/cc variants as translation sources",
    ),
) -> None:
    """Recursively scan a library and produce a read-only translation plan."""
    try:
        settings = get_scan_settings(
            model=model,
            source_lang=from_,
            target_lang=to,
            batch_size=batch_size,
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        result = scan_library(
            library_path,
            settings,
            require_mount=require_mount,
            include_variants=include_variants,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print("[bold]Subtitle Library Scan[/bold]\n")

    summary = Table(show_header=False)
    summary.add_row("Library:", str(library_path.resolve()))
    summary.add_row("Language:", f"{from_} -> {to}")
    summary.add_row("Model:", settings.model)
    summary.add_row("Directories scanned:", f"{result.directories_scanned}")
    summary.add_row("English subtitle files:", f"{result.english_files}")
    summary.add_row("Portuguese subtitle files:", f"{result.portuguese_files}")
    summary.add_row("Eligible translations:", f"{result.eligible_count}")
    summary.add_row("Skipped:", f"{result.skipped_count}")
    summary.add_row("Ambiguous:", f"{result.ambiguous_count}")
    summary.add_row("Errors:", f"{result.error_count}")
    summary.add_row(
        "Estimated input tokens:",
        f"{result.total_estimated_input_tokens:,}",
    )
    summary.add_row(
        "Estimated output tokens:",
        f"{result.total_estimated_output_tokens:,}",
    )
    if result.total_estimated_cost_usd is not None:
        summary.add_row("Estimated total cost:", f"${result.total_estimated_cost_usd:.4f}")
    else:
        summary.add_row("Estimated total cost:", "unknown (no pricing configured)")
    console.print(summary)
    console.print()

    table = Table(title="Per-movie decisions")
    table.add_column("Movie", no_wrap=True)
    table.add_column("Status")
    table.add_column("Reason", overflow="fold")

    def _movie_label(item) -> str:
        try:
            path = item.source or item.movie_dir / item.movie_name
            return str(path.resolve().relative_to(library_path.resolve()))
        except ValueError:
            return str(item.source or item.movie_dir / item.movie_name)

    for item in result.items:
        status_color = {
            ScanStatus.TRANSLATE: "green",
            ScanStatus.SKIP: "yellow",
            ScanStatus.AMBIGUOUS: "magenta",
            ScanStatus.ERROR: "red",
        }.get(item.status, "white")
        table.add_row(
            _movie_label(item),
            f"[{status_color}]{item.status}[/{status_color}]",
            item.reason,
        )
    console.print(table)


@app.command("coverage")
def coverage_command(
    library_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root directory of the movie library",
    ),
    missing_only: bool = typer.Option(  # noqa: B008
        False,
        "--missing-only",
        help="Show only movies without confirmed Brazilian Portuguese subtitles",
    ),
    output_format: Literal["text", "json"] = typer.Option(  # noqa: B008
        "text",
        "--format",
        help="Output format",
    ),
    require_mount: bool = typer.Option(  # noqa: B008
        False,
        "--require-mount",
        help="Require the library path to be under a mounted filesystem",
    ),
) -> None:
    """Recursively inspect a library and report external SRT coverage."""
    is_json = output_format == "json"

    try:
        result = scan_coverage(library_path, require_mount=require_mount)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        if is_json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    movies = [
        movie
        for movie in result.movies
        if not missing_only or movie.category != CoverageStatus.HAS_PT_BR
    ]

    if is_json:
        payload = coverage_result_to_dict(result, library_path)
        if missing_only:
            payload["movies"] = [
                m for m in payload["movies"] if m["category"] != CoverageStatus.HAS_PT_BR
            ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    _print_coverage_table(result, library_path, movies)


def _print_coverage_table(
    result: CoverageResult,
    library_path: Path,
    movies: list[MovieCoverage],
) -> None:
    console.print("[bold]Subtitle Coverage (external SRT only)[/bold]\n")

    summary = Table(show_header=False)
    all_movies = result.movies
    summary.add_row("Library:", str(library_path.resolve()))
    summary.add_row("Directories scanned:", f"{result.directories_scanned}")
    summary.add_row("Movies discovered:", f"{len(all_movies)}")
    summary.add_row("Movies shown:", f"{len(movies)}")
    summary.add_row(
        "Has Brazilian Portuguese:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.HAS_PT_BR)}",
    )
    summary.add_row(
        "Has generic Portuguese:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.HAS_GENERIC_PT)}",
    )
    summary.add_row(
        "Has English only:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.HAS_EN_ONLY)}",
    )
    summary.add_row(
        "Has unlabeled subtitles:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.HAS_UNLABELED)}",
    )
    summary.add_row(
        "Has specialized variants only:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.HAS_SPECIALIZED_ONLY)}",
    )
    summary.add_row(
        "No external subtitles:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.NO_EXTERNAL_SUBS)}",
    )
    summary.add_row(
        "Ambiguous:",
        f"{sum(1 for m in all_movies if m.category == CoverageStatus.AMBIGUOUS)}",
    )
    summary.add_row("Errors:", f"{len(result.errors)}")
    console.print(summary)
    console.print()

    table = Table(title="Per-movie coverage")
    table.add_column("Movie", no_wrap=True, overflow="fold")
    table.add_column("Category")
    table.add_column("Subtitles", overflow="fold")

    resolved_library = library_path.resolve()

    for movie in movies:
        status_color = {
            CoverageStatus.HAS_PT_BR: "green",
            CoverageStatus.HAS_GENERIC_PT: "cyan",
            CoverageStatus.HAS_EN_ONLY: "blue",
            CoverageStatus.HAS_UNLABELED: "yellow",
            CoverageStatus.HAS_SPECIALIZED_ONLY: "bright_black",
            CoverageStatus.NO_EXTERNAL_SUBS: "red",
            CoverageStatus.AMBIGUOUS: "magenta",
        }.get(movie.category, "white")

        try:
            label = str(
                (movie.movie_dir / movie.movie_name).resolve().relative_to(
                    resolved_library
                )
            )
        except ValueError:
            label = str(movie.movie_dir / movie.movie_name)

        subtitle_label = _format_subtitle_inventory(movie.subtitles)
        table.add_row(
            label,
            f"[{status_color}]{movie.category}[/{status_color}]",
            subtitle_label,
        )

    console.print(table)
    console.print(
        "\n[yellow]Note:[/yellow] This report covers external SRT files only. "
        "Movies without external subtitles may still contain embedded subtitles."
    )


def _format_subtitle_inventory(subtitles: tuple) -> str:
    if not subtitles:
        return "-"
    parts = []
    for sub in subtitles:
        tag = sub.language_tag or "unlabeled"
        if sub.attributes:
            tag += "." + ".".join(sub.attributes)
        parts.append(tag)
    return ", ".join(parts)


class CoverageResultSummary:
    @staticmethod
    def labels() -> tuple[str, ...]:
        return (
            CoverageStatus.HAS_PT_BR,
            CoverageStatus.HAS_GENERIC_PT,
            CoverageStatus.HAS_EN_ONLY,
            CoverageStatus.HAS_UNLABELED,
            CoverageStatus.HAS_SPECIALIZED_ONLY,
            CoverageStatus.NO_EXTERNAL_SUBS,
            CoverageStatus.AMBIGUOUS,
        )


@app.command("batch")
def batch_command(
    library_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Root directory of the movie library",
    ),
    to: str = typer.Option(DEFAULT_TARGET_LANG, "--to", help="Target language"),  # noqa: B008
    from_: str = typer.Option(  # noqa: B008
        DEFAULT_SOURCE_LANG,
        "--from",
        help="Source language",
        show_default=True,
    ),
    model: str | None = typer.Option(None, "--model", help="OpenAI model"),  # noqa: B008
    batch_size: int = typer.Option(  # noqa: B008
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Number of subtitles per translation batch",
        min=1,
        max=200,
    ),
    require_mount: bool = typer.Option(  # noqa: B008
        False,
        "--require-mount",
        help="Require the library path to be under a mounted filesystem",
    ),
    include_variants: bool = typer.Option(  # noqa: B008
        False,
        "--include-variants",
        help="Include forced/sdh/hi/cc variants as translation sources",
    ),
    max_cost_usd: float | None = typer.Option(  # noqa: B008
        None,
        "--max-cost-usd",
        help="Maximum estimated API spend for this batch",
    ),
    yes: bool = typer.Option(  # noqa: B008
        False,
        "--yes",
        help="Run without interactive confirmation",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Show verbose output"),  # noqa: B008
) -> None:
    """Scan a library and translate all eligible subtitles."""
    try:
        settings = get_settings(
            model=model,
            source_lang=from_,
            target_lang=to,
            batch_size=batch_size,
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        plan = build_batch_plan(
            library_path,
            settings,
            require_mount=require_mount,
            include_variants=include_variants,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print("[bold]Batch Translation Plan[/bold]\n")
    console.print(f"Library: {library_path.resolve()}")
    console.print(f"Language: {from_} -> {to}")
    console.print(f"Model: {settings.model}")
    console.print(f"Eligible files: {plan.eligible_count}")
    console.print(f"Skipped: {plan.skipped_count}")
    console.print(f"Ambiguous: {plan.ambiguous_count}")
    console.print(f"Errors: {plan.error_count}")
    console.print(
        f"Estimated input tokens: {plan.total_estimated_input_tokens:,}"
    )
    console.print(
        f"Estimated output tokens: {plan.total_estimated_output_tokens:,}"
    )
    if plan.total_estimated_cost_usd is not None:
        console.print(
            f"Estimated total cost: ${plan.total_estimated_cost_usd:.4f}"
        )
    else:
        console.print("Estimated total cost: unknown (no pricing configured)")
    console.print()

    if plan.eligible_count == 0:
        console.print("[yellow]No eligible translations.[/yellow]")
        raise typer.Exit(code=1 if plan.error_count else 0)

    if max_cost_usd is not None:
        if plan.total_estimated_cost_usd is None:
            console.print(
                "[red]Error:[/red] Cost ceiling set but model pricing is unknown."
            )
            raise typer.Exit(code=1)
        if plan.total_estimated_cost_usd > max_cost_usd:
            console.print(
                f"[red]Error:[/red] Estimated cost "
                f"${plan.total_estimated_cost_usd:.4f} exceeds ceiling "
                f"${max_cost_usd:.4f}."
            )
            raise typer.Exit(code=1)

    if not yes:
        try:
            confirmed = typer.confirm("Start translation?")
        except typer.Abort:
            confirmed = False
        if not confirmed:
            console.print("[yellow]Batch cancelled.[/yellow]")
            raise typer.Exit()

    try:
        result = execute_batch(
            library_path,
            settings,
            plan,
            max_cost_usd=max_cost_usd,
            require_mount=require_mount,
            include_variants=include_variants,
            verbose=verbose,
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    _print_batch_summary(result)
    if result.failed or result.budget_exceeded or result.errors:
        raise typer.Exit(code=1)


def _print_batch_summary(result: BatchResult) -> None:
    console.print("\n[bold]Batch Translation Complete[/bold]\n")
    table = Table(show_header=False)
    eligible_files = (
        result.translated + result.skipped + result.failed + result.budget_exceeded
    )
    table.add_row("Eligible files:", f"{eligible_files}")
    table.add_row("Translated:", f"{result.translated}")
    table.add_row("Skipped:", f"{result.skipped}")
    table.add_row("Failed:", f"{result.failed}")
    if result.ambiguous:
        table.add_row("Ambiguous:", f"{result.ambiguous}")
    if result.errors:
        table.add_row("Errors:", f"{result.errors}")
    if result.budget_exceeded:
        table.add_row("Budget exceeded:", f"{result.budget_exceeded}")
    table.add_row("Input tokens:", f"{result.total_input_tokens:,}")
    table.add_row("Output tokens:", f"{result.total_output_tokens:,}")
    if result.total_estimated_cost_usd is not None:
        table.add_row("Estimated cost:", f"${result.total_estimated_cost_usd:.4f}")
    else:
        table.add_row("Estimated cost:", "unknown")
    table.add_row("Duration:", f"{result.duration_seconds:.1f}s")
    console.print(table)

    if any(
        item.status in {"failed", "budget_exceeded", "planned_error"}
        for item in result.items
    ):
        console.print("\n[bold]Per-file issues[/bold]")
        issues = Table("Movie", "Status", "Reason")
        for item in result.items:
            if item.status in {"failed", "budget_exceeded", "planned_error"}:
                issues.add_row(
                    str(item.source),
                    item.status,
                    item.reason,
                )
        console.print(issues)


@debug_app.command("batch-by-id")
def debug_batch_by_id_command(
    input_file: Path = typer.Argument(  # noqa: B008
        ..., exists=True, readable=True, help="Input SRT file"
    ),
    subtitle_id: int = typer.Argument(..., help="Subtitle id to debug"),
    to: str = typer.Option(DEFAULT_TARGET_LANG, "--to", help="Target language"),  # noqa: B008
    from_: str = typer.Option(  # noqa: B008
        DEFAULT_SOURCE_LANG, "--from", help="Source language", show_default=True
    ),
    model: str | None = typer.Option(None, "--model", help="OpenAI model"),  # noqa: B008
    batch_size: int = typer.Option(  # noqa: B008
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Number of subtitles per translation batch",
        min=1,
        max=200,
    ),
) -> None:
    """Translate a single batch containing the given subtitle id."""
    try:
        settings = get_settings(
            model=model,
            source_lang=from_,
            target_lang=to,
            batch_size=batch_size,
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        batch, result = translate_batch_by_id(input_file, settings, subtitle_id)
    except Exception as exc:
        console.print(f"[red]Debug error:[/red] {exc}")
        raise typer.Exit(code=4) from exc

    print(json.dumps({
        "target_ids": sorted(batch.target_ids),
        "translations": result.translations,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
    }, ensure_ascii=False, indent=2))


@debug_app.command("batch-by-range")
def debug_batch_by_range_command(
    input_file: Path = typer.Argument(  # noqa: B008
        ..., exists=True, readable=True, help="Input SRT file"
    ),
    start_id: int = typer.Argument(..., help="Start subtitle id"),
    end_id: int = typer.Argument(..., help="End subtitle id"),
    to: str = typer.Option(DEFAULT_TARGET_LANG, "--to", help="Target language"),  # noqa: B008
    from_: str = typer.Option(  # noqa: B008
        DEFAULT_SOURCE_LANG, "--from", help="Source language", show_default=True
    ),
    model: str | None = typer.Option(None, "--model", help="OpenAI model"),  # noqa: B008
    batch_size: int = typer.Option(  # noqa: B008
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        help="Number of subtitles per translation batch",
        min=1,
        max=200,
    ),
) -> None:
    """Translate the first batch fully contained in the given id range."""
    try:
        settings = get_settings(
            model=model,
            source_lang=from_,
            target_lang=to,
            batch_size=batch_size,
        )
    except ConfigurationError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        batch, result = translate_batch_by_range(input_file, settings, start_id, end_id)
    except Exception as exc:
        console.print(f"[red]Debug error:[/red] {exc}")
        raise typer.Exit(code=4) from exc

    print(json.dumps({
        "target_ids": sorted(batch.target_ids),
        "translations": result.translations,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
    }, ensure_ascii=False, indent=2))
