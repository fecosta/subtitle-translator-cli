from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .config import DEFAULT_BATCH_SIZE, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG, get_settings
from .debug import translate_batch_by_id, translate_batch_by_range
from .exceptions import (
    ConfigurationError,
    FileConflictError,
    OutputExistsError,
    SubtitleError,
    TranslationError,
)
from .progress import (
    ProgressReporter,
    print_dry_run,
    print_header,
    print_summary,
)
from .translator import TranslationStats, _default_output_path, get_dry_run_info, translate

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

    output_path = output or _default_output_path(input_file, to)

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
