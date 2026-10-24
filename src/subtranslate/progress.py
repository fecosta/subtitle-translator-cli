from __future__ import annotations

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .translator import DryRunInfo, TranslationStats

console = Console()


class ProgressReporter:
    def __init__(self) -> None:
        self.progress = Progress(
            TextColumn("[bold blue]Translating..."),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.task_id = None

    def start(self, total: int) -> None:
        self.progress.start()
        self.task_id = self.progress.add_task("translate", total=total)

    def update(self, completed: int, total: int) -> None:
        if self.task_id is not None:
            self.progress.update(self.task_id, completed=completed, total=total)

    def stop(self) -> None:
        self.progress.stop()


def print_header(input_path, source_lang: str, target_lang: str, model: str) -> None:
    console.print("[bold]Subtitle Translator CLI[/bold]")
    console.print()
    console.print(f"Input: {input_path}")
    console.print(f"Language: {source_lang} -> {target_lang}")
    console.print(f"Model: {model}")
    console.print()


def print_summary(stats: TranslationStats, output_path) -> None:
    if stats.dry_run:
        status = "[bold yellow]DRY RUN[/bold yellow]"
    else:
        status = "[bold green]PASSED[/bold green]"
    table = Table(show_header=False)
    table.add_row("Subtitles:", f"{stats.total_entries:,}")
    table.add_row("Batches:", f"{stats.batch_count:,}")
    if not stats.dry_run:
        table.add_row("Validation:", status)
        table.add_row("Output:", str(output_path))
        table.add_row("Duration:", f"{stats.duration_seconds:.1f} seconds")
        if stats.input_tokens:
            table.add_row("Input tokens:", f"{stats.input_tokens:,}")
        if stats.output_tokens:
            table.add_row("Output tokens:", f"{stats.output_tokens:,}")
        if stats.estimated_cost_usd is not None:
            table.add_row("Estimated cost:", f"${stats.estimated_cost_usd:.4f}")
    else:
        table.add_row("Mode:", "Dry run (no API calls)")
    console.print(table)


def print_dry_run(info: DryRunInfo) -> None:
    console.print(f"Subtitles: {info.total_entries:,}")
    console.print(f"Batches: {info.batch_count:,}")
    console.print(f"Estimated input tokens: {info.estimated_input_tokens:,}")
    console.print(f"Estimated output tokens: {info.estimated_output_tokens:,}")
    if info.estimated_cost_usd is not None:
        console.print(f"Estimated cost: ${info.estimated_cost_usd:.4f}")
    else:
        console.print("Estimated cost: unknown (no pricing configured)")
