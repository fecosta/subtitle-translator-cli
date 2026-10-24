from __future__ import annotations

from .models import Batch, SubtitleEntry

DEFAULT_CONTEXT_WINDOW = 5


def make_batches(
    entries: list[SubtitleEntry],
    batch_size: int,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> list[Batch]:
    """Split entries into overlapping batches with surrounding context."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total = len(entries)
    batches: list[Batch] = []
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        ctx_start = max(0, start - context_window)
        ctx_end = min(total, end + context_window)

        batch_entries = [entries[i] for i in range(ctx_start, ctx_end)]
        target_ids = {entries[i].id for i in range(start, end)}
        batches.append(Batch(entries=batch_entries, target_ids=target_ids))
        start = end

    return batches
