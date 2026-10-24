from __future__ import annotations

from subtranslate.batching import make_batches
from subtranslate.models import SubtitleEntry


def _entries(count: int) -> list[SubtitleEntry]:
    return [
        SubtitleEntry(id=i + 1, start=i * 1000, end=(i + 1) * 1000, text=f"Line {i + 1}")
        for i in range(count)
    ]


def test_batch_preserves_order_and_ids() -> None:
    entries = _entries(10)
    batches = make_batches(entries, batch_size=3)
    all_target_ids = []
    for batch in batches:
        all_target_ids.extend(sorted(batch.target_ids))
    assert all_target_ids == list(range(1, 11))


def test_context_entries_overlap() -> None:
    entries = _entries(10)
    batches = make_batches(entries, batch_size=3, context_window=2)
    # First batch should include context after target.
    assert {1, 2, 3} == batches[0].target_ids
    assert 4 in batches[0].all_ids
    assert 5 in batches[0].all_ids
    # Last batch should include context before target.
    last = batches[-1]
    assert 10 in last.target_ids
    assert 8 in last.all_ids
    assert 9 in last.all_ids


def test_no_context_duplication_in_targets() -> None:
    entries = _entries(10)
    batches = make_batches(entries, batch_size=3, context_window=2)
    seen_targets: set[int] = set()
    for batch in batches:
        assert batch.target_ids.isdisjoint(seen_targets)
        seen_targets.update(batch.target_ids)
    assert seen_targets == set(range(1, 11))
