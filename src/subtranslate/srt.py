from __future__ import annotations

import re
from pathlib import Path

import pysubs2

from .exceptions import SubtitleError
from .models import SubtitleEntry

_SRT_INDEX_RE = re.compile(
    r"^(\d+)\s*\n\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}",
    re.MULTILINE,
)


def _read_srt_indices(path: Path) -> list[int]:
    """Read the numeric index that precedes each SRT cue block."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        raise SubtitleError(f"Could not read subtitle file: {exc}") from exc
    return [int(match.group(1)) for match in _SRT_INDEX_RE.finditer(text)]


def load_subtitles(path: Path) -> list[SubtitleEntry]:
    """Load an SRT file and return a list of stable subtitle entries."""
    try:
        subs = pysubs2.load(str(path))
    except Exception as exc:
        raise SubtitleError(f"Could not parse subtitle file: {exc}") from exc

    if not subs.events:
        raise SubtitleError("Subtitle file contains no entries.")

    indices = _read_srt_indices(path)
    if len(indices) != len(subs.events) or len(set(indices)) != len(indices):
        indices = list(range(1, len(subs.events) + 1))

    return [
        SubtitleEntry(
            id=indices[index],
            start=event.start,
            end=event.end,
            text=event.text,
        )
        for index, event in enumerate(subs.events)
    ]


def save_subtitles(entries: list[SubtitleEntry], path: Path) -> None:
    """Write subtitle entries to an SRT file."""
    subs = pysubs2.SSAFile()
    for entry in entries:
        event = pysubs2.SSAEvent(start=entry.start, end=entry.end, text=entry.text)
        subs.events.append(event)
    try:
        subs.save(str(path), format="srt", encoding="utf-8")
    except Exception as exc:
        raise SubtitleError(f"Could not write subtitle file: {exc}") from exc
