from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SubtitleEntry:
    """A single subtitle entry with a stable identifier."""

    id: int
    start: int  # milliseconds
    end: int  # milliseconds
    text: str


@dataclass(frozen=True, slots=True)
class Batch:
    """A translation batch containing target entries and optional context."""

    entries: list[SubtitleEntry]
    target_ids: set[int]

    @property
    def all_ids(self) -> set[int]:
        return {entry.id for entry in self.entries}


@dataclass(slots=True)
class Usage:
    """Token usage returned by the API."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class TranslationResult:
    """Result of translating one batch."""

    translations: dict[int, str] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
