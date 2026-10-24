from __future__ import annotations

from pathlib import Path

import pytest

from subtranslate.exceptions import OutputExistsError
from subtranslate.fileio import atomic_write_path
from subtranslate.models import SubtitleEntry
from subtranslate.srt import load_subtitles


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    entries = [
        SubtitleEntry(id=1, start=1000, end=2000, text="Hello"),
    ]
    out = tmp_path / "output.srt"
    atomic_write_path(entries, out)
    assert out.exists()
    loaded = load_subtitles(out)
    assert len(loaded) == 1
    assert loaded[0].text == "Hello"


def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    entries = [
        SubtitleEntry(id=1, start=1000, end=2000, text="Hello"),
    ]
    out = tmp_path / "nested" / "dir" / "output.srt"
    atomic_write_path(entries, out)
    assert out.exists()


def test_atomic_write_refuses_existing_output(tmp_path: Path) -> None:
    entries = [
        SubtitleEntry(id=1, start=1000, end=2000, text="Hello"),
    ]
    out = tmp_path / "output.srt"
    out.write_text("existing")
    with pytest.raises(OutputExistsError):
        atomic_write_path(entries, out)


def test_atomic_write_overwrites_with_force(tmp_path: Path) -> None:
    entries = [
        SubtitleEntry(id=1, start=1000, end=2000, text="Hello"),
    ]
    out = tmp_path / "output.srt"
    out.write_text("existing")
    atomic_write_path(entries, out, force=True)
    loaded = load_subtitles(out)
    assert loaded[0].text == "Hello"
