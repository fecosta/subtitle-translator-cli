from __future__ import annotations

from pathlib import Path

import pytest

from subtranslate.exceptions import SubtitleError
from subtranslate.models import SubtitleEntry
from subtranslate.srt import load_subtitles, save_subtitles


def test_load_sample(sample_srt: Path) -> None:
    entries = load_subtitles(sample_srt)
    assert len(entries) == 5
    assert entries[0].id == 1
    assert entries[0].start == 1000
    assert entries[0].end == 3000
    assert entries[0].text == "Hello, how are you?"


def test_multiline_subtitle(sample_srt: Path) -> None:
    entries = load_subtitles(sample_srt)
    # pysubs2 stores multi-line text with internal \\N separator.
    assert "\\N" in entries[1].text
    assert "I'm fine, thanks." in entries[1].text
    assert "And you?" in entries[1].text


def test_preserve_unicode_and_formatting(sample_srt: Path, tmp_path: Path) -> None:
    entries = load_subtitles(sample_srt)
    assert "Café" in entries[4].text
    # pysubs2 stores SRT italic tags using ASS-style markers internally.
    assert "{\\i1}" in entries[2].text
    # After a save/load round-trip the original SRT tag is restored.
    out = tmp_path / "roundtrip.srt"
    save_subtitles(entries, out)
    assert "<i>This is italic</i>" in out.read_text()


def test_save_and_reload_preserves_data(tmp_path: Path) -> None:
    original = [
        SubtitleEntry(id=1, start=1234, end=5678, text="Test line one\\NTest line two"),
        SubtitleEntry(id=2, start=6000, end=9000, text="Café and São Paulo"),
    ]
    out = tmp_path / "out.srt"
    save_subtitles(original, out)
    loaded = load_subtitles(out)
    assert len(loaded) == len(original)
    for orig, new in zip(original, loaded, strict=True):
        assert orig.id == new.id
        assert orig.start == new.start
        assert orig.end == new.end
        assert orig.text == new.text


def test_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.srt"
    empty.write_text("")
    with pytest.raises(SubtitleError):
        load_subtitles(empty)


def test_malformed_srt_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.srt"
    bad.write_text("not a subtitle file")
    with pytest.raises(SubtitleError):
        load_subtitles(bad)


def test_load_preserves_high_srt_ids_despite_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.srt"
    path.write_bytes(
        b"\xef\xbb\xbf"
        b"1360\n00:00:01,000 --> 00:00:03,000\nBefore\n\n"
        b"1361\n00:00:03,000 --> 00:00:05,000\nTarget\n\n"
    )
    entries = load_subtitles(path)
    assert [entry.id for entry in entries] == [1360, 1361]


def test_load_falls_back_to_sequential_ids_on_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.srt"
    path.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nFirst\n\n"
        "1\n00:00:03,000 --> 00:00:05,000\nSecond\n\n"
    )
    entries = load_subtitles(path)
    assert [entry.id for entry in entries] == [1, 2]
