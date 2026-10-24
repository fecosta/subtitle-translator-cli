from __future__ import annotations

import json
from pathlib import Path

from subtranslate.checkpoint import (
    build_identity,
    compute_source_hash,
    load_checkpoint,
    remove_checkpoint,
    save_checkpoint,
)
from subtranslate.config import Settings


def _settings() -> Settings:
    return Settings(
        api_key="sk-test",
        model="gpt-4.1-mini",
        source_lang="en",
        target_lang="pt-BR",
        batch_size=40,
    )


def test_compute_source_hash_is_stable_and_unique(tmp_path: Path) -> None:
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text("hello")
    second.write_text("world")

    assert compute_source_hash(first) == compute_source_hash(first)
    assert compute_source_hash(first) != compute_source_hash(second)


def test_build_identity_includes_translation_settings(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    settings = _settings()

    identity = build_identity(path, settings, compute_source_hash(path))
    assert identity["source_hash"] == compute_source_hash(path)
    assert identity["source_lang"] == settings.source_lang
    assert identity["target_lang"] == settings.target_lang
    assert identity["model"] == settings.model
    assert identity["batch_size"] == settings.batch_size


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("subtitle content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"

    save_checkpoint(cp, identity, {1: "Olá", 2: "mundo"})
    loaded = load_checkpoint(cp, identity, {1, 2, 3})

    assert loaded == {1: "Olá", 2: "mundo"}


def test_load_checkpoint_returns_empty_for_missing_file(tmp_path: Path) -> None:
    settings = _settings()
    identity = build_identity(tmp_path / "x.srt", settings, "abc")
    assert load_checkpoint(tmp_path / "missing.json", identity) == {}


def test_load_checkpoint_rejects_modified_source(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("original")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá"})

    path.write_text("modified")
    new_identity = build_identity(path, settings, compute_source_hash(path))
    assert load_checkpoint(cp, new_identity) == {}
    assert not cp.exists()


def test_load_checkpoint_rejects_changed_model(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá"})

    other_settings = Settings(
        api_key="sk-test",
        model="gpt-4o",
        source_lang=settings.source_lang,
        target_lang=settings.target_lang,
        batch_size=settings.batch_size,
    )
    other_identity = build_identity(path, other_settings, compute_source_hash(path))
    assert load_checkpoint(cp, other_identity) == {}


def test_load_checkpoint_rejects_changed_language(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá"})

    other_settings = Settings(
        api_key="sk-test",
        model=settings.model,
        source_lang=settings.source_lang,
        target_lang="es",
        batch_size=settings.batch_size,
    )
    other_identity = build_identity(path, other_settings, compute_source_hash(path))
    assert load_checkpoint(cp, other_identity) == {}


def test_load_checkpoint_rejects_changed_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá"})

    other_settings = Settings(
        api_key="sk-test",
        model=settings.model,
        source_lang=settings.source_lang,
        target_lang=settings.target_lang,
        batch_size=settings.batch_size + 1,
    )
    other_identity = build_identity(path, other_settings, compute_source_hash(path))
    assert load_checkpoint(cp, other_identity) == {}


def test_load_checkpoint_handles_corrupted_file(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    cp.write_text("not valid json")

    assert load_checkpoint(cp, identity) == {}
    assert not cp.exists()


def test_remove_checkpoint_is_safe_for_missing_file(tmp_path: Path) -> None:
    remove_checkpoint(tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()


def test_load_checkpoint_rejects_unknown_id(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá", 999: "intruso"})

    assert load_checkpoint(cp, identity, {1, 2}) == {}
    assert not cp.exists()


def test_load_checkpoint_rejects_empty_translation(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá", 2: "   "})

    assert load_checkpoint(cp, identity, {1, 2}) == {}
    assert not cp.exists()


def test_load_checkpoint_accepts_partial_valid(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    save_checkpoint(cp, identity, {1: "Olá", 2: "mundo"})

    assert load_checkpoint(cp, identity, {1, 2, 3, 4, 5}) == {1: "Olá", 2: "mundo"}


def test_load_checkpoint_rejects_non_string_value(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    cp.write_text(
        json.dumps(
            {"identity": identity, "translations": {"1": "Olá", "2": 7}}
        ),
        encoding="utf-8",
    )

    assert load_checkpoint(cp, identity, {1, 2}) == {}
    assert not cp.exists()


def test_load_checkpoint_rejects_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "movie.en.srt"
    path.write_text("content")
    settings = _settings()
    identity = build_identity(path, settings, compute_source_hash(path))
    cp = tmp_path / "checkpoint.json"
    cp.write_text(
        json.dumps(
            {"identity": identity, "translations": {"1": "Olá", "01": "mundo"}}
        ),
        encoding="utf-8",
    )

    assert load_checkpoint(cp, identity, {1, 2}) == {}
    assert not cp.exists()
