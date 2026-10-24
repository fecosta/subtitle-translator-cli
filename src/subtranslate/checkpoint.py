from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import Settings

CHECKPOINT_VERSION = 2


def compute_source_hash(path: Path) -> str:
    """Return a SHA-256 hex digest of the file contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _settings_hash(identity: dict[str, Any]) -> str:
    """Return a short hash of the translation-affecting settings."""
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


CHECKPOINT_BASE_DIR: Path = Path.home() / ".cache" / "subtranslate" / "checkpoints"


def checkpoint_path(
    input_path: Path,
    settings: Settings,
    source_hash: str | None = None,
) -> Path:
    """Return the local checkpoint file path for a given input and settings."""
    base = CHECKPOINT_BASE_DIR
    base.mkdir(parents=True, exist_ok=True)
    if source_hash is None:
        source_hash = compute_source_hash(input_path)
    identity = build_identity(input_path, settings, source_hash)
    return base / f"{source_hash}-{_settings_hash(identity)}.json"


def build_identity(input_path: Path, settings: Settings, source_hash: str) -> dict[str, Any]:
    """Build the checkpoint identity used to validate reuse."""
    return {
        "version": CHECKPOINT_VERSION,
        "source_hash": source_hash,
        "source_path": str(input_path.resolve()),
        "source_lang": settings.source_lang,
        "target_lang": settings.target_lang,
        "model": settings.model,
        "batch_size": settings.batch_size,
    }


def _identity_matches(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check that stored checkpoint identity matches the expected configuration."""
    keys = ("version", "source_hash", "source_lang", "target_lang", "model", "batch_size")
    return all(stored.get(key) == expected.get(key) for key in keys)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file atomically using a temporary file and os.replace."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp.json", prefix=path.stem + ".", dir=path.parent
    )
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    identity: dict[str, Any],
    expected_ids: set[int] | None = None,
) -> dict[int, str]:
    """Load a checkpoint if it exists and matches the current identity.

    If ``expected_ids`` is provided, translations with unknown identifiers or
    empty text cause the checkpoint to be discarded.
    """
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Checkpoint is not a JSON object.")
        if not _identity_matches(data.get("identity", {}), identity):
            raise ValueError("Checkpoint identity does not match.")
        raw_translations = data.get("translations", {})
        if not isinstance(raw_translations, dict):
            raise ValueError("Checkpoint translations are not a mapping.")

        translations: dict[int, str] = {}
        for key, value in raw_translations.items():
            if not isinstance(key, str):
                raise ValueError("Checkpoint translation key is not a string.")
            try:
                entry_id = int(key)
            except ValueError as exc:
                raise ValueError(f"Invalid checkpoint identifier: {key}") from exc
            if not isinstance(value, str):
                raise ValueError(f"Checkpoint translation for id {entry_id} is not a string.")
            if entry_id in translations:
                raise ValueError(f"Duplicate checkpoint identifier: {entry_id}")
            translations[entry_id] = value

        if expected_ids is not None:
            if not set(translations.keys()) <= expected_ids:
                raise ValueError("Checkpoint contains unexpected identifiers.")
            if not all(text.strip() for text in translations.values()):
                raise ValueError("Checkpoint contains empty translations.")
        return translations
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return {}


def save_checkpoint(
    path: Path,
    identity: dict[str, Any],
    translations: dict[int, str],
) -> None:
    """Atomically save translations and identity to the checkpoint file."""
    data = {
        "identity": identity,
        "translations": {str(key): value for key, value in translations.items()},
    }
    _atomic_write_json(path, data)


def remove_checkpoint(path: Path) -> None:
    """Delete the checkpoint file if it exists."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
