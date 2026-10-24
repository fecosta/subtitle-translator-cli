from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from .exceptions import OutputExistsError, SubtitleError
from .models import SubtitleEntry
from .srt import load_subtitles, save_subtitles


def atomic_write_path(
    entries: list[SubtitleEntry], output_path: Path, *, force: bool = False
) -> None:
    """Write entries to a temporary file, validate it, then install output_path."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp.srt", prefix=output_path.stem + ".", dir=output_path.parent
    )
    os.close(fd)
    tmp = Path(tmp_path)

    try:
        save_subtitles(entries, tmp)
        # Validate the temporary SRT before exposing it.
        validated = load_subtitles(tmp)
        if len(validated) != len(entries):
            raise SubtitleError(
                f"Validation failed: expected {len(entries)} entries, got {len(validated)}"
            )
        _install(tmp, output_path, force=force)
    except Exception:
        _cleanup(tmp)
        raise


def _install(tmp: Path, output_path: Path, *, force: bool) -> None:
    """Atomically install the temporary file as the final output."""
    if force:
        os.replace(tmp, output_path)
        return

    # Prefer a hard link for atomic no-clobber installation.
    try:
        os.link(tmp, output_path)
    except FileExistsError as exc:
        raise OutputExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        ) from exc
    except OSError:
        # Some filesystems do not support hard links; reserve the filename
        # atomically with O_EXCL and then replace the placeholder.
        try:
            fd = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise OutputExistsError(
                f"Output file already exists: {output_path}. Use --force to overwrite."
            ) from None
        except OSError:
            # Final best-effort fallback (non-atomic).
            if output_path.exists():
                raise OutputExistsError(
                    f"Output file already exists: {output_path}. Use --force to overwrite."
                ) from None
            os.replace(tmp, output_path)
            return
        else:
            os.close(fd)
            try:
                os.replace(tmp, output_path)
            except OSError:
                with contextlib.suppress(OSError):
                    output_path.unlink(missing_ok=True)
                raise
            return

    tmp.unlink()


def _cleanup(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
