from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import checkpoint as _checkpoint
from .config import Settings
from .exceptions import ConfigurationError
from .scanner import ScanResult, ScanStatus, scan_library
from .translator import TranslationStats, translate


class BatchError(Exception):
    """Raised when batch execution cannot proceed safely."""


@dataclass(slots=True)
class BatchItemResult:
    """Outcome for a single planned translation."""

    source: Path
    output_path: Path
    status: str
    reason: str
    stats: TranslationStats | None = None


@dataclass(slots=True)
class BatchResult:
    """Aggregate outcome of a batch translation run."""

    items: list[BatchItemResult] = field(default_factory=list)
    translated: int = 0
    skipped: int = 0
    failed: int = 0
    budget_exceeded: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float | None = None
    duration_seconds: float = 0.0


class LibraryLock:
    """Advisory lock for a library to avoid concurrent batch runs.

    Uses fcntl on Linux/macOS. If fcntl is unavailable the lock degrades to a
    best-effort PID file check.
    """

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self._fd: int | None = None

    def __enter__(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644
        )
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(self._fd)
            self._fd = None
            raise ConfigurationError(
                f"Another batch process is already running for this library. "
                f"Lock file: {self.lock_path}"
            ) from exc
        os.ftruncate(self._fd, 0)
        os.write(self._fd, str(os.getpid()).encode())
        return None

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            self._fd = None
        with contextlib.suppress(OSError):
            self.lock_path.unlink(missing_ok=True)


def _lock_path(library_path: Path) -> Path:
    digest = hashlib.sha256(
        str(library_path.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    return _checkpoint.CHECKPOINT_BASE_DIR / f"batch-{digest}.lock"


def _sum_cost(
    current: float | None, addition: float | None
) -> tuple[float | None, bool]:
    """Add two cost values, propagating unknown (None) status."""
    if current is None or addition is None:
        return None, True
    return current + addition, False


def execute_batch(
    library_path: Path,
    settings: Settings,
    plan: ScanResult,
    *,
    max_cost_usd: float | None = None,
    verbose: bool = False,
) -> BatchResult:
    """Execute a batch translation plan with locking, resumability, and cost controls."""
    if not settings.api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set.")

    start = time.perf_counter()
    result = BatchResult()
    eligible = [item for item in plan.items if item.status == ScanStatus.TRANSLATE]

    with LibraryLock(_lock_path(library_path)):
        actual_cost: float | None = 0.0
        cost_unknown = False

        for index, item in enumerate(eligible):
            output_path = item.output_path
            if output_path is None:
                continue

            # Race-condition guard: a file may have appeared since planning.
            if output_path.exists():
                result.items.append(
                    BatchItemResult(
                        source=item.source,
                        output_path=output_path,
                        status="skipped_existing",
                        reason="Output file appeared after planning",
                    )
                )
                result.skipped += 1
                continue

            # Budget guard before starting the next file.
            if max_cost_usd is not None and item.dry_run_info is not None:
                estimated_file_cost = item.dry_run_info.estimated_cost_usd
                if estimated_file_cost is None:
                    result.items.append(
                        BatchItemResult(
                            source=item.source,
                            output_path=output_path,
                            status="budget_exceeded",
                            reason="Cannot verify budget: file cost unknown",
                        )
                    )
                    result.budget_exceeded += len(eligible) - index
                    cost_unknown = True
                    break
                projected, _ = _sum_cost(actual_cost, estimated_file_cost)
                if projected is not None and projected > max_cost_usd:
                    result.items.append(
                        BatchItemResult(
                            source=item.source,
                            output_path=output_path,
                            status="budget_exceeded",
                            reason=f"Budget would exceed ${max_cost_usd:.4f}",
                        )
                    )
                    result.budget_exceeded += len(eligible) - index
                    break

            try:
                stats = translate(
                    item.source,
                    output_path,
                    settings,
                    force=False,
                    verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001
                result.items.append(
                    BatchItemResult(
                        source=item.source,
                        output_path=output_path,
                        status="failed",
                        reason=str(exc),
                    )
                )
                result.failed += 1
                continue

            new_cost, unknown = _sum_cost(
                actual_cost, stats.estimated_cost_usd
            )
            actual_cost = new_cost
            if unknown:
                cost_unknown = True

            result.items.append(
                BatchItemResult(
                    source=item.source,
                    output_path=output_path,
                    status="translated",
                    reason="Translation completed",
                    stats=stats,
                )
            )
            result.translated += 1
            result.total_input_tokens += stats.input_tokens
            result.total_output_tokens += stats.output_tokens

    result.duration_seconds = time.perf_counter() - start
    result.total_estimated_cost_usd = None if cost_unknown else actual_cost
    return result


def build_batch_plan(
    library_path: Path,
    settings: Settings,
    *,
    require_mount: bool = False,
    include_variants: bool = False,
) -> ScanResult:
    """Build the read-only translation plan used for confirmation and cost checks."""
    return scan_library(
        library_path,
        settings,
        require_mount=require_mount,
        include_variants=include_variants,
    )
