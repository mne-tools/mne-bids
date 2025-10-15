"""File I/O helpers with automatic cleanup of lock files."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from mne.utils import _soft_import, warn


def _normalize_lock_path(path):
    path = Path(path)
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


_LOCKED_PATHS: ContextVar[tuple[Path, ...]] = ContextVar(
    "_LOCKED_PATHS", default=tuple()
)


def _get_lock_context(path):
    """Return a context manager that locks ``path`` if possible.

    This uses file-based locking (filelock library) which works across
    different processes, unlike ContextVar which is process-local.
    """
    filelock = _soft_import(
        "filelock", purpose="parallel file I/O locking", strict=False
    )

    lock_context = contextlib.nullcontext()
    lock_path = Path(f"{os.fspath(path)}.lock")

    lock_path = _normalize_lock_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    have_lock = False
    backend = None

    if filelock:
        try:
            # Use FileLock (not SoftFileLock) for inter-process synchronization
            # SoftFileLock is more lenient but doesn't prevent concurrent writes
            # Timeout of 30 seconds prevents indefinite blocking
            lock_context = filelock.FileLock(lock_path, timeout=30)
            have_lock = True
            backend = "filelock"
        except OSError as exc:
            warn(f"Could not create lock file. Proceeding without a lock. ({exc})")
            lock_context = contextlib.nullcontext()

    return lock_context, lock_path, have_lock, backend


def _path_is_locked(path) -> bool:
    """Return True if ``path`` is currently locked via :func:`_file_lock`."""
    normalized = _normalize_lock_path(path)
    return normalized in _LOCKED_PATHS.get()


@contextmanager
def _open_lock(path, *args, **kwargs):
    """Wrap :func:`mne.utils.config._open_lock` and remove stale ``.lock`` files."""
    if _path_is_locked(path):
        with open(path, *args, **kwargs) as fid:
            yield fid
        return

    lock_context, lock_path, have_lock, backend = _get_lock_context(path)
    try:
        with lock_context, open(path, *args, **kwargs) as fid:
            yield fid
    finally:
        if have_lock:
            _cleanup_lock_file(lock_path, backend)


@contextmanager
def _file_lock(path):
    """Acquire a lock on ``path`` without opening the file."""
    normalized = _normalize_lock_path(path)
    lock_path = _normalize_lock_path(Path(f"{os.fspath(path)}.lock"))
    already_locked = _path_is_locked(normalized)
    if already_locked:
        lock_context, have_lock, backend = contextlib.nullcontext(), False, None
    else:
        lock_context, lock_path, have_lock, backend = _get_lock_context(path)
    token = None
    if have_lock and not already_locked:
        current = _LOCKED_PATHS.get()
        token = _LOCKED_PATHS.set(current + (normalized,))
    try:
        with lock_context:
            yield
    finally:
        if token is not None:
            _LOCKED_PATHS.reset(token)
        if have_lock:
            _cleanup_lock_file(lock_path, backend)


def _cleanup_lock_file(lock_path: Path, backend: str | None) -> None:
    """Attempt to remove ``lock_path`` once no other process holds it."""
    # Try to remove the lock file, but don't fail if it doesn't exist or is in use
    # In concurrent scenarios, multiple processes might try to clean up the same
    # lock file, so we need to be resilient to missing_ok scenarios
    import time

    # Try multiple times to delete the lock file with delays
    # This gives the filelock library time to fully release the lock
    for attempt in range(10):
        try:
            lock_path.unlink(missing_ok=False)
            return  # Successfully deleted
        except FileNotFoundError:
            # File doesn't exist, which is fine
            # (may have been deleted by another process)
            return
        except (OSError, PermissionError):
            # File is in use or we don't have permission
            if attempt < 9:
                # Try again after a short delay
                time.sleep(0.05)
            # On the last attempt, just give up silently
            pass
