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
    """Return a context manager that locks ``path`` if possible."""
    filelock = _soft_import(
        "filelock", purpose="parallel config set and get", strict=False
    )

    lock_context = contextlib.nullcontext()
    lock_path = Path(f"{os.fspath(path)}.lock")

    lock_path = _normalize_lock_path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    have_lock = False

    if filelock:
        try:
            # a FileLock instance is itself a context manager and blocks
            # until the lock becomes available (timeout=-1).
            lock_path.touch(exist_ok=True)
            lock_context = filelock.FileLock(lock_path, timeout=-1)
            have_lock = True
        except OSError as exc:
            warn(f"Could not create lock file. Proceeding without a lock. ({exc})")
            lock_context = contextlib.nullcontext()

    return lock_context, lock_path, have_lock


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

    lock_context, lock_path, have_lock = _get_lock_context(path)
    try:
        with lock_context, open(path, *args, **kwargs) as fid:
            yield fid
    finally:
        if have_lock:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def _file_lock(path):
    """Acquire a lock on ``path`` without opening the file."""
    normalized = _normalize_lock_path(path)
    lock_path = _normalize_lock_path(Path(f"{os.fspath(path)}.lock"))
    already_locked = _path_is_locked(normalized)
    if already_locked:
        lock_context, have_lock = contextlib.nullcontext(), False
    else:
        lock_context, lock_path, have_lock = _get_lock_context(path)
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
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
