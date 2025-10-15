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
            # Attempt to acquire the file lock with a 30-second timeout.
            # The timeout is for acquiring the lock (not for holding it).
            # 30 seconds was chosen as a balance between responsiveness and allowing
            # for slow file operations. If the lock cannot be acquired within this
            # period, an exception is raised and we proceed without a lock.
            lock_obj = filelock.FileLock(lock_path, timeout=30)
            lock_context = _FileLockContext(lock_obj)
            have_lock = True
            backend = "filelock"
        except OSError as exc:
            warn(f"Could not create lock file. Proceeding without a lock. ({exc})")
            lock_context = contextlib.nullcontext()

    return lock_context, lock_path, have_lock, backend


class _FileLockContext:
    """A wrapper around FileLock that handles acquisition/release safely."""

    def __init__(self, lock_obj):
        """Initialize with a filelock.FileLock object."""
        self.lock_obj = lock_obj
        self._is_locked = False

    def __enter__(self):
        """Acquire the lock."""
        try:
            self.lock_obj.acquire()
            self._is_locked = True
        except (OSError, getattr(self.lock_obj, "Timeout", type(None))):
            # If we can't acquire the lock, log a warning and continue without it
            warn(
                f"Could not acquire file lock for {self.lock_obj.lock_file}. Proceeding without a lock."
            )
            self._is_locked = False
        return self

    def __exit__(self, *args):
        """Release the lock if we acquired it."""
        if self._is_locked:
            try:
                # Use force=False to ensure we only release the lock if it was acquired
                # by this process. Setting force=True would forcibly release the lock,
                # which can be unsafe if other processes are still using it. Here,
                # force=False is correct to avoid interfering with other processes.
                self.lock_obj.release(force=False)
            except Exception:
                # Ignore errors during release
                pass
            finally:
                self._is_locked = False


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
    with lock_context, open(path, *args, **kwargs) as fid:
        yield fid


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
