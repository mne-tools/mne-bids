"""File I/O helpers with automatic cleanup of lock files."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, warn


def _get_lock_context(path):
    """Return a context manager that locks ``path`` if possible."""
    filelock = _soft_import(
        "filelock", purpose="parallel config set and get", strict=False
    )

    lock_context = contextlib.nullcontext()
    lock_path = Path(f"{os.fspath(path)}.lock")
    have_lock = False

    if filelock:
        try:
            lock_context = filelock.FileLock(lock_path, timeout=5)
            lock_context.acquire()
            have_lock = True
        except TimeoutError:
            warn(
                "Could not acquire lock file after 5 seconds, consider deleting it "
                f"if you know the corresponding file is usable:\n{lock_path}"
            )
            lock_context = contextlib.nullcontext()
        except OSError:
            warn(
                "Could not create lock file due to insufficient permissions. "
                "Proceeding without a lock."
            )
            lock_context = contextlib.nullcontext()

    return lock_context, lock_path, have_lock


@contextmanager
def _open_lock(path, *args, **kwargs):
    """Wrap :func:`mne.utils.config._open_lock` and remove stale ``.lock`` files."""
    lock_context, lock_path, have_lock = _get_lock_context(path)
    try:
        with lock_context, open(path, *args, **kwargs) as fid:
            yield fid
    finally:
        if have_lock and lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass


@contextmanager
def _file_lock(path):
    """Acquire a lock on ``path`` without opening the file."""
    lock_context, lock_path, have_lock = _get_lock_context(path)
    try:
        with lock_context:
            yield
    finally:
        if have_lock and lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
