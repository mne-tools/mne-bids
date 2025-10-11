"""File I/O helpers with automatic cleanup of lock files."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, warn


@contextlib.contextmanager
def _mne_open_lock(path, *args, **kwargs):
    """
    Context manager that opens a file with an optional file lock.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock').

    Otherwise, a null context is used. The path is then opened in the
    specified mode.

    Parameters
    ----------
    path : str
        The path to the file to be opened.
    *args, **kwargs : optional
        Additional arguments and keyword arguments to be passed to the
        `open` function.

    """
    filelock = _soft_import(
        "filelock", purpose="parallel config set and get", strict=False
    )

    lock_context = contextlib.nullcontext()  # default to no lock

    if filelock:
        lock_path = f"{path}.lock"
        try:
            lock_context = filelock.FileLock(lock_path, timeout=5)
            lock_context.acquire()
        except TimeoutError:
            warn(
                "Could not acquire lock file after 5 seconds, consider deleting it "
                f"if you know the corresponding file is usable:\n{lock_path}"
            )
            lock_context = contextlib.nullcontext()

    with lock_context, open(path, *args, **kwargs) as fid:
        yield fid


@contextmanager
def _open_lock(path, *args, **kwargs):
    """Wrap :func:`mne.utils.config._open_lock` and remove stale ``.lock`` files."""
    lock_path = Path(f"{os.fspath(path)}.lock")
    try:
        with _mne_open_lock(path, *args, **kwargs) as fid:
            yield fid
    finally:
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
