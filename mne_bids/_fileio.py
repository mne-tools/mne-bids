"""File I/O helpers with file locking support."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, warn


@contextmanager
def _open_lock(path, *args, **kwargs):
    """Context manager that opens a file with an optional file lock.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock'). The lock file is
    automatically removed after the lock is released.

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
        "filelock", purpose="parallel file I/O locking", strict=False
    )

    lock_path = f"{path}.lock"
    lock_context = contextlib.nullcontext()

    if filelock:
        try:
            # Ensure parent directory exists
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock_context = filelock.FileLock(lock_path, timeout=60)
        except (OSError, TypeError):
            # TypeError for invalid timeout
            warn("Could not create lock. Proceeding without a lock.")

    try:
        with lock_context:
            with open(path, *args, **kwargs) as fid:
                yield fid
    finally:
        # Lock files are left behind to avoid race conditions with concurrent processes.
        # They should be cleaned up explicitly after all parallel operations complete.
        pass


@contextmanager
def _file_lock(path):
    """Acquire a lock on ``path`` without opening the file.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock'). The lock file is
    automatically removed after the lock is released.

    Parameters
    ----------
    path : str
        The path to acquire a lock for.

    """
    filelock = _soft_import(
        "filelock", purpose="parallel file I/O locking", strict=False
    )

    lock_path = f"{path}.lock"
    lock_context = contextlib.nullcontext()

    if filelock:
        try:
            # Ensure parent directory exists
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock_context = filelock.FileLock(lock_path, timeout=60)
        except (OSError, TypeError):
            # TypeError for invalid timeout
            warn("Could not create lock. Proceeding without a lock.")

    try:
        with lock_context:
            yield
    finally:
        # Lock files are left behind to avoid race conditions with concurrent processes.
        # They should be cleaned up explicitly after all parallel operations complete.
        pass


def cleanup_lock_files(root_path):
    """Remove all .lock files in a directory tree.

    This function should be called after parallel operations complete to clean up
    lock files that may have been left behind.

    Parameters
    ----------
    root_path : str | Path
        Root directory to search for .lock files.
    """
    root_path = Path(root_path)
    if not root_path.exists():
        return

    # Find and remove all .lock files
    for lock_file in root_path.rglob("*.lock"):
        try:
            lock_file.unlink()
        except OSError:
            # If we can't remove it, skip it (might be in use by another process)
            pass
