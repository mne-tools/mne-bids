"""File I/O helpers with file locking support."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, warn


@contextmanager
def _get_lock_context(path, timeout=5):
    """Get a file lock context for the given path.

    Internal helper function that creates a FileLock if available,
    or returns a nullcontext() as fallback.

    Parameters
    ----------
    path : str
        The path to acquire a lock for.
    timeout : float
        Timeout in seconds for acquiring the lock (default 60).

    Yields
    ------
    context : context manager
        Either a FileLock or nullcontext.
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
            lock_context = filelock.FileLock(lock_path, timeout=timeout)
        except (OSError, TypeError):
            # OSError: permission issues creating lock file
            # TypeError: invalid timeout parameter
            warn("Could not create lock. Proceeding without a lock.")

    yield lock_context


@contextmanager
def _open_lock(path, *args, **kwargs):
    """Context manager that acquires a file lock with optional file opening.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock'). Lock files are left behind
    to avoid race conditions during concurrent operations.

    If file opening arguments (*args, **kwargs) are provided, the file is opened
    in the specified mode. Otherwise, just the lock is acquired.

    Parameters
    ----------
    path : str
        The path to acquire a lock for (and optionally open).
    *args, **kwargs : optional
        Additional arguments and keyword arguments to be passed to the
        `open` function. If provided, the file will be opened. If empty,
        only the lock will be acquired.

    Yields
    ------
    fid : file object or None
        File object if file opening args were provided, None otherwise.

    """
    with _get_lock_context(path, timeout=5) as lock_context:
        try:
            with lock_context:
                if args or kwargs:
                    # File opening arguments provided - open the file
                    with open(path, *args, **kwargs) as fid:
                        yield fid
                else:
                    # No file opening arguments - just yield the lock context
                    yield None
        finally:
            # Lock files are left behind to avoid race conditions with concurrent
            # processes. They should be cleaned up explicitly after all parallel
            # operations complete via cleanup_lock_files().
            cleanup_lock_files(path)


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
