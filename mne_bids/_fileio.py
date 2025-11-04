"""File I/O helpers with file locking support."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import inspect
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, logger, warn

_LOCK_TIMEOUT_FALLBACK = 60.0
_env_lock_timeout = os.getenv("MNE_BIDS_FILELOCK_TIMEOUT", "")
try:
    DEFAULT_LOCK_TIMEOUT = (
        float(_env_lock_timeout) if _env_lock_timeout else _LOCK_TIMEOUT_FALLBACK
    )
    if DEFAULT_LOCK_TIMEOUT <= 0:
        raise ValueError
except ValueError:
    DEFAULT_LOCK_TIMEOUT = _LOCK_TIMEOUT_FALLBACK

_ACTIVE_LOCKS: dict[str, int] = {}
_ACTIVE_LOCKS_GUARD = threading.RLock()


def _canonical_lock_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute, normalised path without requiring it to exist."""
    return Path(path).expanduser().resolve(strict=False)


@contextmanager
def _get_lock_context(path, timeout=None):
    """Get a file lock context for the given path.

    Internal helper function that creates a FileLock if available,
    or returns a nullcontext() as fallback.

    Parameters
    ----------
    path : str
        The path to acquire a lock for.
    timeout : float
        Timeout in seconds for acquiring the lock. If ``None``, the value of
        ``DEFAULT_LOCK_TIMEOUT`` (default 60 seconds) is used. The timeout can
        be overridden via the ``MNE_BIDS_FILELOCK_TIMEOUT`` environment
        variable.

    Yields
    ------
    context : context manager
        Either a FileLock or nullcontext.
    """
    if timeout is None:
        timeout = DEFAULT_LOCK_TIMEOUT

    canonical_path = _canonical_lock_path(path)

    filelock = _soft_import(
        "filelock", purpose="parallel file I/O locking", strict=False
    )

    lock_path = canonical_path.with_name(f"{canonical_path.name}.lock")
    lock_context = contextlib.nullcontext()

    stack = "unknown"
    try:  # this should always work but let's be safe
        # [0] = here
        # [1] = contextlib __enter__
        # [2] = _open_lock
        # [3] = contextlib __enter__
        # [4] = caller of _open_lock
        where = inspect.stack()[4]
        stack = f"{where.filename}:{where.lineno} {where.function}"
        del where
    except Exception:
        pass
    logger.debug(f"Lock: acquiring {canonical_path} from {stack}")

    if filelock:
        try:
            # Ensure parent directory exists
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock_context = filelock.FileLock(
                str(lock_path),
                timeout=timeout,
            )
        except (OSError, TypeError):
            # OSError: permission issues creating lock file
            # TypeError: invalid timeout parameter
            warn("Could not create lock. Proceeding without a lock.")
    try:
        yield lock_context
    except Exception:
        logger.debug(f"Lock: exception {canonical_path} from {stack}")
        raise
    finally:
        logger.debug(f"Lock: released  {canonical_path} from {stack}")


@contextmanager
def _open_lock(path, *args, lock_timeout=None, **kwargs):
    """Context manager that acquires a file lock with optional file opening.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock'). Lock files are left behind
    to avoid race conditions during concurrent operations.

    The lock is re-entrant per process: nested calls for the same ``path`` will
    reuse the existing lock instead of attempting to acquire it again.

    If file opening arguments (*args, **kwargs) are provided, the file is opened
    in the specified mode. Otherwise, just the lock is acquired.

    Parameters
    ----------
    path : str
        The path to acquire a lock for (and optionally open).
    *args : tuple
        Additional positional arguments forwarded to ``open``.
    lock_timeout : float | None
        Timeout in seconds for acquiring the lock. If ``None``, the default
        timeout applies.
    **kwargs : dict
        Additional keyword arguments forwarded to ``open``.

    Yields
    ------
    fid : file object or None
        File object if file opening args were provided, None otherwise.
    """
    canonical_path = _canonical_lock_path(path)
    lock_key = str(canonical_path)

    with _ACTIVE_LOCKS_GUARD:
        lock_depth = _ACTIVE_LOCKS.get(lock_key, 0)
        _ACTIVE_LOCKS[lock_key] = lock_depth + 1
    is_reentrant = lock_depth > 0

    try:
        if is_reentrant:
            if args or kwargs:
                with open(canonical_path, *args, **kwargs) as fid:
                    yield fid
            else:
                yield None
            return

        # Increment multiprocess refcount before acquiring lock
        _increment_lock_refcount(canonical_path)

        with _get_lock_context(
            canonical_path,
            timeout=lock_timeout,
        ) as lock_context:
            with lock_context:
                if args or kwargs:
                    with open(canonical_path, *args, **kwargs) as fid:
                        yield fid
                else:
                    yield None
    finally:
        with _ACTIVE_LOCKS_GUARD:
            _ACTIVE_LOCKS[lock_key] -= 1
            if _ACTIVE_LOCKS[lock_key] == 0:
                del _ACTIVE_LOCKS[lock_key]

        # Clean up lock files safely using reference counting across processes.
        # Only clean up when this was the outermost lock (not re-entrant) and
        # when the lock depth was 0 (meaning we actually acquired the lock).
        if not is_reentrant and lock_depth == 0:
            _decrement_and_cleanup_lock_file(canonical_path)


def _increment_lock_refcount(file_path):
    """Increment the multiprocess reference count for a lock file.

    Parameters
    ----------
    file_path : Path
        The original file path (not the lock file path).
    """
    file_path = Path(file_path)
    refcount_path = file_path.parent / f"{file_path.name}.lock.refcount"
    refcount_lock_path = Path(f"{refcount_path}.lock")

    filelock = _soft_import("filelock", purpose="lock refcounting", strict=False)
    if not filelock:
        return

    refcount_lock = filelock.FileLock(str(refcount_lock_path), timeout=5.0)
    try:
        with refcount_lock:
            # Read current refcount or initialize to 0
            current_count = 0
            if refcount_path.exists():
                try:
                    current_count = int(refcount_path.read_text().strip())
                except (ValueError, OSError):
                    logger.debug(f"Could not read refcount from {refcount_path}")

            # Increment and write back
            current_count += 1
            try:
                refcount_path.write_text(str(current_count))
            except OSError as exp:
                logger.debug(f"Could not write refcount to {refcount_path}: {exp}")
    except TimeoutError:
        # Another process is updating refcount concurrently
        logger.debug(f"Timeout acquiring refcount lock for {file_path}")
    except Exception as exp:
        logger.debug(f"Error incrementing refcount for {file_path}: {exp}")
    finally:
        # Clean up the refcount lock file
        try:
            if refcount_lock_path.exists():
                refcount_lock_path.unlink()
        except OSError:
            pass


def _decrement_and_cleanup_lock_file(file_path):
    """Decrement refcount and remove lock file if no longer in use.

    Maintains a reference count in a .refcount file to track how many processes
    are currently using or waiting for the lock. Only deletes the lock file when
    the reference count reaches zero.

    Parameters
    ----------
    file_path : Path
        The original file path (not the lock file path).
    """
    from pathlib import Path

    file_path = Path(file_path)
    lock_path = file_path.parent / f"{file_path.name}.lock"
    refcount_path = file_path.parent / f"{file_path.name}.lock.refcount"
    refcount_lock_path = Path(f"{refcount_path}.lock")

    # Early return if lock file doesn't exist
    if not lock_path.exists():
        # Clean up orphaned refcount file if it exists
        try:
            if refcount_path.exists():
                refcount_path.unlink()
        except OSError:
            pass
        return

    filelock = _soft_import("filelock", purpose="lock refcounting", strict=False)
    if not filelock:
        return

    refcount_lock = filelock.FileLock(str(refcount_lock_path), timeout=5.0)
    try:
        with refcount_lock:
            # Read current refcount or initialize to 0
            current_count = 0
            if refcount_path.exists():
                try:
                    current_count = int(refcount_path.read_text().strip())
                except (ValueError, OSError):
                    logger.debug(f"Could not read refcount from {refcount_path}")

            # Decrement refcount, ensuring it doesn't go negative
            current_count = max(0, current_count - 1)

            if current_count == 0:
                # No more processes using this lock, safe to delete
                try:
                    lock_path.unlink()
                except OSError as exp:
                    logger.debug(f"Could not remove lock file {lock_path}: {exp}")
                try:
                    if refcount_path.exists():
                        refcount_path.unlink()
                except OSError as exp:
                    logger.debug(f"Could not remove refcount {refcount_path}: {exp}")
            else:
                # Write back decremented count
                try:
                    refcount_path.write_text(str(current_count))
                except OSError as exp:
                    logger.debug(f"Could not write refcount to {refcount_path}: {exp}")
    except TimeoutError:
        # Another process is updating refcount, it will handle cleanup
        logger.debug(f"Timeout acquiring refcount lock for {file_path}")
    except Exception as exp:
        logger.debug(f"Error decrementing refcount for {file_path}: {exp}")
    finally:
        # Clean up the refcount lock file
        try:
            if refcount_lock_path.exists():
                refcount_lock_path.unlink()
        except OSError:
            pass


def cleanup_lock_files(root_path):
    """Remove lock files associated with a path or an entire tree.

    Parameters
    ----------
    root_path : str | Path
        Root directory or file path used to derive lock file locations.
    """
    root_path = Path(root_path)

    if root_path.is_dir():
        for lock_file in root_path.rglob("*.lock"):
            try:
                lock_file.unlink()
            except OSError:
                pass
        return

    lock_candidate = root_path.parent / f"{root_path.name}.lock"
    if lock_candidate.exists():
        try:
            lock_candidate.unlink()
        except OSError:
            pass
