"""File I/O helpers with file locking support."""

# Authors: The MNE-BIDS developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import inspect
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from mne.utils import _soft_import, logger, warn

_LOCK_TIMEOUT_FALLBACK = 60.0
_READ_MODES = (None, "r", "rt", "rb")
_REPLACE_MODES = ("w", "wt", "wb")
_env_lock_timeout = os.getenv("MNE_BIDS_FILELOCK_TIMEOUT", "")
try:
    DEFAULT_LOCK_TIMEOUT = (
        float(_env_lock_timeout) if _env_lock_timeout else _LOCK_TIMEOUT_FALLBACK
    )
    if DEFAULT_LOCK_TIMEOUT <= 0:
        raise ValueError
except ValueError:
    DEFAULT_LOCK_TIMEOUT = _LOCK_TIMEOUT_FALLBACK


def _canonical_lock_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute, normalised path without following symlinks.

    Symlinks are preserved so that the ``.lock`` companion file is written
    next to the symlink itself rather than its target. This matters for
    datalad/git-annex datasets, where the symlink target lives in a
    read-only ``.git/annex/objects/`` directory (see issue #1569).
    """
    path = Path(path)
    return Path(os.path.abspath(path.expanduser()))


@contextmanager
def _get_lock_context(path, *, timeout=None, lock=True):
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
        "filelock",
        purpose="parallel file I/O locking",
        strict=False,
        min_version="3.20.4",
    )

    lock_path = canonical_path.with_name(f"{canonical_path.name}.lock")
    lock_context = contextlib.nullcontext()

    where = None
    stack = "unknown"
    try:  # this should always work but let's be safe
        # [0] = here
        # [1] = contextlib __enter__
        # [2] = _open_lock
        # [3] = contextlib __enter__
        # [4] = caller of _open_lock
        # Using inspect.stack is expensive, so traverse directly instead
        where = inspect.currentframe().f_back.f_back.f_back.f_back
        stack = f"{where.f_code.co_filename}:{where.f_lineno} {where.f_code.co_name}"
    except Exception:
        pass
    finally:
        del where
    logger.debug(f"Lock: acquiring {canonical_path} from {stack}")

    if lock and filelock:
        try:
            # Ensure parent directory exists
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock_context = filelock.SoftFileLock(
                str(lock_path),
                timeout=timeout,
            )
        except (OSError, TypeError) as exp:
            # OSError: permission issues creating lock file
            # TypeError: invalid timeout parameter
            warn(
                f"Could not create lock at {lock_path} ({exp}); "
                "proceeding without a lock."
            )
    try:
        yield lock_context
    except Exception:
        logger.debug(f"Lock: exception {canonical_path} from {stack}")
        raise
    finally:
        logger.debug(f"Lock: released  {canonical_path} from {stack}")


def _has_text(path, text, encoding="utf-8"):
    """Whether ``path`` already holds exactly ``text``."""
    try:
        with open(path, encoding=encoding, newline="") as fid:
            return fid.read() == text.replace("\n", os.linesep)
    except (OSError, UnicodeError):
        return False


@contextmanager
def _replace_on_exit(path, fid, tmp, *, lock=True):
    """Move ``tmp`` over ``path`` once ``fid`` is written; drop it on failure.

    Readers see either the previous or the new file, never a partially written
    one. The new file gets the permissions of the file it replaces. On Windows a
    target held open by another process cannot be replaced and is overwritten in
    place, under the lock.
    """
    try:
        with fid:
            yield fid
        with contextlib.suppress(OSError):  # nothing to replace yet
            shutil.copymode(path, tmp)
        try:
            os.replace(tmp, path)
        except PermissionError:
            with _get_lock_context(path, lock=lock) as lock_context, lock_context:
                shutil.copyfile(tmp, path)
            os.unlink(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextmanager
def _open_lock(path, *args, lock_timeout=None, lock=True, **kwargs):
    """Context manager that acquires a file lock with optional file opening.

    If the `filelock` package is available, a lock is acquired on a lock file
    based on the given path (by appending '.lock'). Lock files are left behind
    to avoid race conditions during concurrent operations.

    The lock is re-entrant per process: nested calls for the same ``path`` will
    reuse the existing lock instead of attempting to acquire it again.

    If file opening arguments (*args, **kwargs) are provided, the file is opened
    in the specified mode. Otherwise, just the lock is acquired.

    Files opened for writing (``"w"``/``"wb"``) are replaced in one step (see
    :func:`_replace_on_exit`), so a reader can never observe a partially written
    file. Files opened read-only are therefore read without a lock, and so is a
    plain overwrite: only a sequence that reads a file and then writes it needs
    the lock, which its caller holds around the whole sequence.

    Parameters
    ----------
    path : str
        The path to acquire a lock for (and optionally open).
    *args : tuple
        Additional positional arguments forwarded to ``open``.
    lock_timeout : float | None
        Timeout in seconds for acquiring the lock. If ``None``, the default
        timeout applies.
    lock : bool
        If True, use a FileLock (if available) to lock the file.
    **kwargs : dict
        Additional keyword arguments forwarded to ``open``.

    Yields
    ------
    fid : file object or None
        File object if file opening args were provided, None otherwise.
    """
    canonical_path = _canonical_lock_path(path)
    mode = args[0] if args else kwargs.get("mode")
    if (args or kwargs) and mode in _READ_MODES:
        with open(canonical_path, *args, **kwargs) as fid:
            yield fid
        return
    # A symlink is written through, so datalad/git-annex datasets behave as before
    # (#1569), and so is a writable file in a read-only directory.
    if mode in _REPLACE_MODES and not canonical_path.is_symlink():
        name = f".{canonical_path.name}.{os.urandom(4).hex()}.tmp"
        tmp = canonical_path.with_name(name)
        try:
            fid = open(tmp, *args, **kwargs)
        except PermissionError:
            pass
        else:
            with _replace_on_exit(canonical_path, fid, tmp, lock=lock):
                yield fid
            return

    with _get_lock_context(
        canonical_path,
        timeout=lock_timeout,
        lock=lock,
    ) as lock_context:
        with contextlib.ExitStack() as stack:
            try:
                stack.enter_context(lock_context)
            except OSError as exp:
                warn(
                    f"Could not acquire lock for {canonical_path} "
                    f"({exp}); proceeding without a lock."
                )
            if args or kwargs:
                fid = stack.enter_context(open(canonical_path, *args, **kwargs))
                yield fid
            else:
                yield None


def _chmod_rw_R(path):
    assert os.path.isdir(path), f"Expected a directory, got {path}"
    os.chmod(path, os.stat(path).st_mode | 0o770)
    for root, dirs, files in os.walk(path):
        for name in files:
            os.chmod(os.path.join(root, name), 0o660)
        for name in dirs:
            os.chmod(os.path.join(root, name), 0o770)
